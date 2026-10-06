"""
クロップ点群からの 2D グリッドマップ生成および幾何オブジェクト (外壁/机/バケツ) 抽出モジュール
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from pcd_map_generator.coordinates import BoundingBox2D, CoordinateTransformMeta


def _generate_outer_walls(
    bbox: BoundingBox2D,
    wall_thickness_m: float = 0.024,
    wall_height_max_m: float = 0.150
) -> List[Dict[str, Any]]:
    """選択領域の外周四方を囲む立上りフェンス (外壁) オブジェクト定義を生成

    Args:
        bbox: 選択バウンディングボックス
        wall_thickness_m: 壁の厚み [m] (デフォルト: 24mm)
        wall_height_max_m: 立上りフェンスの高さ上限 [m] (デフォルト: 150mm)

    Returns:
        List[Dict[str, Any]]: gn10-pointcloud-localization 形式の外壁辞書リスト
    """
    center_x = (bbox.min_x_m + bbox.max_x_m) / 2.0
    center_y = (bbox.min_y_m + bbox.max_y_m) / 2.0
    half_x = (bbox.max_x_m - bbox.min_x_m) / 2.0
    half_y = (bbox.max_y_m - bbox.min_y_m) / 2.0

    outer_walls = [
        ("外壁 (上) 立上り", center_x, bbox.max_y_m, half_x, wall_thickness_m / 2.0),
        ("外壁 (下) 立上り", center_x, bbox.min_y_m, half_x, wall_thickness_m / 2.0),
        ("外壁 (左) 立上り", bbox.min_x_m, center_y, wall_thickness_m / 2.0, half_y),
        ("外壁 (右) 立上り", bbox.max_x_m, center_y, wall_thickness_m / 2.0, half_y),
    ]

    wall_objects = []
    for wall_name, wx, wy, p1, p2 in outer_walls:
        wall_objects.append({
            "comment": wall_name,
            "type": "BOX",
            "x": round(wx, 3),
            "y": round(wy, 3),
            "z_min": 0.024,
            "z_max": round(wall_height_max_m, 3),
            "param1": round(p1, 3),
            "param2": round(p2, 3),
        })
    return wall_objects


def extract_field_objects_and_create_maps(
    cropped_points: np.ndarray,
    bbox: BoundingBox2D,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float,
    generate_outer_walls: bool = True
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, CoordinateTransformMeta]:
    """クロップ点群から 2D 占有格子マップを生成し、輪郭検出で障害物オブジェクトを抽出

    Args:
        cropped_points: クロップ済み点群 (N, 3)
        bbox: 選択バウンディングボックス
        floor_z_m: 床面の Z 座標 [m]
        ground_margin_m: 床面ノイズ除外マージン [m]
        robot_height_m: ロボット全高上限 [m]
        resolution_m: グリッド解像度 [m/pixel]
        generate_outer_walls: 外周壁を自動生成するかどうか

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, CoordinateTransformMeta]:
            - field_objects: gn10-pointcloud-localization 互換オブジェクトリスト
            - closed_obstacle_grid: モルフォロジー処理済み 2D 占有バイナリグリッド
            - preview_bgr_image: 認識オブジェクト枠線を描画したプレビュー画像
            - meta: クロップ領域の座標変換メタデータ
    """
    z_min_filter = floor_z_m + ground_margin_m
    z_max_filter = floor_z_m + robot_height_m

    obstacle_mask = (cropped_points[:, 2] >= z_min_filter) & (cropped_points[:, 2] <= z_max_filter)
    obstacle_points = cropped_points[obstacle_mask]

    width_pixel = int(np.ceil((bbox.max_x_m - bbox.min_x_m) / resolution_m))
    height_pixel = int(np.ceil((bbox.max_y_m - bbox.min_y_m) / resolution_m))

    meta = CoordinateTransformMeta(
        resolution_m=resolution_m,
        min_x_m=bbox.min_x_m,
        max_x_m=bbox.max_x_m,
        min_y_m=bbox.min_y_m,
        max_y_m=bbox.max_y_m,
        image_width_pixel=width_pixel,
        image_height_pixel=height_pixel
    )

    binary_obstacle_grid = np.zeros((height_pixel, width_pixel), dtype=np.uint8)
    cols = ((obstacle_points[:, 0] - bbox.min_x_m) / resolution_m).astype(int)
    rows = ((bbox.max_y_m - obstacle_points[:, 1]) / resolution_m).astype(int)
    valid = (cols >= 0) & (cols < width_pixel) & (rows >= 0) & (rows < height_pixel)
    binary_obstacle_grid[rows[valid], cols[valid]] = 255

    # モルフォロジー演算 (Closing) で微小点群ノイズを繋ぎ、物体化
    kernel_size_pixel = max(3, int(0.10 / resolution_m))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_pixel, kernel_size_pixel))
    closed_obstacle_grid = cv2.morphologyEx(binary_obstacle_grid, cv2.MORPH_CLOSE, kernel)

    field_objects: List[Dict[str, Any]] = []
    preview_bgr_image = cv2.cvtColor(closed_obstacle_grid, cv2.COLOR_GRAY2BGR)

    # 1. 外周壁 (選択範囲の外枠) を自動生成
    if generate_outer_walls:
        field_objects.extend(_generate_outer_walls(bbox))

    # 2. 内部の障害物（机、バケツなど）を輪郭検出
    inner_mask = np.copy(closed_obstacle_grid)
    # 外枠 2 ピクセルをゼロクリア（外枠境界の巻き込みを防止）
    inner_mask[0:2, :] = 0
    inner_mask[-2:, :] = 0
    inner_mask[:, 0:2] = 0
    inner_mask[:, -2:] = 0

    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_pixel = (0.15 / resolution_m) * (0.15 / resolution_m)  # 15cm x 15cm 未満のノイズは無視
    object_index = 1

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area_pixel:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0.0

        # マスクを作成して、この輪郭内の点の Z 範囲を取得
        contour_mask = np.zeros((height_pixel, width_pixel), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        point_in_contour = contour_mask[rows[valid], cols[valid]] == 255
        matched_z = obstacle_points[valid][point_in_contour, 2]

        if len(matched_z) > 0:
            obj_z_min = float(np.percentile(matched_z, 5) - floor_z_m)
            obj_z_max = float(np.percentile(matched_z, 95) - floor_z_m)
        else:
            obj_z_min = 0.024
            obj_z_max = 0.76

        obj_z_min = max(0.0, obj_z_min)
        obj_z_max = max(obj_z_min + 0.05, obj_z_max)

        # 形状判定 (円形度 >= 0.82 は円柱、それ以外は直方体)
        if circularity >= 0.82:
            # 円柱 (CYLINDER)
            (circle_col, circle_row), circle_radius_pixel = cv2.minEnclosingCircle(contour)
            center_x_m, center_y_m = meta.pixel_to_world(int(circle_col), int(circle_row))
            radius_m = circle_radius_pixel * resolution_m

            comment_str = f"円柱/バケツ {object_index} (半径:{radius_m:.2f}m)"
            field_object = {
                "comment": comment_str,
                "type": "CYLINDER",
                "x": round(center_x_m, 3),
                "y": round(center_y_m, 3),
                "z_min": round(obj_z_min, 3),
                "z_max": round(obj_z_max, 3),
                "param1": round(radius_m, 4),
                "param2": 0.0
            }
            cv2.circle(preview_bgr_image, (int(circle_col), int(circle_row)), int(circle_radius_pixel), (0, 0, 255), 2)
            cv2.putText(preview_bgr_image, f"[{object_index}] Cyl", (int(circle_col) + 5, int(circle_row)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        else:
            # 直方体 (BOX)
            rot_rect = cv2.minAreaRect(contour)
            (rect_col, rect_row), (rect_w_pixel, rect_h_pixel), rect_angle_deg = rot_rect
            center_x_m, center_y_m = meta.pixel_to_world(int(rect_col), int(rect_row))

            half_width_m = (rect_w_pixel * resolution_m) / 2.0
            half_height_m = (rect_h_pixel * resolution_m) / 2.0

            aspect_ratio = max(half_width_m, half_height_m) / max(0.01, min(half_width_m, half_height_m))

            if aspect_ratio >= 4.0:
                comment_str = f"仕切り板/構造物 {object_index}"
                draw_color = (255, 120, 0)
            else:
                comment_str = f"机/台座 {object_index}"
                draw_color = (0, 255, 0)

            field_object = {
                "comment": comment_str,
                "type": "BOX",
                "x": round(center_x_m, 3),
                "y": round(center_y_m, 3),
                "z_min": round(obj_z_min, 3),
                "z_max": round(obj_z_max, 3),
                "param1": round(half_width_m, 3),
                "param2": round(half_height_m, 3)
            }
            box_points = cv2.boxPoints(rot_rect)
            box_points = np.int32(box_points)
            cv2.drawContours(preview_bgr_image, [box_points], 0, draw_color, 2)
            cv2.putText(preview_bgr_image, f"[{object_index}] Box", (int(rect_col) + 5, int(rect_row)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1)

        field_objects.append(field_object)
        object_index += 1

    return field_objects, closed_obstacle_grid, preview_bgr_image, meta
