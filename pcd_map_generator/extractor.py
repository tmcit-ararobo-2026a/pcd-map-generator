"""
クロップ点群からの 2D グリッドマップ生成およびフィールド幾何オブジェクト (外壁/指定障害物) 生成モジュール
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pcd_map_generator.coordinates import BoundingBox2D, CoordinateTransformMeta


def generate_outer_walls(
    bbox: BoundingBox2D,
    wall_thickness_m: float = 0.024,
    wall_height_max_m: float = 0.150,
    include_visual_foot: bool = True
) -> List[Dict[str, Any]]:
    """選択領域の外周四方を囲む立上りフェンス (および土台) オブジェクト定義を生成

    高専ロボコン/NHK学生ロボコンの CAD 規格（150mm L字断面・木材厚24mm）に準拠:
    - 立上り部 (BOX): LiDAR 点群と照合される 24mm 厚の木材 (Z: 0.024m 〜 0.150m)
    - 土台部 (VISUAL_BOX): 床面に置かれる 150mm 幅の土台 (Z: 0.000m 〜 0.024m)

    Args:
        bbox: 選択バウンディングボックス (競技エリアの内寸境界)
        wall_thickness_m: 立上り木材の厚み [m] (デフォルト: 24mm)
        wall_height_max_m: 立上りフェンスの高さ上限 [m] (デフォルト: 150mm)
        include_visual_foot: 表示専用の土台 (VISUAL_BOX) を含めるか

    Returns:
        List[Dict[str, Any]]: gn10-pointcloud-localization 形式の外壁定義リスト
    """
    center_x = (bbox.min_x_m + bbox.max_x_m) / 2.0
    center_y = (bbox.min_y_m + bbox.max_y_m) / 2.0
    half_x = (bbox.max_x_m - bbox.min_x_m) / 2.0
    half_y = (bbox.max_y_m - bbox.min_y_m) / 2.0

    half_thickness = wall_thickness_m / 2.0
    foot_width_m = 0.150
    half_foot_width = foot_width_m / 2.0

    wall_objects: List[Dict[str, Any]] = []

    # 1. 表示用 土台 (VISUAL_BOX)
    if include_visual_foot:
        visual_foots = [
            ("外壁 (上) 土台", center_x, bbox.max_y_m + half_foot_width, half_x, half_foot_width),
            ("外壁 (下) 土台", center_x, bbox.min_y_m - half_foot_width, half_x, half_foot_width),
            ("外壁 (左) 土台", bbox.min_x_m - half_foot_width, center_y, half_foot_width, half_y),
            ("外壁 (右) 土台", bbox.max_x_m + half_foot_width, center_y, half_foot_width, half_y),
        ]
        for name, wx, wy, p1, p2 in visual_foots:
            wall_objects.append({
                "comment": name,
                "type": "VISUAL_BOX",
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z_min": 0.0,
                "z_max": 0.024,
                "param1": round(p1, 3),
                "param2": round(p2, 3),
            })

    # 2. LiDAR 照合用 立上りフェンス (BOX)
    matching_uprights = [
        ("外壁 (上) 立上り", center_x, bbox.max_y_m + half_thickness, half_x, half_thickness),
        ("外壁 (下) 立上り", center_x, bbox.min_y_m - half_thickness, half_x, half_thickness),
        ("外壁 (左) 立上り", bbox.min_x_m - half_thickness, center_y, half_thickness, half_y),
        ("外壁 (右) 立上り", bbox.max_x_m + half_thickness, center_y, half_thickness, half_y),
    ]
    for name, wx, wy, p1, p2 in matching_uprights:
        wall_objects.append({
            "comment": name,
            "type": "BOX",
            "x": round(wx, 3),
            "y": round(wy, 3),
            "z_min": 0.024,
            "z_max": round(wall_height_max_m, 3),
            "param1": round(p1, 3),
            "param2": round(p2, 3),
        })

    return wall_objects


def create_occupancy_grid_and_meta(
    cropped_points: np.ndarray,
    bbox: BoundingBox2D,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float
) -> Tuple[np.ndarray, CoordinateTransformMeta, np.ndarray]:
    """クロップ点群から 2D 点群プレビュー用グリッドと座標メタデータを生成

    Args:
        cropped_points: クロップ済み点群 (N, 3)
        bbox: 選択バウンディングボックス
        floor_z_m: 床面の Z 座標 [m]
        ground_margin_m: 床面ノイズ除外マージン [m]
        robot_height_m: ロボット全高上限 [m]
        resolution_m: グリッド解像度 [m/pixel]

    Returns:
        Tuple[np.ndarray, CoordinateTransformMeta, np.ndarray]:
            - closed_obstacle_grid: 点群可視化用の 2D バイナリグリッド
            - meta: 座標変換メタデータ
            - obstacle_points: 高さフィルタ通過点群
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
    if len(obstacle_points) > 0:
        cols = ((obstacle_points[:, 0] - bbox.min_x_m) / resolution_m).astype(int)
        rows = ((bbox.max_y_m - obstacle_points[:, 1]) / resolution_m).astype(int)
        valid = (cols >= 0) & (cols < width_pixel) & (rows >= 0) & (rows < height_pixel)
        binary_obstacle_grid[rows[valid], cols[valid]] = 255

    kernel_size_pixel = max(3, int(0.10 / resolution_m))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_pixel, kernel_size_pixel))
    closed_obstacle_grid = cv2.morphologyEx(binary_obstacle_grid, cv2.MORPH_CLOSE, kernel)

    return closed_obstacle_grid, meta, obstacle_points


def create_clean_nav2_map(
    meta: CoordinateTransformMeta,
    field_objects: List[Dict[str, Any]],
    wall_thickness_m: float = 0.05
) -> np.ndarray:
    """内部を完全な白（フリースペース）にし、外壁と登録障害物のみを正確に描画した Nav2 用グリッドを生成

    Args:
        meta: 座標変換メタデータ
        field_objects: 登録オブジェクト (外壁 + 手動追加障害物)
        wall_thickness_m: 外壁フェンスの描画厚み [m]

    Returns:
        np.ndarray: クリーンな Nav2 用バイナリグリッド (255: 障害物, 0: フリースペース)
    """
    # 内部はすべて 0 (フリースペース = 白床)
    clean_grid = np.zeros((meta.image_height_pixel, meta.image_width_pixel), dtype=np.uint8)

    # 1. 外枠四辺に真っ直ぐな外壁 (255 = 障害物) を描画
    thickness_px = max(1, int(wall_thickness_m / meta.resolution_m))
    cv2.rectangle(
        clean_grid,
        (0, 0),
        (meta.image_width_pixel - 1, meta.image_height_pixel - 1),
        255,
        thickness=thickness_px
    )

    # 2. 手動追加された障害物 (机・バケツなど) を正確な幾何形状で描画
    for obj in field_objects:
        obj_type = obj.get("type", "BOX")
        # 外壁は既に四辺に描画済みなのでスキップ
        if "外壁" in obj.get("comment", "") or obj_type == "VISUAL_BOX":
            continue

        cx, cy = obj["x"], obj["y"]
        col_c, row_c = meta.world_to_pixel(cx, cy)

        if obj_type == "CYLINDER":
            radius_px = max(1, int(obj["param1"] / meta.resolution_m))
            cv2.circle(clean_grid, (col_c, row_c), radius_px, 255, -1)  # 塗りつぶし
        elif obj_type == "BOX":
            hw_px = max(1, int(obj["param1"] / meta.resolution_m))
            hd_px = max(1, int(obj["param2"] / meta.resolution_m))
            p1 = (max(0, col_c - hw_px), max(0, row_c - hd_px))
            p2 = (min(meta.image_width_pixel - 1, col_c + hw_px), min(meta.image_height_pixel - 1, row_c + hd_px))
            cv2.rectangle(clean_grid, p1, p2, 255, -1)  # 塗りつぶし

    return clean_grid


def draw_objects_on_preview(
    preview_bgr_image: np.ndarray,
    meta: CoordinateTransformMeta,
    field_objects: List[Dict[str, Any]]
) -> np.ndarray:
    """プレビュー画像上に登録オブジェクトの枠線とラベルを描画"""
    output_image = preview_bgr_image.copy()

    for idx, obj in enumerate(field_objects, 1):
        obj_type = obj.get("type", "BOX")
        cx, cy = obj["x"], obj["y"]
        col_c, row_c = meta.world_to_pixel(cx, cy)

        label_type = "Cyl" if obj_type == "CYLINDER" else "Box"
        if "外壁" in obj.get("comment", ""):
            label_type = "Wall"

        if obj_type == "CYLINDER":
            radius_pixel = int(obj["param1"] / meta.resolution_m)
            cv2.circle(output_image, (col_c, row_c), radius_pixel, (0, 0, 255), 2)
            cv2.putText(
                output_image, f"[{idx}] {label_type}",
                (col_c + 5, row_c), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
            )
        elif obj_type == "BOX":
            half_w_pixel = int(obj["param1"] / meta.resolution_m)
            half_h_pixel = int(obj["param2"] / meta.resolution_m)
            p1 = (col_c - half_w_pixel, row_c - half_h_pixel)
            p2 = (col_c + half_w_pixel, row_c + half_h_pixel)
            cv2.rectangle(output_image, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                output_image, f"[{idx}] {label_type}",
                (p1[0] + 3, max(15, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
            )
        elif obj_type == "VISUAL_BOX":
            half_w_pixel = int(obj["param1"] / meta.resolution_m)
            half_h_pixel = int(obj["param2"] / meta.resolution_m)
            p1 = (col_c - half_w_pixel, row_c - half_h_pixel)
            p2 = (col_c + half_w_pixel, row_c + half_h_pixel)
            cv2.rectangle(output_image, p1, p2, (80, 80, 80), 1)

    return output_image


def build_field_map_data(
    cropped_points: np.ndarray,
    bbox: BoundingBox2D,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float,
    generate_outer_walls_flag: bool = True,
    additional_objects: Optional[List[Dict[str, Any]]] = None,
    clean_map_mode: bool = True
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, CoordinateTransformMeta]:
    """クロップ点群からクリーンな Nav2 2D 地図と gn10 互換フィールド定義を構築

    Args:
        cropped_points: クロップ済み点群 (N, 3)
        bbox: 選択バウンディングボックス
        floor_z_m: 床面 Z 座標 [m]
        ground_margin_m: 床面ノイズ除外マージン [m]
        robot_height_m: ロボット全高上限 [m]
        resolution_m: グリッド解像度 [m/pixel]
        generate_outer_walls_flag: 外壁を生成するか
        additional_objects: 手動追加されたオブジェクト定義のリスト
        clean_map_mode: 内部ノイズを一掃し、直線壁と登録障害物のみを描画するモード

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, CoordinateTransformMeta]:
            - field_objects: gn10-pointcloud-localization 互換オブジェクトリスト
            - nav2_obstacle_grid: Nav2 用 2D 占有バイナリグリッド (255: 障害物, 0: 床)
            - preview_bgr_image: 描画済みプレビュー画像
            - meta: クロップ領域の座標変換メタデータ
    """
    raw_obstacle_grid, meta, _ = create_occupancy_grid_and_meta(
        cropped_points=cropped_points,
        bbox=bbox,
        floor_z_m=floor_z_m,
        ground_margin_m=ground_margin_m,
        robot_height_m=robot_height_m,
        resolution_m=resolution_m
    )

    field_objects: List[Dict[str, Any]] = []

    # 1. 外周壁 (立上りフェンス & 土台)
    if generate_outer_walls_flag:
        field_objects.extend(generate_outer_walls(bbox))

    # 2. 手動追加オブジェクト
    if additional_objects:
        field_objects.extend(additional_objects)

    # 3. Nav2 用グリッドの作成
    if clean_map_mode:
        # ユーザー提案のクリーンモード: 内部を完全な白床にし、直線壁と登録障害物のみ描画
        nav2_obstacle_grid = create_clean_nav2_map(meta, field_objects)
    else:
        nav2_obstacle_grid = raw_obstacle_grid

    # プレビュー画像生成 & オブジェクト枠描画
    preview_base = cv2.cvtColor(raw_obstacle_grid, cv2.COLOR_GRAY2BGR)
    preview_bgr_image = draw_objects_on_preview(preview_base, meta, field_objects)

    return field_objects, nav2_obstacle_grid, preview_bgr_image, meta
