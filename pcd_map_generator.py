#!/usr/bin/env python3
"""
PCD インタラクティブ領域抽出 & 2D マップ / JSON 生成ツール

生 PCD 点群データから、GUI マウス操作（または CLI 範囲指定）によってテストエリアを切り抜き、
自己位置推定（gn10-pointcloud-localization）および障害物回避（Nav2）に必要なファイルを一括自動生成します。
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import open3d as o3d


@dataclass
class CoordinateTransformMeta:
    """ピクセル座標と実世界（メートル）座標の変換メタデータ"""
    resolution_m: float
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    image_width_pixel: int
    image_height_pixel: int

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """メートル座標 (x, y) を画像ピクセル座標 (col, row) に変換"""
        col_pixel = int(np.clip((x_m - self.min_x_m) / self.resolution_m, 0, self.image_width_pixel - 1))
        row_pixel = int(np.clip((self.max_y_m - y_m) / self.resolution_m, 0, self.image_height_pixel - 1))
        return col_pixel, row_pixel

    def pixel_to_world(self, col_pixel: int, row_pixel: int) -> Tuple[float, float]:
        """画像ピクセル座標 (col, row) をメートル座標 (x, y) に変換"""
        x_m = self.min_x_m + col_pixel * self.resolution_m
        y_m = self.max_y_m - row_pixel * self.resolution_m
        return x_m, y_m


@dataclass
class BoundingBox2D:
    """2D 平面上の選択バウンディングボックス [m]"""
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float


def load_point_cloud(pcd_file_path: str) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    PCD ファイルを読み込み、Open3D オブジェクトと NumPy 配列を返却します。

    :param pcd_file_path: 入力 PCD ファイルの絶対/相対パス
    :return: (o3d_point_cloud, points_numpy_array)
    """
    if not os.path.exists(pcd_file_path):
        raise FileNotFoundError(f"PCD ファイルが存在しません: {pcd_file_path}")

    print(f"[1/6] PCD ファイルを読み込み中: {pcd_file_path}")
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    points_array = np.asarray(point_cloud.points)

    if len(points_array) == 0:
        raise ValueError("PCD ファイル内に点群が存在しません。")

    print(f"      読み込み完了: {len(points_array):,} 点")
    return point_cloud, points_array


def detect_floor_elevation_z(points_array: np.ndarray) -> float:
    """
    Z 軸のヒストグラムから、最も点群が密集している床面の高さ (Z座標 [m]) を推定します。

    :param points_array: 点群座標配列 (N, 3)
    :return: 推定された床面の Z 座標 [m]
    """
    z_coordinates = points_array[:, 2]
    z_lower_bound = np.percentile(z_coordinates, 2)
    z_upper_bound = np.percentile(z_coordinates, 50)

    candidate_mask = (z_coordinates >= z_lower_bound) & (z_coordinates <= z_upper_bound)
    candidate_z = z_coordinates[candidate_mask]

    bin_width_m = 0.02
    bin_count = max(10, int((z_upper_bound - z_lower_bound) / bin_width_m))
    histogram_counts, bin_edges = np.histogram(candidate_z, bins=bin_count)

    peak_bin_index = np.argmax(histogram_counts)
    estimated_floor_z_m = float((bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0)

    print(f"[2/6] 床面高さを自動検出: Z = {estimated_floor_z_m:.3f} m")
    return estimated_floor_z_m


def generate_topdown_preview_image(
    points_array: np.ndarray,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float
) -> Tuple[np.ndarray, CoordinateTransformMeta]:
    """
    障害物高さ範囲の点群から、全体鳥瞰図（グリッド線・座標目盛り付き）画像を生成します。
    """
    z_min_filter = floor_z_m + ground_margin_m
    z_max_filter = floor_z_m + robot_height_m
    obstacle_mask = (points_array[:, 2] >= z_min_filter) & (points_array[:, 2] <= z_max_filter)
    filtered_points = points_array[obstacle_mask]

    if len(filtered_points) == 0:
        raise ValueError("指定された高さ範囲内に障害物点群が存在しません。")

    min_x_m = float(np.floor(filtered_points[:, 0].min()))
    max_x_m = float(np.ceil(filtered_points[:, 0].max()))
    min_y_m = float(np.floor(filtered_points[:, 1].min()))
    max_y_m = float(np.ceil(filtered_points[:, 1].max()))

    image_width_pixel = int(np.ceil((max_x_m - min_x_m) / resolution_m))
    image_height_pixel = int(np.ceil((max_y_m - min_y_m) / resolution_m))

    meta = CoordinateTransformMeta(
        resolution_m=resolution_m,
        min_x_m=min_x_m,
        max_x_m=max_x_m,
        min_y_m=min_y_m,
        max_y_m=max_y_m,
        image_width_pixel=image_width_pixel,
        image_height_pixel=image_height_pixel
    )

    binary_grid = np.zeros((image_height_pixel, image_width_pixel), dtype=np.uint8)
    cols = ((filtered_points[:, 0] - min_x_m) / resolution_m).astype(int)
    rows = ((max_y_m - filtered_points[:, 1]) / resolution_m).astype(int)
    valid = (cols >= 0) & (cols < image_width_pixel) & (rows >= 0) & (rows < image_height_pixel)
    binary_grid[rows[valid], cols[valid]] = 255

    preview_image = cv2.cvtColor(binary_grid, cv2.COLOR_GRAY2BGR)

    # 1メートルごとのグリッド線
    for x_grid_m in range(int(min_x_m), int(max_x_m) + 1):
        col, _ = meta.world_to_pixel(float(x_grid_m), 0.0)
        cv2.line(preview_image, (col, 0), (col, image_height_pixel), (40, 40, 40), 1)
        if x_grid_m % 2 == 0:
            cv2.putText(preview_image, f"{x_grid_m}m", (col + 2, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    for y_grid_m in range(int(min_y_m), int(max_y_m) + 1):
        _, row = meta.world_to_pixel(0.0, float(y_grid_m))
        cv2.line(preview_image, (0, row), (image_width_pixel, row), (40, 40, 40), 1)
        if y_grid_m % 2 == 0:
            cv2.putText(preview_image, f"{y_grid_m}m", (5, row - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # 原点 (0, 0)
    orig_col, orig_row = meta.world_to_pixel(0.0, 0.0)
    cv2.drawMarker(preview_image, (orig_col, orig_row), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(preview_image, "(0,0)", (orig_col + 6, orig_row - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return preview_image, meta


def select_region_of_interest(
    preview_image: np.ndarray,
    meta: CoordinateTransformMeta,
    bbox_override: Optional[List[float]] = None
) -> BoundingBox2D:
    """
    GUI ウィンドウ上でマウスドラッグして領域を選択します（CLI指定時は直接変換）。
    """
    if bbox_override is not None:
        min_x_m, max_x_m, min_y_m, max_y_m = bbox_override
        print(f"[3/6] CLI引数から領域を直接指定: X=[{min_x_m:.2f}, {max_x_m:.2f}], Y=[{min_y_m:.2f}, {max_y_m:.2f}]")
        return BoundingBox2D(min_x_m=min_x_m, max_x_m=max_x_m, min_y_m=min_y_m, max_y_m=max_y_m)

    print("\n=======================================================")
    print("【操作方法】")
    print(" 1. ポップアップしたウィンドウで、使いたい部屋・フィールドをマウスドラッグで囲んでください。")
    print(" 2. 囲んだら [Space] または [Enter] キーを押して確定します。")
    print(" 3. やり直したい場合は [c] キーを押してください。")
    print("=======================================================\n")

    window_name = "PCD Map Generator - Drag to Select Field Area (Enter/Space: Confirm, c: Retry)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, preview_image.shape[1]), min(900, preview_image.shape[0]))

    col_pixel, row_pixel, width_pixel, height_pixel = cv2.selectROI(window_name, preview_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if width_pixel == 0 or height_pixel == 0:
        raise RuntimeError("領域が選択されませんでした（キャンセルされました）。")

    x1_m, y1_m = meta.pixel_to_world(col_pixel, row_pixel)
    x2_m, y2_m = meta.pixel_to_world(col_pixel + width_pixel, row_pixel + height_pixel)

    min_x_m = min(x1_m, x2_m)
    max_x_m = max(x1_m, x2_m)
    min_y_m = min(y1_m, y2_m)
    max_y_m = max(y1_m, y2_m)

    print(f"[3/6] 選択完了: X=[{min_x_m:.2f}, {max_x_m:.2f}] m, Y=[{min_y_m:.2f}, {max_y_m:.2f}] m")
    return BoundingBox2D(min_x_m=min_x_m, max_x_m=max_x_m, min_y_m=min_y_m, max_y_m=max_y_m)


def crop_and_save_point_cloud(
    raw_point_cloud: o3d.geometry.PointCloud,
    bbox: BoundingBox2D,
    floor_z_m: float,
    robot_height_m: float,
    output_pcd_path: str
) -> o3d.geometry.PointCloud:
    """
    指定されたバウンディングボックスで点群をクロップし、新しい PCD として保存します。
    """
    z_min_bound = floor_z_m - 0.10
    z_max_bound = floor_z_m + robot_height_m + 0.50

    axis_aligned_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([bbox.min_x_m, bbox.min_y_m, z_min_bound]),
        max_bound=np.array([bbox.max_x_m, bbox.max_y_m, z_max_bound])
    )
    cropped_cloud = raw_point_cloud.crop(axis_aligned_box)
    o3d.io.write_point_cloud(output_pcd_path, cropped_cloud)
    print(f"[4/6] クロップ済み PCD を保存しました ({len(cropped_cloud.points):,} 点): {output_pcd_path}")
    return cropped_cloud


def extract_field_objects_and_create_maps(
    cropped_points: np.ndarray,
    bbox: BoundingBox2D,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float,
    generate_outer_walls: bool = True
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, CoordinateTransformMeta]:
    """
    クロップされた点群から 2D グリッドマップを生成し、輪郭検出で幾何オブジェクト（BOX/CYLINDER）を抽出します。
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

    # モルフォロジー演算 (Closing) で微細な点群ノイズを繋ぎ、物体化
    kernel_size_pixel = max(3, int(0.10 / resolution_m))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_pixel, kernel_size_pixel))
    closed_obstacle_grid = cv2.morphologyEx(binary_obstacle_grid, cv2.MORPH_CLOSE, kernel)

    field_objects: List[Dict[str, Any]] = []
    preview_bgr_image = cv2.cvtColor(closed_obstacle_grid, cv2.COLOR_GRAY2BGR)

    object_index = 1

    # 1. 外周壁 (選択範囲の外枠) を自動生成
    if generate_outer_walls:
        wall_thickness_m = 0.024  # 厚み約 24mm (半幅 0.012m)
        wall_height_max_m = 0.150 # 立上りフェンス標準高さ

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

        for wall_name, wx, wy, p1, p2 in outer_walls:
            field_objects.append({
                "comment": wall_name,
                "type": "BOX",
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z_min": 0.024,
                "z_max": round(wall_height_max_m, 3),
                "param1": round(p1, 3),
                "param2": round(p2, 3)
            })

    # 2. 内部の障害物（机、バケツなど）を輪郭検出
    # 全体を覆う画像端をマスクアウトして、内部の独立した塊を取り出す
    inner_mask = np.copy(closed_obstacle_grid)
    # 外枠 2 ピクセルをゼロクリア（外枠境界の巻き込みを防止）
    inner_mask[0:2, :] = 0
    inner_mask[-2:, :] = 0
    inner_mask[:, 0:2] = 0
    inner_mask[:, -2:] = 0

    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_pixel = (0.15 / resolution_m) * (0.15 / resolution_m)  # 15cm x 15cm 未満のノイズは無視

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

        # 形状判定
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


def save_ros_nav2_map(
    obstacle_grid: np.ndarray,
    meta: CoordinateTransformMeta,
    output_pgm_path: str,
    output_yaml_path: str
) -> None:
    """
    ROS 2 Navigation (Nav2) 標準の 2D マップ (PGM + YAML) を保存します。
    """
    ros_map_image = np.full_like(obstacle_grid, 254, dtype=np.uint8)
    ros_map_image[obstacle_grid == 255] = 0

    ros_map_image_flipped = cv2.flip(ros_map_image, 0)
    cv2.imwrite(output_pgm_path, ros_map_image_flipped)

    map_yaml_content = f"""image: {os.path.basename(output_pgm_path)}
mode: trinary
resolution: {meta.resolution_m}
origin: [{meta.min_x_m:.3f}, {meta.min_y_m:.3f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(map_yaml_content)

    print(f"[5/6] ROS 標準マップを保存しました: {output_pgm_path}, {output_yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCD インタラクティブ領域抽出 & 2D マップ / JSON 生成ツール"
    )
    parser.add_argument("pcd_path", type=str, help="入力 PCD ファイルのパス")
    parser.add_argument("--resolution_m", type=float, default=0.05, help="2D グリッド解像度 [m] (デフォルト: 0.05)")
    parser.add_argument("--robot_height_m", type=float, default=1.5, help="ロボット全高上限 [m] (デフォルト: 1.5)")
    parser.add_argument("--ground_margin_m", type=float, default=0.03, help="床面ノイズ除去マージン [m] (デフォルト: 0.03)")
    parser.add_argument("--output_dir", type=str, default="output", help="出力先ディレクトリ (デフォルト: output)")
    parser.add_argument("--prefix", type=str, default="practice_field", help="ファイル名接頭辞 (デフォルト: practice_field)")
    parser.add_argument("--no_outer_walls", action="store_true", help="外周壁の自動生成を無効化する場合")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
                        help="CLI から領域を直接指定する場合 [min_x max_x min_y max_y]")
    parser.add_argument("--headless", action="store_true", help="GUI ウィンドウを開かないモード")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. PCD 読み込み
    point_cloud, points_array = load_point_cloud(args.pcd_path)

    # 2. 床面高さの自動検出
    floor_z_m = detect_floor_elevation_z(points_array)

    # 3. 全体プレビュー画像の生成
    topdown_preview, meta = generate_topdown_preview_image(
        points_array=points_array,
        floor_z_m=floor_z_m,
        ground_margin_m=args.ground_margin_m,
        robot_height_m=args.robot_height_m,
        resolution_m=args.resolution_m
    )

    # 4. 領域選択 (GUI または CLI)
    bbox_override = args.bbox if (args.bbox or args.headless) else None
    if args.headless and bbox_override is None:
        bbox_override = [meta.min_x_m, meta.max_x_m, meta.min_y_m, meta.max_y_m]

    selected_bbox = select_region_of_interest(topdown_preview, meta, bbox_override=bbox_override)

    # 5. 点群クロップ & PCD 保存
    output_pcd_path = os.path.join(args.output_dir, f"{args.prefix}_cropped.pcd")
    cropped_cloud = crop_and_save_point_cloud(
        raw_point_cloud=point_cloud,
        bbox=selected_bbox,
        floor_z_m=floor_z_m,
        robot_height_m=args.robot_height_m,
        output_pcd_path=output_pcd_path
    )

    # 6. オブジェクト抽出 & 2D マップ生成
    cropped_points = np.asarray(cropped_cloud.points)
    field_objects, obstacle_grid, preview_image, local_meta = extract_field_objects_and_create_maps(
        cropped_points=cropped_points,
        bbox=selected_bbox,
        floor_z_m=floor_z_m,
        ground_margin_m=args.ground_margin_m,
        robot_height_m=args.robot_height_m,
        resolution_m=args.resolution_m,
        generate_outer_walls=(not args.no_outer_walls)
    )

    # 7. ROS 標準マップ (PGM + YAML) 保存
    output_pgm_path = os.path.join(args.output_dir, f"{args.prefix}_map.pgm")
    output_yaml_path = os.path.join(args.output_dir, f"{args.prefix}_map.yaml")
    save_ros_nav2_map(obstacle_grid, local_meta, output_pgm_path, output_yaml_path)

    # 8. 人間用確認画像 (PNG) 保存
    output_preview_png = os.path.join(args.output_dir, f"{args.prefix}_preview.png")
    cv2.imwrite(output_preview_png, preview_image)

    # 9. gn10-pointcloud-localization 互換 JSON 保存
    output_json_path = os.path.join(args.output_dir, f"{args.prefix}_map.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(field_objects, f, indent=2, ensure_ascii=False)

    print(f"[6/6] gn10-pointcloud-localization 互換 JSON を保存しました: {output_json_path}")
    print(f"      検出オブジェクト数: {len(field_objects)} 個")
    print(f"      確認用プレビュー画像: {output_preview_png}")
    print("\nすべての生成処理が正常に完了しました！")


if __name__ == "__main__":
    main()
