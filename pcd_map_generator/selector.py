"""
点群の鳥瞰図プレビュー生成および切り抜き領域 (ROI) 選択モジュール
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from pcd_map_generator.coordinates import BoundingBox2D, CoordinateTransformMeta


def generate_topdown_preview_image(
    points_array: np.ndarray,
    floor_z_m: float,
    ground_margin_m: float,
    robot_height_m: float,
    resolution_m: float
) -> Tuple[np.ndarray, CoordinateTransformMeta]:
    """障害物高さ範囲の点群から、グリッド線・原点マーカー付きの全体鳥瞰図画像を生成

    Args:
        points_array: 点群座標配列 (N, 3)
        floor_z_m: 床面の Z 座標 [m]
        ground_margin_m: 床面ノイズ除外マージン [m]
        robot_height_m: ロボット全高上限 [m] (天井・梁を除去)
        resolution_m: 2D グリッド解像度 [m/pixel]

    Returns:
        Tuple[np.ndarray, CoordinateTransformMeta]:
            - preview_image: 描画済み BGR 画像 (NumPy配列)
            - meta: 座標変換メタデータ

    Raises:
        ValueError: 指定高さ範囲に点群が存在しない場合
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

    # 原点 (0, 0) マーカー
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
    """GUI ウィンドウ上でマウスドラッグして領域を選択 (CLI 引数指定時は直接変換)

    Args:
        preview_image: 鳥瞰図プレビュー画像
        meta: 座標変換メタデータ
        bbox_override: CLI 等から直接与えられた境界 [min_x, max_x, min_y, max_y]

    Returns:
        BoundingBox2D: 確定した実世界座標系の選択領域

    Raises:
        RuntimeError: GUIで領域選択がキャンセルされた場合
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

    col_pixel, row_pixel, width_pixel, height_pixel = cv2.selectROI(
        window_name, preview_image, fromCenter=False, showCrosshair=True
    )
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
