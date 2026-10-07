"""
点群の鳥瞰図プレビュー生成、領域選択 (ROI)、および手動障害物追加モジュール
"""

from typing import Any, Dict, List, Optional, Tuple

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
    print("【操作方法: フィールド領域選択】")
    print(" 1. ポップアップしたウィンドウで、使いたい部屋・フィールドをマウスドラッグで囲んでください。")
    print(" 2. 囲んだら [Space] または [Enter] キーを押して確定します。")
    print(" 3. やり直したい場合は [c] キーを押してください。")
    print("=======================================================\n")

    window_name = "PCD Map Generator - Select Field Area (Enter/Space: Confirm, c: Retry)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 画像のアスペクト比を維持しつつ、画面上で見やすい大きさに拡大 (幅1000〜1400px、高さ800〜950px程度)
    img_h, img_w = preview_image.shape[:2]
    scale = max(1.0, min(1300.0 / max(1, img_w), 900.0 / max(1, img_h)))
    display_w = int(img_w * scale)
    display_h = int(img_h * scale)
    cv2.resizeWindow(window_name, display_w, display_h)

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


def create_obstacle_selection_guide_image(
    cropped_points: np.ndarray,
    local_meta: CoordinateTransformMeta,
    floor_z_m: float,
    ground_margin_m: float = 0.03,
    robot_height_m: float = 1.5
) -> np.ndarray:
    """障害物選択用の直感的でクリーンな下書き画像を生成

    - 背景: 白 (走行可能床)
    - 外壁: ピシッとした緑の枠線
    - 点群: 位置の目印となる薄いグレースケールドット (ノイズに見えないよう淡く表示)
    """
    guide_image = np.full((local_meta.image_height_pixel, local_meta.image_width_pixel, 3), 255, dtype=np.uint8)

    # 1. 障害物高さの点群を薄いドットとして描画
    z_min_filter = floor_z_m + ground_margin_m
    z_max_filter = floor_z_m + robot_height_m
    mask = (cropped_points[:, 2] >= z_min_filter) & (cropped_points[:, 2] <= z_max_filter)
    pts = cropped_points[mask]

    if len(pts) > 0:
        cols = ((pts[:, 0] - local_meta.min_x_m) / local_meta.resolution_m).astype(int)
        rows = ((local_meta.max_y_m - pts[:, 1]) / local_meta.resolution_m).astype(int)
        valid = (cols >= 0) & (cols < local_meta.image_width_pixel) & (rows >= 0) & (rows < local_meta.image_height_pixel)
        # 淡いグレー (190, 190, 190) で目印として描画
        guide_image[rows[valid], cols[valid]] = (190, 190, 190)

    # 2. 外壁フェンスの四辺を鮮やかな緑で描画
    thickness_px = max(1, int(0.024 / local_meta.resolution_m))
    cv2.rectangle(
        guide_image,
        (0, 0),
        (local_meta.image_width_pixel - 1, local_meta.image_height_pixel - 1),
        (0, 180, 0),
        thickness_px
    )

    return guide_image


def select_obstacles_interactively(
    cropped_points: np.ndarray,
    local_meta: CoordinateTransformMeta,
    floor_z_m: float,
    ground_margin_m: float = 0.03,
    robot_height_m: float = 1.5
) -> List[Dict[str, Any]]:
    """GUI マウス操作で障害物 (机・バケツなど) を対話的に手動選択して追加

    Args:
        cropped_points: 切り抜きエリア内の点群 (N, 3)
        local_meta: 切り抜きエリアの座標変換メタデータ
        floor_z_m: 床面の Z 座標 [m]
        ground_margin_m: 床面マージン [m]
        robot_height_m: ロボット全高 [m]

    Returns:
        List[Dict[str, Any]]: 追加されたオブジェクト定義辞書のリスト
    """
    print("\n=======================================================")
    print("【操作方法: 障害物の手動追加 (GUI)】")
    print(" 薄いグレーの点群を参考に、追加したい机やバケツをマウスドラッグで囲んで [Enter] を押してください。")
    print(" 追加する障害物がない（または終わった）場合は、何も囲まずに [Enter] または [ESC] を押すと完了します。")
    print("=======================================================\n")

    current_preview = create_obstacle_selection_guide_image(
        cropped_points=cropped_points,
        local_meta=local_meta,
        floor_z_m=floor_z_m,
        ground_margin_m=ground_margin_m,
        robot_height_m=robot_height_m
    )
    added_objects: List[Dict[str, Any]] = []
    object_count = 1

    window_name = "Add Obstacles (Drag box -> Enter to confirm / Empty Enter to finish)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 選択しやすいように画面上で見やすい大きさに拡大 (幅・高さ約800〜1000px)
    scale = max(1.0, min(1000.0 / max(1, current_preview.shape[1]), 850.0 / max(1, current_preview.shape[0])))
    win_w = int(current_preview.shape[1] * scale)
    win_h = int(current_preview.shape[0] * scale)
    cv2.resizeWindow(window_name, win_w, win_h)

    while True:
        col, row, width, height = cv2.selectROI(
            window_name, current_preview, fromCenter=False, showCrosshair=True
        )

        # 何も囲まれずに確定されたら終了
        if width == 0 or height == 0:
            break

        # 選択された矩形の実世界座標
        x1_m, y1_m = local_meta.pixel_to_world(col, row)
        x2_m, y2_m = local_meta.pixel_to_world(col + width, row + height)
        sel_min_x, sel_max_x = min(x1_m, x2_m), max(x1_m, x2_m)
        sel_min_y, sel_max_y = min(y1_m, y2_m), max(y1_m, y2_m)

        center_x = (sel_min_x + sel_max_x) / 2.0
        center_y = (sel_min_y + sel_max_y) / 2.0
        half_w = (sel_max_x - sel_min_x) / 2.0
        half_h = (sel_max_y - sel_min_y) / 2.0

        # 囲まれた範囲内の点群から高さを自動推定
        in_x = (cropped_points[:, 0] >= sel_min_x) & (cropped_points[:, 0] <= sel_max_x)
        in_y = (cropped_points[:, 1] >= sel_min_y) & (cropped_points[:, 1] <= sel_max_y)
        points_in_roi = cropped_points[in_x & in_y]

        if len(points_in_roi) > 0:
            est_z_min = float(np.percentile(points_in_roi[:, 2], 5) - floor_z_m)
            est_z_max = float(np.percentile(points_in_roi[:, 2], 95) - floor_z_m)
        else:
            est_z_min, est_z_max = 0.0, 0.76

        est_z_min = max(0.0, round(est_z_min, 3))
        est_z_max = max(est_z_min + 0.05, round(est_z_max, 3))

        # コンソールで形状タイプと高さを確認・入力
        print(f"\n--- 障害物 #{object_count} を選択 ---")
        print(f" 位置: X={center_x:.2f}m, Y={center_y:.2f}m")
        print(f" 寸法: 幅={half_w*2:.2f}m, 奥行き={half_h*2:.2f}m")
        print(f" 点群から推定された高さ: Z_min={est_z_min:.2f}m, Z_max={est_z_max:.2f}m")

        type_input = input(" 形状を選択 [1: 直方体 (BOX), 2: 円柱 (CYLINDER)] (デフォルト: 1): ").strip()
        obj_type = "CYLINDER" if type_input == "2" else "BOX"

        comment_default = f"手動追加_{'バケツ' if obj_type == 'CYLINDER' else '机'}_{object_count}"
        comment_input = input(f" コメント名 (デフォルト: {comment_default}): ").strip()
        comment = comment_input if comment_input else comment_default

        z_input = input(f" 高さ設定 [z_min, z_max] (Enterで推定値 [{est_z_min}, {est_z_max}] を採用): ").strip()
        if z_input:
            try:
                parts = [float(p) for p in z_input.replace(",", " ").split()]
                if len(parts) >= 2:
                    est_z_min, est_z_max = parts[0], parts[1]
            except ValueError:
                pass

        if obj_type == "CYLINDER":
            radius = (half_w + half_h) / 2.0
            obj_def = {
                "comment": comment,
                "type": "CYLINDER",
                "x": round(center_x, 3),
                "y": round(center_y, 3),
                "z_min": round(est_z_min, 3),
                "z_max": round(est_z_max, 3),
                "param1": round(radius, 4),
                "param2": 0.0
            }
            # プレビュー上に円を描画 (赤色)
            col_c, row_c = local_meta.world_to_pixel(center_x, center_y)
            rad_px = int(radius / local_meta.resolution_m)
            cv2.circle(current_preview, (col_c, row_c), rad_px, (0, 0, 255), 2)
            cv2.putText(current_preview, f"[{object_count}] Cyl", (col_c + 5, row_c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            obj_def = {
                "comment": comment,
                "type": "BOX",
                "x": round(center_x, 3),
                "y": round(center_y, 3),
                "z_min": round(est_z_min, 3),
                "z_max": round(est_z_max, 3),
                "param1": round(half_w, 3),
                "param2": round(half_h, 3)
            }
            # プレビュー上に矩形を描画 (青色)
            p1_px = (col, row)
            p2_px = (col + width, row + height)
            cv2.rectangle(current_preview, p1_px, p2_px, (255, 0, 0), 2)
            cv2.putText(current_preview, f"[{object_count}] Box", (col + 3, row - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        added_objects.append(obj_def)
        print(f" -> 障害物 #{object_count} を登録しました！\n")
        object_count += 1

    cv2.destroyAllWindows()
    print(f"手動障害物の登録を完了しました (合計 {len(added_objects)} 個追加)。")
    return added_objects
