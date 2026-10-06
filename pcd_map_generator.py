#!/usr/bin/env python3
"""
PCD インタラクティブ領域抽出 & 2D マップ / JSON 生成ツール

生 PCD 点群データから、GUI マウス操作（または CLI 範囲指定）によってテストエリアを切り抜き、
自己位置推定（gn10-pointcloud-localization）および障害物回避（Nav2）に必要なファイルを一括自動生成します。
"""

import argparse
import os
import sys
import cv2
import numpy as np

from pcd_map_generator import (
    load_point_cloud,
    detect_floor_elevation_z,
    generate_topdown_preview_image,
    select_region_of_interest,
    select_obstacles_interactively,
    crop_and_save_point_cloud,
    build_field_map_data,
    create_occupancy_grid_and_meta,
    draw_objects_on_preview,
    save_ros_nav2_map,
    save_field_json_map,
    save_preview_image,
)


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="PCD インタラクティブ領域抽出 & 2D マップ / JSON 生成ツール"
    )
    parser.add_argument("pcd_path", type=str, help="入力 PCD ファイルのパス")
    parser.add_argument(
        "--resolution_m",
        type=float,
        default=0.05,
        help="2D グリッド解像度 [m] (デフォルト: 0.05)"
    )
    parser.add_argument(
        "--robot_height_m",
        type=float,
        default=1.5,
        help="ロボット全高上限 [m] (デフォルト: 1.5)"
    )
    parser.add_argument(
        "--ground_margin_m",
        type=float,
        default=0.03,
        help="床面ノイズ除去マージン [m] (デフォルト: 0.03)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="出力先ディレクトリ (デフォルト: output)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="practice_field",
        help="ファイル名接頭辞 (デフォルト: practice_field)"
    )
    parser.add_argument(
        "--no_outer_walls",
        action="store_true",
        help="外周壁の自動生成を無効化する場合"
    )
    parser.add_argument(
        "--add_obstacles",
        action="store_true",
        help="GUI で机やバケツなどの障害物を手動追加するモード"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help="CLI から領域を直接指定する場合 [min_x max_x min_y max_y]"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="GUI ウィンドウを開かないモード"
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """マップ生成パイプラインを実行

    Args:
        args: パース済みコマンドライン引数
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. PCD 読み込み
    point_cloud, points_array = load_point_cloud(args.pcd_path)

    # 2. 床面高さの自動検出
    floor_z_m = detect_floor_elevation_z(points_array)

    # 3. 全体プレビュー鳥瞰図の生成
    topdown_preview, meta = generate_topdown_preview_image(
        points_array=points_array,
        floor_z_m=floor_z_m,
        ground_margin_m=args.ground_margin_m,
        robot_height_m=args.robot_height_m,
        resolution_m=args.resolution_m
    )

    # 4. フィールド領域選択 (GUI マウスドラッグ または CLI 引数)
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
    cropped_points = np.asarray(cropped_cloud.points)

    # 6. クロップ領域の 2D 占有グリッドを作成
    closed_obstacle_grid, local_meta, _ = create_occupancy_grid_and_meta(
        cropped_points=cropped_points,
        bbox=selected_bbox,
        floor_z_m=floor_z_m,
        ground_margin_m=args.ground_margin_m,
        robot_height_m=args.robot_height_m,
        resolution_m=args.resolution_m
    )

    # 7. 手動障害物追加 (GUI モードかつユーザーが希望した場合)
    additional_objects = []
    if not args.headless:
        prompt_add = args.add_obstacles
        if not prompt_add:
            user_choice = input("\nフィールド内部の障害物 (机・バケツ等) を GUI で追加しますか？ [y/N]: ").strip().lower()
            prompt_add = user_choice in ["y", "yes"]

        if prompt_add:
            base_preview_bgr = cv2.cvtColor(closed_obstacle_grid, cv2.COLOR_GRAY2BGR)
            additional_objects = select_obstacles_interactively(
                cropped_points=cropped_points,
                local_meta=local_meta,
                floor_z_m=floor_z_m,
                base_preview_image=base_preview_bgr
            )

    # 8. フィールド定義オブジェクト (外壁 + 追加障害物) およびプレビュー生成
    field_objects, _, preview_image, _ = build_field_map_data(
        cropped_points=cropped_points,
        bbox=selected_bbox,
        floor_z_m=floor_z_m,
        ground_margin_m=args.ground_margin_m,
        robot_height_m=args.robot_height_m,
        resolution_m=args.resolution_m,
        generate_outer_walls_flag=(not args.no_outer_walls),
        additional_objects=additional_objects
    )

    # 9. ROS 2 Nav2 標準マップ (PGM + YAML) 保存
    output_pgm_path = os.path.join(args.output_dir, f"{args.prefix}_map.pgm")
    output_yaml_path = os.path.join(args.output_dir, f"{args.prefix}_map.yaml")
    save_ros_nav2_map(closed_obstacle_grid, local_meta, output_pgm_path, output_yaml_path)

    # 10. 人間確認用プレビュー画像 (PNG) 保存
    output_preview_png = os.path.join(args.output_dir, f"{args.prefix}_preview.png")
    save_preview_image(preview_image, output_preview_png)

    # 11. gn10-pointcloud-localization 互換 JSON 保存
    output_json_path = os.path.join(args.output_dir, f"{args.prefix}_map.json")
    save_field_json_map(field_objects, output_json_path)

    print("\nすべての生成処理が正常に完了しました！")


def main() -> None:
    args = parse_arguments()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n[エラー] 処理中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
