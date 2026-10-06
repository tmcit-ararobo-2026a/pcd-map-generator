#!/usr/bin/env python3
"""
PCD インタラクティブ領域抽出 & 2D マップ / JSON 生成ツール

生 PCD 点群データから、GUI マウス操作（または CLI 範囲指定）によってテストエリアを切り抜き、
自己位置推定（gn10-pointcloud-localization）および障害物回避（Nav2）に必要なファイルを一括自動生成します。
"""

import argparse
import os
import sys
import numpy as np

from pcd_map_generator import (
    load_point_cloud,
    detect_floor_elevation_z,
    generate_topdown_preview_image,
    select_region_of_interest,
    crop_and_save_point_cloud,
    extract_field_objects_and_create_maps,
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

    # 4. 領域選択 (GUI マウスドラッグ または CLI 引数)
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

    # 7. ROS 2 Nav2 標準マップ (PGM + YAML) 保存
    output_pgm_path = os.path.join(args.output_dir, f"{args.prefix}_map.pgm")
    output_yaml_path = os.path.join(args.output_dir, f"{args.prefix}_map.yaml")
    save_ros_nav2_map(obstacle_grid, local_meta, output_pgm_path, output_yaml_path)

    # 8. 人間確認用プレビュー画像 (PNG) 保存
    output_preview_png = os.path.join(args.output_dir, f"{args.prefix}_preview.png")
    save_preview_image(preview_image, output_preview_png)

    # 9. gn10-pointcloud-localization 互換 JSON 保存
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
