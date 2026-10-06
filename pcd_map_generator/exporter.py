"""
生成された地図・オブジェクト定義のファイル出力モジュール
"""

import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np

from pcd_map_generator.coordinates import CoordinateTransformMeta


def save_ros_nav2_map(
    obstacle_grid: np.ndarray,
    meta: CoordinateTransformMeta,
    output_pgm_path: str,
    output_yaml_path: str
) -> None:
    """ROS 2 Navigation (Nav2) 標準の 2D Occupancy Grid (PGM + YAML) を保存

    PGM 形式の白黒仕様:
        - 0 (黒): 占有障害物
        - 254 (白): 自由空間 (フリースペース)

    Args:
        obstacle_grid: 2D 占有グリッド (255: 障害物, 0: 未占有)
        meta: 座標変換メタデータ
        output_pgm_path: 出力先 PGM 画像パス
        output_yaml_path: 出力先 YAML 設定ファイルパス
    """
    ros_map_image = np.full_like(obstacle_grid, 254, dtype=np.uint8)
    ros_map_image[obstacle_grid == 255] = 0

    # ROS の地図画像は Y 軸が上向きのため上下反転して保存
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


def save_field_json_map(field_objects: List[Dict[str, Any]], output_json_path: str) -> None:
    """gn10-pointcloud-localization 互換のフィールド定義 JSON を保存

    Args:
        field_objects: オブジェクト定義辞書のリスト
        output_json_path: 出力先 JSON ファイルパス
    """
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(field_objects, f, indent=2, ensure_ascii=False)
    print(f"[6/6] gn10-pointcloud-localization 互換 JSON を保存しました: {output_json_path}")
    print(f"      検出オブジェクト数: {len(field_objects)} 個")


def save_preview_image(preview_image: np.ndarray, output_preview_path: str) -> None:
    """人間確認用の認識プレビュー PNG 画像を保存

    Args:
        preview_image: 描画済み BGR 画像
        output_preview_path: 出力先 PNG 画像パス
    """
    cv2.imwrite(output_preview_path, preview_image)
    print(f"      確認用プレビュー画像: {output_preview_path}")
