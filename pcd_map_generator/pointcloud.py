"""
3D 点群データの読み込み・床面検出・クロップ処理モジュール
"""

import os
from typing import Tuple

import numpy as np
import open3d as o3d

from pcd_map_generator.coordinates import BoundingBox2D


def load_point_cloud(pcd_file_path: str) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """PCD ファイルを読み込み、Open3D オブジェクトと NumPy 配列を返却

    Args:
        pcd_file_path: 入力 PCD ファイルのパス

    Returns:
        Tuple[o3d.geometry.PointCloud, np.ndarray]:
            - o3d_point_cloud: Open3D の点群オブジェクト
            - points_numpy_array: (N, 3) 形状の座標配列

    Raises:
        FileNotFoundError: 指定パスにファイルが存在しない場合
        ValueError: 点群データが空の場合
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


def detect_floor_elevation_z(points_array: np.ndarray, bin_width_m: float = 0.02) -> float:
    """Z 軸のヒストグラムから、最も点群が密集している床面の高さ (Z [m]) を推定

    点群下部（下位2%〜50%）の Z 座標分布を調べ、もっとも出現頻度が高い Z 座標を
    床面（グラウンドレベル）として同定します。

    Args:
        points_array: 点群座標配列 (N, 3)
        bin_width_m: ヒストグラムのビン幅 [m]

    Returns:
        float: 推定された床面の Z 座標 [m]
    """
    z_coordinates = points_array[:, 2]
    z_lower_bound = np.percentile(z_coordinates, 2)
    z_upper_bound = np.percentile(z_coordinates, 50)

    candidate_mask = (z_coordinates >= z_lower_bound) & (z_coordinates <= z_upper_bound)
    candidate_z = z_coordinates[candidate_mask]

    bin_count = max(10, int((z_upper_bound - z_lower_bound) / bin_width_m))
    histogram_counts, bin_edges = np.histogram(candidate_z, bins=bin_count)

    peak_bin_index = np.argmax(histogram_counts)
    estimated_floor_z_m = float((bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0)

    print(f"[2/6] 床面高さを自動検出: Z = {estimated_floor_z_m:.3f} m")
    return estimated_floor_z_m


def crop_and_save_point_cloud(
    raw_point_cloud: o3d.geometry.PointCloud,
    bbox: BoundingBox2D,
    floor_z_m: float,
    robot_height_m: float,
    output_pcd_path: str,
    z_bottom_margin_m: float = 0.10,
    z_top_margin_m: float = 0.50
) -> o3d.geometry.PointCloud:
    """指定された 2D バウンディングボックスと高さ範囲で点群をクロップし、PCD として保存

    Args:
        raw_point_cloud: 入力 Open3D 点群オブジェクト
        bbox: 選択された 2D 水平バウンディングボックス
        floor_z_m: 床面の Z 座標 [m]
        robot_height_m: ロボットの全高 [m]
        output_pcd_path: 出力先 PCD ファイルパス
        z_bottom_margin_m: 床面下の許容マージン [m] (デフォルト: 0.10m)
        z_top_margin_m: 天井方向の許容マージン [m] (デフォルト: 0.50m)

    Returns:
        o3d.geometry.PointCloud: クロップされた点群オブジェクト
    """
    z_min_bound = floor_z_m - z_bottom_margin_m
    z_max_bound = floor_z_m + robot_height_m + z_top_margin_m

    axis_aligned_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([bbox.min_x_m, bbox.min_y_m, z_min_bound]),
        max_bound=np.array([bbox.max_x_m, bbox.max_y_m, z_max_bound])
    )
    cropped_cloud = raw_point_cloud.crop(axis_aligned_box)
    o3d.io.write_point_cloud(output_pcd_path, cropped_cloud)
    print(f"[4/6] クロップ済み PCD を保存しました ({len(cropped_cloud.points):,} 点): {output_pcd_path}")
    return cropped_cloud
