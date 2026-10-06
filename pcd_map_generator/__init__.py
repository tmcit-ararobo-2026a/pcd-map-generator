"""
pcd-map-generator: 3D PCD 点群から 2D 地図およびフィールド定義を自動生成するパッケージ
"""

from pcd_map_generator.coordinates import BoundingBox2D, CoordinateTransformMeta
from pcd_map_generator.exporter import save_field_json_map, save_preview_image, save_ros_nav2_map
from pcd_map_generator.extractor import (
    build_field_map_data,
    create_occupancy_grid_and_meta,
    draw_objects_on_preview,
    generate_outer_walls,
)
from pcd_map_generator.pointcloud import (
    crop_and_save_point_cloud,
    detect_floor_elevation_z,
    load_point_cloud,
)
from pcd_map_generator.selector import (
    generate_topdown_preview_image,
    select_obstacles_interactively,
    select_region_of_interest,
)

# 後方互換性エイリアス
extract_field_objects_and_create_maps = build_field_map_data

__all__ = [
    "BoundingBox2D",
    "CoordinateTransformMeta",
    "load_point_cloud",
    "detect_floor_elevation_z",
    "crop_and_save_point_cloud",
    "generate_topdown_preview_image",
    "select_region_of_interest",
    "select_obstacles_interactively",
    "generate_outer_walls",
    "create_occupancy_grid_and_meta",
    "draw_objects_on_preview",
    "build_field_map_data",
    "extract_field_objects_and_create_maps",
    "save_ros_nav2_map",
    "save_field_json_map",
    "save_preview_image",
]
