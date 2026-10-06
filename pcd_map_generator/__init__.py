"""
pcd-map-generator: 3D PCD 点群から 2D 地図およびフィールド定義を自動生成するパッケージ
"""

from pcd_map_generator.coordinates import BoundingBox2D, CoordinateTransformMeta
from pcd_map_generator.exporter import save_field_json_map, save_preview_image, save_ros_nav2_map
from pcd_map_generator.extractor import extract_field_objects_and_create_maps
from pcd_map_generator.pointcloud import (
    crop_and_save_point_cloud,
    detect_floor_elevation_z,
    load_point_cloud,
)
from pcd_map_generator.selector import (
    generate_topdown_preview_image,
    select_region_of_interest,
)

__all__ = [
    "BoundingBox2D",
    "CoordinateTransformMeta",
    "load_point_cloud",
    "detect_floor_elevation_z",
    "crop_and_save_point_cloud",
    "generate_topdown_preview_image",
    "select_region_of_interest",
    "extract_field_objects_and_create_maps",
    "save_ros_nav2_map",
    "save_field_json_map",
    "save_preview_image",
]
