"""
座標系変換およびバウンディングボックスの定義モジュール
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CoordinateTransformMeta:
    """ピクセル座標と実世界（メートル）座標の相互変換メタデータ

    Attributes:
        resolution_m: 1ピクセルあたりのメートル幅 [m/pixel]
        min_x_m: 画像左端の実世界 X 座標 [m]
        max_x_m: 画像右端の実世界 X 座標 [m]
        min_y_m: 画像下端の実世界 Y 座標 [m]
        max_y_m: 画像上端の実世界 Y 座標 [m]
        image_width_pixel: 画像の横幅 [pixel]
        image_height_pixel: 画像の縦幅 [pixel]
    """
    resolution_m: float
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    image_width_pixel: int
    image_height_pixel: int

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """実世界メートル座標 (x, y) を画像ピクセル座標 (col, row) に変換

        Args:
            x_m: 実世界 X 座標 [m]
            y_m: 実世界 Y 座標 [m]

        Returns:
            Tuple[int, int]: (col, row) ピクセル座標
        """
        col_pixel = int(np.clip((x_m - self.min_x_m) / self.resolution_m, 0, self.image_width_pixel - 1))
        row_pixel = int(np.clip((self.max_y_m - y_m) / self.resolution_m, 0, self.image_height_pixel - 1))
        return col_pixel, row_pixel

    def pixel_to_world(self, col_pixel: int, row_pixel: int) -> Tuple[float, float]:
        """画像ピクセル座標 (col, row) を実世界メートル座標 (x, y) に変換

        Args:
            col_pixel: 列番号 (X方向ピクセル)
            row_pixel: 行番号 (Y方向ピクセル)

        Returns:
            Tuple[float, float]: (x_m, y_m) メートル座標
        """
        x_m = self.min_x_m + col_pixel * self.resolution_m
        y_m = self.max_y_m - row_pixel * self.resolution_m
        return x_m, y_m


@dataclass
class BoundingBox2D:
    """2D 平面上の矩形領域 (バウンディングボックス) [m]

    Attributes:
        min_x_m: 領域の最小 X 座標 [m]
        max_x_m: 領域の最大 X 座標 [m]
        min_y_m: 領域の最小 Y 座標 [m]
        max_y_m: 領域の最大 Y 座標 [m]
    """
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
