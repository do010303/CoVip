# -*- coding: utf-8 -*-
"""
Module chuyển đổi tọa độ Hand-Eye Calibration (Camera Optical Frame -> Robot Base Frame).

Quy ước:
  Camera Optical Frame (OpenCV Pinhole standard):
    - X_c: Hướng sang phải (pixel u tăng)
    - Y_c: Hướng xuống dưới (pixel v tăng)
    - Z_c: Trục quang học hướng ra phía trước (độ sâu depth)

  Robot Base Frame (ASSARM 4-DOF):
    - X_b: Hướng sang phải (đối xứng qua tâm đế)
    - Y_b: Hướng ra phía trước mặt robot (chiều cánh tay vươn tới)
    - Z_b: Hướng thẳng đứng lên trời

Vị trí gắn Camera vật lý (theo ảnh chụp thực tế):
  Gắn trên 2 thanh ray màu đen nối từ base_link màu cam:
    - X_offset = 0.0 cm (nằm chính giữa tim robot)
    - Forward offset (Z_mount) = 7.0 cm (nhô ra phía trước 7cm so với tâm base_link)
    - Downward offset (Y_mount) = 1.5 cm (thấp hơn 1.5cm so với tâm base_link)
    - base_link height = 0.56 m (56.0 cm từ mặt bàn tới tâm base_link)
"""

from typing import Tuple
import numpy as np


class HandEyeTransformer:
    def __init__(
        self,
        forward_offset_m: float = 0.070,   # Nhô ra trước 7cm
        downward_offset_m: float = 0.015,  # Thấp hơn 1.5cm
        lateral_offset_m: float = 0.000,   # Lệch ngang 0cm (chính tâm)
        base_height_m: float = 0.560,      # Chiều cao tâm base_link so với mặt bàn
        z_cam_world: float = 0.400         # Độ cao quang tâm camera (mặc định 0.40m khi đặt ngang bàn)
    ):
        self.x_off = float(lateral_offset_m)
        self.y_off = float(forward_offset_m)
        if z_cam_world is not None:
            self.z_cam_world = float(z_cam_world)
        else:
            self.z_cam_world = float(base_height_m - downward_offset_m) # 0.545m

    def camera_to_base(self, x_c: float, y_c: float, z_c: float) -> Tuple[float, float, float]:
        """
        Chuyển đổi tọa độ 3D từ hệ camera sang hệ robot base.

        Công thức:
          X_b = X_c + x_off
          Y_b = Z_c + y_off  (độ sâu Z của camera chính là phương vươn tới trước Y của robot)
          Z_b = z_cam_world - Y_c (Y camera hướng xuống -> vật càng thấp thì Y_c càng lớn -> Z_b càng giảm)
        """
        x_b = float(x_c + self.x_off)
        y_b = float(z_c + self.y_off)
        z_b = float(self.z_cam_world - y_c)
        return x_b, y_b, z_b

    def camera_vel_to_base(self, vx_c: float, vy_c: float, vz_c: float) -> Tuple[float, float, float]:
        """
        Chuyển đổi vận tốc 3D (dùng cho Deep RL SAC).
        """
        vx_b = float(vx_c)
        vy_b = float(vz_c)
        vz_b = float(-vy_c)
        return vx_b, vy_b, vz_b

    def update_offsets(self, forward_cm: float, downward_cm: float, lateral_cm: float = 0.0):
        """Cho phép cập nhật lại offset linh hoạt."""
        self.y_off = float(forward_cm / 100.0)
        self.x_off = float(lateral_cm / 100.0)
        self.z_cam_world = float(0.560 - (downward_cm / 100.0))
