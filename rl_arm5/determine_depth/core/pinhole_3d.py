"""Module tính toán tọa độ không gian 3D (X, Y, Z) theo Mô hình Camera Pinhole.

Nguyên lý:
  - Dựa trên tam giác đồng dạng giữa cảm biến 2D và không gian 3D thực:
      w_px / fx = D_real / Z
    => Z = (fx * D_real) / w_px

  - Chiếu ngược (Back-projection) từ điểm tâm ảnh (u, v) qua tâm quang học (cx, cy):
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
"""

import math
from typing import Tuple, Dict, Any

class Pinhole3DEstimator:
    def __init__(
        self,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
        d_real_m: float = 0.1,
        distance_scale: float = 1.0,
        distance_offset_m: float = 0.0
    ):
        """Khởi tạo bộ tính toán 3D Pinhole.

        Args:
            fx (float): Tiêu cự theo trục X (đơn vị: pixel).
            fy (float): Tiêu cự theo trục Y (đơn vị: pixel).
            cx (float): Tọa độ tâm quang học X (pixel).
            cy (float): Tọa độ tâm quang học Y (pixel).
            d_real_m (float): Đường kính thực tế của khối trụ (đơn vị: mét).
            distance_scale (float): Hệ số tỷ lệ hiệu chỉnh khoảng cách (mặc định 1.0).
            distance_offset_m (float): Độ bù khoảng cách mốc đo (mét, ví dụ -0.025m nếu đo tới mặt trước).
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.d_real = float(d_real_m)
        self.distance_scale = float(distance_scale)
        self.distance_offset_m = float(distance_offset_m)

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Cập nhật các thông số ma trận camera K."""
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def update_diameter(self, d_real_m: float):
        """Cập nhật kích thước thực tế của khối trụ."""
        self.d_real = float(d_real_m)

    def update_tuning(self, distance_scale: float = 1.0, distance_offset_m: float = 0.0):
        """Cập nhật hệ số bù khoảng cách thực nghiệm."""
        self.distance_scale = float(distance_scale)
        self.distance_offset_m = float(distance_offset_m)

    def estimate(self, u: float, v: float, w_px: float) -> Tuple[float, float, float]:
        """Tính toán tọa độ không gian (X, Y, Z) so với hệ tọa độ camera.

        Args:
            u (float): Tọa độ tâm theo trục ngang ảnh (pixel).
            v (float): Tọa độ tâm theo trục dọc ảnh (pixel).
            w_px (float): Độ rộng chiếu của khối trụ trên ảnh (pixel).

        Returns:
            Tuple[float, float, float]: (X, Y, Z) với đơn vị tính theo mét.
                X: Trục ngang (âm = bên trái camera, dương = bên phải camera)
                Y: Trục đứng (âm = phía trên camera, dương = phía dưới camera)
                Z: Độ sâu (khoảng cách vuông góc từ mặt phẳng thấu kính đến vật)
        """
        if w_px <= 0 or self.fx <= 0:
            return 0.0, 0.0, 0.0

        # 1. Tính khoảng cách độ sâu Z (Depth) theo Pinhole và áp dụng hiệu chỉnh thực nghiệm
        raw_Z = (self.fx * self.d_real) / w_px
        Z = max(0.01, (raw_Z * self.distance_scale) + self.distance_offset_m)

        # 2. Chiếu ngược 2D sang X và Y
        X = ((u - self.cx) * Z) / self.fx
        Y = ((v - self.cy) * Z) / self.fy

        return float(X), float(Y), float(Z)

    def get_full_metrics(self, u: float, v: float, w_px: float, h_px: float = 0.0) -> Dict[str, Any]:
        """Tính toán đầy đủ các chỉ số không gian bổ sung."""
        X, Y, Z = self.estimate(u, v, w_px)
        distance = math.sqrt(X**2 + Y**2 + Z**2)

        # Góc lệch so với trục nhìn chính của camera (độ)
        yaw_deg = math.degrees(math.atan2(X, Z)) if Z > 0 else 0.0
        pitch_deg = math.degrees(math.atan2(Y, Z)) if Z > 0 else 0.0

        # Ước tính chiều cao thực tế của khối trụ từ h_px để đối chiếu
        h_calc_m = (h_px * Z) / self.fy if (self.fy > 0 and Z > 0) else 0.0

        return {
            "X": X,
            "Y": Y,
            "Z": Z,
            "distance": distance,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "estimated_height_m": h_calc_m
        }
