# -*- coding: utf-8 -*-
"""
assarm_common.perception_filter — Bộ lọc vận tốc khối (alpha-beta filter).

Sai phân thô quá nhiễu (σ vị trí 4 mm ở 10 Hz → nhiễu vận tốc ~ 5.7 cm/s,
bằng chính vận tốc con lắc). Dùng alpha-beta filter với mô hình vận tốc
hằng số là đủ cho bước đầu; có thể nâng cấp lên Kalman với mô hình con lắc.

Dùng chung sim/thật.
"""

import numpy as np


class AlphaBetaFilter3D:
    """
    Alpha-beta filter cho vị trí 3D và vận tốc 3D.

    Thông số mặc định: alpha=0.4, beta=0.1 — đáp ứng đủ nhanh theo
    con lắc 0.91 Hz (T=1.1s) mà không quá nhiễu.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.1, dt: float = 0.1):
        """
        Parameters
        ----------
        alpha : float
            Hệ số smoothing vị trí (0–1). Cao = ít trễ, nhiều nhiễu.
        beta : float
            Hệ số smoothing vận tốc (0–1). Cao = đáp ứng nhanh thay đổi vận tốc.
        dt : float
            Chu kỳ lấy mẫu (s).
        """
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray | None = None

    def reset(self, pos: np.ndarray) -> None:
        """Khởi tạo bộ lọc tại vị trí ban đầu, vận tốc = 0."""
        self._pos = pos.copy()
        self._vel = np.zeros(3)

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Cập nhật bộ lọc với phép đo mới.

        Parameters
        ----------
        measurement : ndarray(3)
            Vị trí đo được (có nhiễu).

        Returns
        -------
        (pos_filtered, vel_filtered) : tuple of ndarray(3)
        """
        if self._pos is None:
            self.reset(measurement)
            return self._pos.copy(), self._vel.copy()

        # Dự đoán
        pos_pred = self._pos + self._vel * self.dt
        # Residual
        residual = measurement - pos_pred
        # Cập nhật
        self._pos = pos_pred + self.alpha * residual
        self._vel = self._vel + (self.beta / self.dt) * residual

        return self._pos.copy(), self._vel.copy()

    @property
    def position(self) -> np.ndarray:
        """Vị trí đã lọc hiện tại."""
        return self._pos.copy() if self._pos is not None else np.zeros(3)

    @property
    def velocity(self) -> np.ndarray:
        """Vận tốc đã lọc hiện tại."""
        return self._vel.copy() if self._vel is not None else np.zeros(3)


class PerceptionFilter:
    """
    Bộ lọc perception toàn diện: lọc vị trí + vận tốc khối,
    xử lý mất dữ liệu (giữ ước lượng cuối), và cờ pose_valid.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.1,
        dt: float = 0.1,
        max_missing_steps: int = 5,
    ):
        self.filter = AlphaBetaFilter3D(alpha=alpha, beta=beta, dt=dt)
        self.dt = dt
        self.max_missing_steps = max_missing_steps
        self._missing_count = 0
        self._pose_valid = False
        self._last_yaw = 0.0
        self._yaw_vel = 0.0

    def reset(self, pos: np.ndarray, yaw: float = 0.0) -> None:
        """Khởi tạo bộ lọc."""
        self.filter.reset(pos)
        self._missing_count = 0
        self._pose_valid = True
        self._last_yaw = yaw
        self._yaw_vel = 0.0

    def update(
        self,
        pos_measurement: np.ndarray | None,
        yaw_measurement: float | None,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Cập nhật perception.

        Parameters
        ----------
        pos_measurement : ndarray(3) hoặc None
            Vị trí đo được. None = mất dữ liệu.
        yaw_measurement : float hoặc None
            Yaw đo được. None = mất dữ liệu.

        Returns
        -------
        (pos_filtered, vel_filtered, yaw_filtered, pose_valid)
        """
        if pos_measurement is not None:
            pos, vel = self.filter.update(pos_measurement)
            self._missing_count = 0
            self._pose_valid = True
        else:
            # Mất dữ liệu: dùng dự đoán (vận tốc hằng)
            self._missing_count += 1
            pos = self.filter.position + self.filter.velocity * self.dt
            vel = self.filter.velocity
            self._pose_valid = self._missing_count <= self.max_missing_steps

        if yaw_measurement is not None:
            # Sai phân yaw đơn giản (chưa lọc mạnh)
            self._yaw_vel = (yaw_measurement - self._last_yaw) / self.dt
            self._last_yaw = yaw_measurement
        else:
            self._last_yaw += self._yaw_vel * self.dt

        return pos, vel, self._last_yaw, self._pose_valid
