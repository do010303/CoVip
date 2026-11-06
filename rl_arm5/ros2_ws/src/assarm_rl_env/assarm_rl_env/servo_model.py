# -*- coding: utf-8 -*-
"""
assarm_rl_env.servo_model — Mô hình phi tuyến của servo.

Đặt giữa action RL (target) và lệnh gửi vào Gazebo controller.
Mô phỏng:
- Vùng chết (deadband)
- Độ rơ (backlash)
- Giới hạn tốc độ theo tải
- Trễ nội suy (plugin `position_proportional_gain`)

LƯU Ý: position_proportional_gain = 0.1 trong gz_ros2_control plugin
tạo hằng số thời gian ~0.1s ở 100 Hz update. ServoModel KHÔNG cộng thêm
trễ vì plugin đã có. ServoModel chỉ giữ phi tuyến (vùng chết, rơ, tốc độ).

Tham số khởi đầu — cần hiệu chỉnh từ đo thật (Phase R).
"""

import numpy as np

from assarm_common.config import Q_LO_ARM, Q_HI_ARM


class ServoModel:
    """
    Mô hình phi tuyến servo giữa target RL và lệnh gửi Gazebo.

    Parameters
    ----------
    deadband : ndarray(4)
        Vùng chết mỗi khớp (rad). Mặc định ~0.5–1°.
    backlash : ndarray(4)
        Độ rơ mỗi khớp (rad). Mặc định ~1–2°.
    max_speed : ndarray(4)
        Giới hạn tốc độ (rad/s). Mặc định theo servo thật.
    """

    def __init__(
        self,
        deadband: np.ndarray | None = None,
        backlash: np.ndarray | None = None,
        max_speed: np.ndarray | None = None,
        dt: float = 0.1,
    ):
        # Vùng chết: 0.5° cho J1–J3, 1.0° cho J4 (ảnh hưởng yaw)
        self.deadband = (
            deadband if deadband is not None
            else np.radians([0.5, 0.5, 0.5, 1.0])
        )
        # Độ rơ: 1° cho J1–J3, 2° cho J4
        self.backlash = (
            backlash if backlash is not None
            else np.radians([1.0, 1.0, 1.0, 2.0])
        )
        # Tốc độ tối đa (rad/s) — ước tính từ datasheet 60°/0.16s
        self.max_speed = (
            max_speed if max_speed is not None
            else np.array([6.5, 6.5, 6.5, 6.5])
        )
        self.dt = dt
        self._prev_cmd = np.zeros(4)
        self._backlash_state = np.zeros(4)  # hướng chuyển động gần nhất

    def reset(self, q_init: np.ndarray) -> None:
        """Reset mô hình servo."""
        self._prev_cmd = q_init.copy()
        self._backlash_state = np.zeros(4)

    def apply(self, target: np.ndarray) -> np.ndarray:
        """
        Áp dụng mô hình phi tuyến lên target.

        Parameters
        ----------
        target : ndarray(4)
            Target position từ RL.

        Returns
        -------
        ndarray(4)
            Lệnh gửi vào Gazebo controller.
        """
        cmd = target.copy()

        # 1. Giới hạn tốc độ
        delta = cmd - self._prev_cmd
        max_delta = self.max_speed * self.dt
        delta = np.clip(delta, -max_delta, max_delta)
        cmd = self._prev_cmd + delta

        # 2. Vùng chết: nếu thay đổi nhỏ hơn deadband, giữ nguyên
        small_change = np.abs(cmd - self._prev_cmd) < self.deadband
        cmd = np.where(small_change, self._prev_cmd, cmd)

        # 3. Backlash: thêm offset ngẫu nhiên nhỏ mô phỏng rơ cơ khí
        # (đơn giản hóa: không mô hình đầy đủ hysteresis)
        # TODO: Mô hình backlash đầy đủ sau khi đo (Phase R)

        # 4. Kẹp giới hạn khớp
        cmd = np.clip(cmd, Q_LO_ARM, Q_HI_ARM)

        self._prev_cmd = cmd.copy()
        return cmd

    def step(self, target: np.ndarray) -> np.ndarray:
        """Alias cho apply(target)."""
        return self.apply(target)


class ServoModelDR:
    """
    ServoModel wrapper với domain randomization.

    Mỗi episode reset sẽ ngẫu nhiên hóa các tham số servo.
    """

    def __init__(self, dt: float = 0.1, rng: np.random.Generator | None = None):
        self.dt = dt
        self.rng = rng or np.random.default_rng()
        self.model = ServoModel(dt=dt)

    def randomize(self) -> None:
        """Ngẫu nhiên hóa tham số servo cho episode mới."""
        # Vùng chết: 0.3–1.5°
        self.model.deadband = np.radians(
            self.rng.uniform(0.3, 1.5, size=4)
        )
        # Độ rơ: 0.5–3°
        self.model.backlash = np.radians(
            self.rng.uniform(0.5, 3.0, size=4)
        )
        # Tốc độ: ±20%
        self.model.max_speed = np.array([6.5, 6.5, 6.5, 6.5]) * self.rng.uniform(
            0.8, 1.2, size=4
        )

    def reset(self, q_init: np.ndarray) -> None:
        self.randomize()
        self.model.reset(q_init)

    def apply(self, target: np.ndarray) -> np.ndarray:
        return self.model.apply(target)
