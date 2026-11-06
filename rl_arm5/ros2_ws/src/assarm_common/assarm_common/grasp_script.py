# -*- coding: utf-8 -*-
"""
assarm_common.grasp_script — Máy trạng thái gắp-nhấc-đặt.

Dùng chung sim và robot thật. Chỉ dùng feedback góc và perception,
KHÔNG dùng contact/pose thật (sim privileged info).

State machine (xem rl_recommendation_v5_0.md mục 1.2):
  APPROACH → CLOSE → VERIFY → LIFT → HOME → HOLD → PLACE → RELEASE → DONE
                       ↓                ↓
                    APPROACH          APPROACH (grip trượt)
                    (retry)

Giao diện: mỗi step 10 Hz gọi step() → nhận (arm_target, gripper_target, state).
"""

from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from assarm_common.config import (
    Q_STAR, EPS_GRIP, Q_OPEN, DT_CTRL,
    VERIFY_WINDOW_LO, VERIFY_WINDOW_HI, MAX_VERIFY_RETRIES,
)
from assarm_common.fk import jacobian_numerical


class GraspState(Enum):
    """Trạng thái máy gắp."""
    APPROACH = auto()
    CLOSE = auto()
    VERIFY = auto()
    LIFT = auto()
    HOME = auto()
    HOLD = auto()
    PLACE = auto()
    RELEASE = auto()
    DONE = auto()
    FAIL = auto()


class GraspCommand(NamedTuple):
    """Lệnh đầu ra của script mỗi step."""
    arm_target: np.ndarray     # (4,) target 4 khớp tay
    gripper_target: float      # target gripper (rad)
    state: GraspState          # trạng thái hiện tại


class GraspScript:
    """
    Máy trạng thái gắp-nhấc-đặt.

    Parameters
    ----------
    arm_q_home : ndarray(4)
        Cấu hình home (mặc định [0,0,0,0]).
    close_speed : float
        Tốc độ đóng gripper (rad/s), mặc định 0.8.
    lift_dh : float
        Chiều cao nhấc (m), mặc định 0.04.
    home_speed : float
        Tốc độ về home (rad/s), mặc định 0.5.
    hold_duration : float
        Thời gian giữ ở home (s), mặc định 3.0.
    stall_threshold : float
        Ngưỡng phát hiện bị chặn |target − q| (rad), mặc định 0.05.
    stall_count_needed : int
        Số mẫu liên tiếp vượt ngưỡng, mặc định 3.
    """

    def __init__(
        self,
        arm_q_home: np.ndarray | None = None,
        close_speed: float = 0.8,
        lift_dh: float = 0.04,
        home_speed: float = 0.5,
        hold_duration: float = 3.0,
        stall_threshold: float = 0.05,
        stall_count_needed: int = 3,
        dt: float = DT_CTRL,
    ):
        self.q_home = arm_q_home if arm_q_home is not None else np.zeros(4)
        self.close_speed = close_speed
        self.lift_dh = lift_dh
        self.home_speed = home_speed
        self.hold_duration = hold_duration
        self.stall_threshold = stall_threshold
        self.stall_count_needed = stall_count_needed
        self.dt = dt

        # State
        self.state = GraspState.APPROACH
        self.retry_count = 0
        self._arm_target = np.zeros(4)
        self._gripper_target = Q_OPEN
        self._step_in_state = 0
        self._stall_count = 0
        self._grip_stall_angle = 0.0
        self._hold_elapsed = 0.0

    def reset(self, arm_q: np.ndarray) -> None:
        """Reset máy trạng thái khi bắt đầu episode mới."""
        self.state = GraspState.APPROACH
        self.retry_count = 0
        self._arm_target = arm_q.copy()
        self._gripper_target = Q_OPEN
        self._step_in_state = 0
        self._stall_count = 0
        self._grip_stall_angle = 0.0
        self._hold_elapsed = 0.0

    def trigger_handoff(self, arm_q: np.ndarray) -> None:
        """Gọi khi điều kiện bàn giao đúng — chuyển sang CLOSE."""
        if self.state == GraspState.APPROACH:
            self.state = GraspState.CLOSE
            self._arm_target = arm_q.copy()
            self._gripper_target = Q_OPEN
            self._step_in_state = 0
            self._stall_count = 0

    def step(
        self,
        arm_q_feedback: np.ndarray,
        gripper_q_feedback: float,
    ) -> GraspCommand:
        """
        Một bước máy trạng thái (10 Hz).

        Parameters
        ----------
        arm_q_feedback : ndarray(4)
            Góc feedback 4 khớp tay.
        gripper_q_feedback : float
            Góc feedback gripper (rad).

        Returns
        -------
        GraspCommand
        """
        self._step_in_state += 1

        if self.state == GraspState.CLOSE:
            return self._step_close(arm_q_feedback, gripper_q_feedback)
        elif self.state == GraspState.VERIFY:
            return self._step_verify(gripper_q_feedback)
        elif self.state == GraspState.LIFT:
            return self._step_lift(arm_q_feedback)
        elif self.state == GraspState.HOME:
            return self._step_home(arm_q_feedback)
        elif self.state == GraspState.HOLD:
            return self._step_hold(arm_q_feedback, gripper_q_feedback)
        elif self.state == GraspState.RELEASE:
            return self._step_release(arm_q_feedback)
        else:
            # DONE, FAIL, APPROACH (RL đang điều khiển)
            return GraspCommand(self._arm_target, self._gripper_target, self.state)

    # === Private state step methods ===

    def _step_close(self, arm_q: np.ndarray, grip_q: float) -> GraspCommand:
        """Nội suy gripper đóng, phát hiện bị chặn."""
        target_close = Q_STAR + EPS_GRIP
        self._gripper_target = min(
            self._gripper_target + self.close_speed * self.dt,
            target_close,
        )
        # Phát hiện bị chặn: kiểm tra khi target tới gần Q_STAR (khối 5 cm)
        error = abs(self._gripper_target - grip_q)
        if self._gripper_target >= (Q_STAR - 0.05) and error > self.stall_threshold:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= self.stall_count_needed:
            # Gripper bị chặn bởi khối → ghi góc chặn → VERIFY
            self._grip_stall_angle = grip_q
            self.state = GraspState.VERIFY
            self._step_in_state = 0
        elif self._gripper_target >= target_close:
            # Đóng hết mà không bị chặn → chuyển sang VERIFY kiểm tra
            self._grip_stall_angle = grip_q
            self.state = GraspState.VERIFY
            self._step_in_state = 0

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    def _step_verify(self, grip_q: float) -> GraspCommand:
        """Kiểm tra góc chặn nằm trong cửa sổ hợp lệ."""
        # Chờ gripper ổn định (2 bước)
        if self._step_in_state < 2:
            return GraspCommand(self._arm_target, self._gripper_target, self.state)

        q_stall = self._grip_stall_angle
        if VERIFY_WINDOW_LO <= q_stall <= VERIFY_WINDOW_HI:
            # Kẹp đúng → đứt dây, giữ nguyên vị trí kẹp hiện tại (HOLD)
            self.state = GraspState.HOLD
            self._step_in_state = 0
            self._hold_elapsed = 0.0
        elif self.retry_count < MAX_VERIFY_RETRIES:
            # Kẹp sai → mở lại, quay về APPROACH
            self.retry_count += 1
            self._gripper_target = Q_OPEN
            self.state = GraspState.APPROACH
            self._step_in_state = 0
        else:
            # Hết lượt thử → FAIL
            self._gripper_target = Q_OPEN
            self.state = GraspState.FAIL

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    def _step_lift(self, arm_q: np.ndarray) -> GraspCommand:
        """Nhấc tâm kẹp lên Δh bằng IK số (Jacobian)."""
        # Mục tiêu: dịch lên Δh, giữ hướng
        # Dùng Jacobian để tính delta q cho delta z = lift_speed * dt
        lift_speed = self.lift_dh / 1.0  # nhấc trong ~1s
        dz_step = lift_speed * self.dt

        J = jacobian_numerical(self._arm_target)
        # Pseudo-inverse: dq = J^+ · [0, 0, dz_step]
        desired_dx = np.array([0.0, 0.0, dz_step])
        dq = np.linalg.lstsq(J, desired_dx, rcond=None)[0]

        # Giới hạn bước
        max_dq = 0.1  # rad/step
        dq = np.clip(dq, -max_dq, max_dq)
        self._arm_target = self._arm_target + dq

        # Kiểm tra đã nhấc đủ (heuristic: ~10 step ở 10 Hz)
        if self._step_in_state >= 10:
            self.state = GraspState.HOME
            self._step_in_state = 0

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    def _step_home(self, arm_q: np.ndarray) -> GraspCommand:
        """Nội suy 4 khớp về home với tốc độ giới hạn."""
        diff = self.q_home - self._arm_target
        max_step = self.home_speed * self.dt
        step = np.clip(diff, -max_step, max_step)
        self._arm_target = self._arm_target + step

        # Kiểm tra đã về home (dung sai 0.02 rad)
        if np.all(np.abs(self._arm_target - self.q_home) < 0.02):
            self._arm_target = self.q_home.copy()
            self.state = GraspState.HOLD
            self._step_in_state = 0
            self._hold_elapsed = 0.0

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    def _step_hold(self, arm_q: np.ndarray, grip_q: float) -> GraspCommand:
        """Giữ ở home ≥ hold_duration giây."""
        self._hold_elapsed += self.dt

        # Kiểm tra khối trượt: nếu gripper mở bất thường → FAIL
        if grip_q < VERIFY_WINDOW_LO - 0.1:
            # Khối đã tuột
            self.state = GraspState.FAIL
            return GraspCommand(self._arm_target, self._gripper_target, self.state)

        if self._hold_elapsed >= self.hold_duration:
            self.state = GraspState.RELEASE
            self._step_in_state = 0

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    def _step_release(self, arm_q: np.ndarray) -> GraspCommand:
        """Mở gripper."""
        self._gripper_target = Q_OPEN
        if self._step_in_state >= 5:  # chờ 0.5s
            self.state = GraspState.DONE

        return GraspCommand(self._arm_target, self._gripper_target, self.state)

    @property
    def is_terminal(self) -> bool:
        """True nếu đã kết thúc (DONE hoặc FAIL)."""
        return self.state in (GraspState.DONE, GraspState.FAIL)

    @property
    def success(self) -> bool:
        """True nếu gắp thành công."""
        return self.state == GraspState.DONE
