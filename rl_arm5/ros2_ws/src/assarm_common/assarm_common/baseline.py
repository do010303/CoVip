# -*- coding: utf-8 -*-
"""
assarm_common.baseline — Baseline điều khiển tiếp cận bằng IK + Máy trạng thái GraspScript.

Dùng để thiết lập baseline đánh giá trước khi huấn luyện RL (Phase S).
Xem implementation_plan.md mục Phase S.
"""

from typing import Tuple, NamedTuple
import numpy as np

from assarm_common.config import (
    DT_CTRL, MAX_DELTA, HANDOFF_EX, HANDOFF_EY, HANDOFF_EZ,
    HANDOFF_TILT, HANDOFF_CONSEC_STEPS, Q_OPEN, Q_LO_ARM, Q_HI_ARM
)
from assarm_common.fk import fk_arm, p_close, gripper_tilt, jacobian_numerical
from assarm_common.grasp_script import GraspScript, GraspState, GraspCommand


class BaselineStepResult(NamedTuple):
    arm_target: np.ndarray
    gripper_target: float
    state: GraspState
    handoff_triggered: bool


class IKBaselineController:
    """
    Bộ điều khiển Baseline: Dùng IK số để đưa P_close tiếp cận vị trí khối,
    khi thỏa điều kiện bàn giao thì kích hoạt GraspScript.
    """

    def __init__(self, script: GraspScript | None = None, dt: float = DT_CTRL):
        self.script = script if script is not None else GraspScript(dt=dt)
        self.dt = dt
        self.consec_handoff_count = 0
        self.in_script_phase = False

    def reset(self, q_init: np.ndarray) -> None:
        """Reset bộ điều khiển và máy trạng thái script."""
        self.script.reset(q_init)
        self.consec_handoff_count = 0
        self.in_script_phase = False

    def step(
        self,
        q_feedback: np.ndarray,
        gripper_feedback: float,
        cube_pos_perceived: np.ndarray,
        cube_vel_perceived: np.ndarray | None = None,
        cube_yaw_perceived: float = 0.0,
    ) -> BaselineStepResult:
        """
        Một bước điều khiển Baseline (10 Hz).

        Parameters
        ----------
        q_feedback : ndarray(4)
            Góc feedback 4 khớp tay.
        gripper_feedback : float
            Góc feedback gripper.
        cube_pos_perceived : ndarray(3)
            Vị trí khối nhận từ perception.
        cube_vel_perceived : ndarray(3), optional
            Vận tốc khối từ perception.
        cube_yaw_perceived : float, optional
            Góc yaw của khối từ perception.

        Returns
        -------
        BaselineStepResult
        """
        # Nếu đã ở trong pha script (CLOSE -> VERIFY -> HOLD)
        if self.in_script_phase:
            cmd = self.script.step(q_feedback, gripper_feedback)
            return BaselineStepResult(
                arm_target=cmd.arm_target,
                gripper_target=cmd.gripper_target,
                state=cmd.state,
                handoff_triggered=False,
            )

        # 1. Tính toán P_close hiện tại và sai lệch tới khối trong hệ gripper
        p_cur = p_close(q_feedback)
        pos_err_world = cube_pos_perceived - p_cur  # vector sai số vị trí hệ base
        dist_err = np.linalg.norm(pos_err_world)

        _, R_grip = fk_arm(q_feedback)
        pos_err_grip = R_grip.T @ pos_err_world  # chuyển sang hệ gripper (ex, ey, ez)

        # 2. Kiểm tra độ nghiêng gripper
        tilt = gripper_tilt(q_feedback)

        # 3. Vận tốc tương đối
        v_rel = np.linalg.norm(cube_vel_perceived) if cube_vel_perceived is not None else 0.0

        # 4. Kiểm tra điều kiện bàn giao (handoff condition) trong hệ gripper
        is_aligned = (
            abs(pos_err_grip[0]) <= HANDOFF_EX and
            abs(pos_err_grip[1]) <= HANDOFF_EY and
            abs(pos_err_grip[2]) <= HANDOFF_EZ and
            tilt <= HANDOFF_TILT and
            v_rel <= 0.05
        )

        if is_aligned:
            self.consec_handoff_count += 1
        else:
            self.consec_handoff_count = 0

        # Nếu đạt đủ số step bàn giao liên tiếp -> Kích hoạt Script!
        if self.consec_handoff_count >= HANDOFF_CONSEC_STEPS:
            self.in_script_phase = True
            self.script.trigger_handoff(q_feedback)
            cmd = self.script.step(q_feedback, gripper_feedback)
            return BaselineStepResult(
                arm_target=cmd.arm_target,
                gripper_target=cmd.gripper_target,
                state=cmd.state,
                handoff_triggered=True,
            )

        # 5. Pha APPROACH (IK tiếp cận)
        # Tính delta step theo Jacobian cho vị trí
        J = jacobian_numerical(q_feedback)
        desired_dx = pos_err_world * min(1.0, 0.04 / max(dist_err, 1e-4))  # Tốc độ tiếp cận max 4 cm/step
        dq = np.linalg.solve(J.T @ J + 1e-3 * np.eye(4), J.T @ desired_dx)

        # Giữ q3 ≈ -q2 để duy trì tilt gần 0
        target_q3 = -q_feedback[1]
        dq[2] += 0.5 * (target_q3 - q_feedback[2])

        # Căn chỉnh Yaw bằng J4: target_q4 = -(cube_yaw_perceived + q1)
        target_q4 = -(cube_yaw_perceived + q_feedback[0])
        dq[3] = 0.5 * (target_q4 - q_feedback[3])

        # Kẹp max_delta
        dq = np.clip(dq, -MAX_DELTA, MAX_DELTA)
        arm_target = np.clip(q_feedback + dq, Q_LO_ARM, Q_HI_ARM)

        return BaselineStepResult(
            arm_target=arm_target,
            gripper_target=Q_OPEN,
            state=GraspState.APPROACH,
            handoff_triggered=False,
        )
