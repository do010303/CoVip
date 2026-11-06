# -*- coding: utf-8 -*-
"""
assarm_rl_env.assarm_env — Gymnasium Environment cho bài toán gắp con lắc ASSARM.

Đáp ứng đầy đủ đặc tả Phase 1 (Stage A) và Phase 2 (Stage B):
- Observation space: Box(28,) chuẩn hóa thủ công
- Action space: Box(4,) delta target 4 khớp tay (-1.0 -> +1.0)
- Reward: Dense distance shaping + penalities (yaw, tilt, speed, rate, contact)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from assarm_common.config import (
    Q_LO_ARM, Q_HI_ARM, MAX_DELTA, EPS_ARM, DT_CTRL,
    MAX_STEPS_STAGE_A, MAX_STEPS_STAGE_B, SPAWN_LEVELS
)
from assarm_common.fk import fk_arm, p_close, gripper_yaw, gripper_tilt, wrap_to_pm45
from assarm_common.obs_builder import build_obs
from assarm_common.perception_filter import PerceptionFilter
from assarm_rl_env.sim_interface import FastSimEngine
from assarm_rl_env.reward import compute_reward, compute_terminal_reward


class AssarmGraspEnv(gym.Env):
    """
    Gymnasium Environment cho bài toán gắp con lắc của robot tay 4-DOF ASSARM.

    Parameters
    ----------
    spawn_level : int
        Level spawn (0 hoặc 1 cho Stage A; 2 hoặc 3 cho Stage B). Mặc định 0.
    max_steps : int, optional
        Số bước tối đa mỗi episode. Mặc định 60 (Stage A).
    dt : float
        Chu kỳ điều khiển (s), mặc định 0.1s (10 Hz).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        spawn_level: int = 0,
        max_steps: int = MAX_STEPS_STAGE_A,
        dt: float = DT_CTRL,
        dynamic_motion: bool = False,
        motion_stage: int = 1,
    ):
        super().__init__()

        self.spawn_level = spawn_level
        self.max_steps = max_steps
        self.dt = dt
        self.dynamic_motion = dynamic_motion
        self.motion_stage = motion_stage

        # Spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Sim Engine & Perception Filter
        self.sim = FastSimEngine(dt=dt)
        self.sim.motion_stage = self.motion_stage
        self.perception_filter = PerceptionFilter(dt=dt)

        # Internal state
        self.step_count = 0
        self.q_prev = np.zeros(4)
        self.prev_action = np.zeros(4)
        self.target_q = np.zeros(4)

    def set_motion_stage(self, stage: int) -> None:
        """Cập nhật giai đoạn di chuyển cho sim engine (1: Cố định danh định | 2: Ngẫu nhiên + Nhiễu)."""
        self.motion_stage = int(stage)
        self.sim.set_motion_stage(stage)

    def set_target_velocity(self, v_mps: float) -> None:
        """Cập nhật vận tốc mục tiêu cố định cho sim engine (m/s)."""
        self.sim.set_target_velocity(v_mps)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset môi trường khi bắt đầu episode mới."""
        super().reset(seed=seed)

        if options and "spawn_level" in options:
            level = options["spawn_level"]
        elif isinstance(self.spawn_level, (list, tuple)):
            level = int(np.random.choice(self.spawn_level))
        elif self.spawn_level in (-1, "all"):
            level = int(np.random.choice([1, 2, 3]))
        else:
            level = self.spawn_level

        # Hỗ trợ bật/tắt dynamic_motion qua options
        dyn = self.dynamic_motion
        if options and "dynamic_motion" in options:
            dyn = bool(options["dynamic_motion"])

        # Hỗ trợ cấu hình motion_stage qua options
        if options and "motion_stage" in options:
            self.set_motion_stage(options["motion_stage"])

        # Reset sim engine & perception
        q_init, cube_pos = self.sim.reset(spawn_level=level, seed=seed, dynamic_motion=dyn)
        self.perception_filter.reset(cube_pos, self.sim.cube_yaw)

        self.step_count = 0
        self.q_prev = q_init.copy()
        self.prev_action = np.zeros(4)
        self.target_q = q_init.copy()

        obs = self._get_obs(pose_valid=True)
        info = {
            "spawn_level": level,
            "cube_pos": cube_pos,
            "q_init": q_init,
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Một bước môi trường (10 Hz).

        Parameters
        ----------
        action : ndarray(4)
            Action từ policy (-1.0 -> +1.0).

        Returns
        -------
        obs : ndarray(28)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # 1. Update target: clip delta và clip giới hạn khớp
        # Dynamic Action Scaling: thu nhỏ bước lệnh khi d < 5 cm để tăng độ phân giải vi mô (tới 20%)
        p_prev = p_close(self.sim.q_meas)
        d_prev = float(np.linalg.norm(self.sim.cube_pos - p_prev))
        scale = float(np.clip(d_prev / 0.05, 0.2, 1.0))
        delta_q = action * MAX_DELTA * scale
        self.target_q = np.clip(self.target_q + delta_q, Q_LO_ARM, Q_HI_ARM)
        # Khóa an toàn: target không được lệch quá EPS_ARM so với q_meas
        self.target_q = np.clip(self.target_q, self.sim.q_meas - EPS_ARM, self.sim.q_meas + EPS_ARM)

        # 2. Sim step (mô phỏng servo model & pendulum)
        q_meas, cube_pos_true, cube_vel_true, handoff_ok = self.sim.step(self.target_q)

        # 3. Perception filter (lọc vận tốc khối)
        obj_pos_perceived, obj_vel_filtered, yaw_perceived, pose_valid = self.perception_filter.update(
            cube_pos_true, self.sim.cube_yaw
        )

        # 4. Read observation
        obs = self._get_obs(pose_valid=pose_valid)

        # 5. Reward calculation (chỉ số thực của sim cho ground truth reward)
        p_cur = p_close(q_meas)
        d = float(np.linalg.norm(cube_pos_true - p_cur))

        _, R_grip = fk_arm(q_meas)
        yaw_jaw = gripper_yaw(q_meas)
        dpsi = wrap_to_pm45(self.sim.cube_yaw - yaw_jaw)
        tilt = gripper_tilt(q_meas)
        tilt_deg = float(np.degrees(tilt))

        # Vận tốc của đầu kẹp (m/s) qua sai phân vị trí
        p_prev = p_close(self.q_prev)
        v_gripper = (p_cur - p_prev) / self.dt
        # Vận tốc tương đối thực tế giữa ngàm kẹp và khối vật
        v_rel = v_gripper - cube_vel_true
        rel_speed = float(np.linalg.norm(v_rel))
        # Độ đồng pha vận tốc: v_match > 0 khi đầu kẹp di chuyển cùng hướng với vật
        v_match = float(np.dot(v_gripper, cube_vel_true))

        # Check contact ngón–khối không mong muốn khi ngón đang mở:
        # Miệng kẹp mở rộng 8.0 cm, khối 5.0 cm -> khe hở mỗi bên 1.5 cm theo trục x gripper.
        # Va chạm chỉ xảy ra khi ngón đã hạ bao quanh khối (|ez| < 0.025 m) mà bị lệch ngang |ex| > 0.015 m
        jaw_contact = False
        pos_err_grip = R_grip.T @ (cube_pos_true - p_cur)
        if abs(pos_err_grip[0]) > 0.015 and abs(pos_err_grip[2]) < 0.025 and not handoff_ok:
            jaw_contact = True

        # Hướng miệng ngàm chĩa thẳng vào khối trụ (-Z của Plate_1 chĩa về cube_pos_true):
        pos_plate, R_grip = fk_arm(q_meas)
        v_cavity = -R_grip[:, 2]
        vec_to_obj = cube_pos_true - pos_plate
        dist_plate_to_obj = float(np.linalg.norm(vec_to_obj))
        if dist_plate_to_obj > 1e-4:
            cavity_align = float(np.dot(v_cavity, vec_to_obj / dist_plate_to_obj))
        else:
            cavity_align = 1.0

        # Sai số cao độ |P_close,z - cube_pos,z|:
        dz_err = float(abs(p_cur[2] - cube_pos_true[2]))

        step_reward = compute_reward(
            d=d,
            dpsi=dpsi,
            tilt_deg=tilt_deg,
            rel_speed=rel_speed,
            jaw_contact=jaw_contact,
            action=action,
            prev_action=self.prev_action,
            cavity_align=cavity_align,
            dz_err=dz_err,
            v_match=v_match,
            q_j4=float(q_meas[3]),
        )

        reward = step_reward
        terminated = False
        truncated = False
        good_handoff = False

        # 6. Terminal & Handoff condition
        if handoff_ok:
            terminated = True
            # Kiểm tra chất lượng bàn giao dựa trên dung sai cơ học chuẩn xác của ngàm kẹp
            good_handoff = (
                abs(pos_err_grip[0]) <= 0.015 and
                abs(pos_err_grip[1]) <= 0.022 and
                abs(pos_err_grip[2]) <= 0.025 and
                rel_speed <= 0.05 and
                cavity_align >= 0.70
            )
            term_reward = compute_terminal_reward(good_handoff)
            reward += term_reward

        elif self.step_count >= self.max_steps:
            truncated = True

        # Ghi nhận thông số bước (bao gồm is_success cho SB3 Monitor wrapper)
        info = {
            "handoff_ok": handoff_ok,
            "good_handoff": good_handoff,
            "is_success": bool(handoff_ok and good_handoff),
            "distance": d,
            "tilt_deg": tilt_deg,
            "dpsi_deg": np.degrees(dpsi),
            "step_reward": step_reward,
        }

        self.q_prev = q_meas.copy()
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def _get_obs(self, pose_valid: bool = True) -> np.ndarray:
        """Tạo vector observation 28D chuẩn hóa."""
        obj_pos = self.perception_filter.filter.position
        obj_vel = self.perception_filter.filter.velocity

        _, R_grip = fk_arm(self.sim.q_meas)
        p_cur = p_close(self.sim.q_meas)
        yaw_jaw = gripper_yaw(self.sim.q_meas)
        tilt = gripper_tilt(self.sim.q_meas)

        return build_obs(
            q_arm=self.sim.q_meas,
            q_prev=self.q_prev,
            target=self.target_q,
            p_close=p_cur,
            obj_pos=obj_pos,
            obj_vel_filtered=obj_vel,
            R_gripper=R_grip,
            yaw_obj=self.sim.cube_yaw,
            yaw_jaw=yaw_jaw,
            tilt=tilt,
            pose_valid=pose_valid,
            dt=self.dt,
        )
