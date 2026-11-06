# -*- coding: utf-8 -*-
"""
assarm_rl_env.sim_interface — Giao diện mô phỏng (SimEngine & ROS2 Bridge).

Hỗ trợ 2 chế độ:
1. Fast Simulation Engine (Thuần NumPy/Kinematics + ServoModel + Pendulum):
   - Đạt tốc độ > 200 step/s trên CPU i5-1035G4 cho huấn luyện SAC 100k–300k steps.
2. ROS2 Gazebo Bridge:
   - Kết nối với node Gazebo / ros_gz_bridge thông qua ROS2 topics.
"""

from typing import Tuple, Dict, Any
import numpy as np

from assarm_common.config import (
    DT_CTRL, DT_PHYS, SIM_SUBSTEPS, SPAWN_LEVELS,
    HANDOFF_EX, HANDOFF_EY, HANDOFF_EZ, HANDOFF_DPSI, HANDOFF_TILT, HANDOFF_REL_SPEED, HANDOFF_CONSEC_STEPS
)
from assarm_common.fk import fk_arm, p_close, gripper_yaw, gripper_tilt, wrap_to_pm45
from assarm_rl_env.servo_model import ServoModel


class FastSimEngine:
    """
    Trình mô phỏng vật lý & động học tốc độ cao cho Gymnasium Env.

    Mô phỏng động học cánh tay 4-DOF, mô hình phi tuyến servo ESP32,
    và chuyển động con lắc dây 30 cm của khối lập phương.
    """

    def __init__(self, dt: float = DT_CTRL):
        self.dt = dt
        self.servo_model = ServoModel(dt=dt)

        # Trạng thái tay & khối
        self.q_meas = np.zeros(4)
        self.q_target = np.zeros(4)
        self.gripper_meas = 0.0

        # Trạng thái khối con lắc (Mặc định lệch trái X=-0.10m, Y=0.28m, Z=0.40m kẹp ngang)
        self.cube_base_pos = np.array([-0.10, 0.28, 0.40])
        self.swing_amp = 0.0
        self.swing_freq = 5.72  # rad/s (~0.91 Hz với L = 0.3m)
        self.swing_phase = 0.0
        self.cube_pos = self.cube_base_pos.copy()
        self.cube_vel = np.zeros(3)
        self.cube_yaw = 0.0

        # Cấu hình Stage B+ Dynamic Drift (Vận tốc ngẫu nhiên 1.2 - 2.0 cm/s trong FOV)
        self.dynamic_motion = False
        self.motion_stage = 1  # 1: Quỹ đạo Cố định danh định | 2: Ngẫu nhiên hóa toàn diện + Nhiễu bất định
        self.v_target_fixed = 0.016  # m/s — Tốc độ cố định mặc định (có thể thay đổi qua set_target_velocity)
        self.speed_range = (0.012, 0.020)  # m/s (1.2 - 2.0 cm/s)
        self.wind_noise_state = np.zeros(2)
        self.arm_len = 0.035
        self.omega_cross = 1.0
        self.t_stroke = 3.0
        self.t_cycle = 12.0
        self.start_stroke = 0

        # Bộ đếm step
        self.sim_time = 0.0
        self.consec_handoff_steps = 0

    def reset(self, spawn_level: int = 0, seed: int | None = None, dynamic_motion: bool | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset trạng thái khối và robot về vị trí ban đầu.

        Parameters
        ----------
        spawn_level : int
            Cấp độ spawn (0..3).
        seed : int, optional
            Hạt giống ngẫu nhiên.
        dynamic_motion : bool, optional
            Bật/tắt chuyển động ngẫu nhiên động (Stage B+ 1.2 - 2.0 cm/s).
        """
        if seed is not None:
            np.random.seed(seed)

        if dynamic_motion is not None:
            self.dynamic_motion = dynamic_motion

        params = SPAWN_LEVELS.get(spawn_level, SPAWN_LEVELS[0])

        if self.dynamic_motion and self.motion_stage == 1:
            # GIAI ĐOẠN 1 (0 - 30k steps): Góc hẹp quanh tâm [-10°, +10°], v = 0.8 cm/s
            r = np.random.uniform(0.27, 0.30)
            theta = np.random.uniform(-np.radians(10), np.radians(10))
            z = np.clip(np.random.uniform(0.38, 0.42), 0.36, 0.44)
            self.cube_base_pos = np.array([r * np.sin(theta), r * np.cos(theta), z], dtype=np.float64)
            self.cube_yaw = 0.0
            self.v_target = self.v_target_fixed if self.v_target_fixed > 0 else 0.008
            self.arm_len = 0.035
            self.omega_cross = 2.0 * self.v_target / self.arm_len
            self.t_stroke = 2.0 * np.pi / self.omega_cross
            self.t_cycle = 4.0 * self.t_stroke
            self.start_stroke = 0  # Luôn bắt đầu từ tâm theo nhánh +X
            self.wind_noise_state = np.zeros(2)
            self.swing_amp = 0.0
        elif self.dynamic_motion and self.motion_stage == 2:
            # GIAI ĐOẠN 2 (30k - 70k steps): Mở rộng góc trung bình [-22°, +22°], v = 1.4 cm/s
            r = np.random.uniform(0.26, 0.31)
            theta = np.random.uniform(-np.radians(22), np.radians(22))
            z = np.clip(np.random.uniform(0.37, 0.43), 0.36, 0.44)
            self.cube_base_pos = np.array([r * np.sin(theta), r * np.cos(theta), z], dtype=np.float64)
            self.cube_yaw = 0.0
            self.v_target = self.v_target_fixed if self.v_target_fixed > 0 else 0.014
            self.arm_len = 0.035
            self.omega_cross = 2.0 * self.v_target / self.arm_len
            self.t_stroke = 2.0 * np.pi / self.omega_cross
            self.t_cycle = 4.0 * self.t_stroke
            self.start_stroke = int(np.random.randint(0, 4))
            self.wind_noise_state = np.zeros(2)
            self.swing_amp = 0.0
        elif self.dynamic_motion and self.motion_stage == 3:
            # GIAI ĐOẠN 3 (70k - 120k steps): Toàn dải Level 3 [-35°, +35°], v = 1.6 cm/s chuẩn tắc
            r = np.random.uniform(params["r"][0], params["r"][1])
            theta = np.random.uniform(-np.radians(35), np.radians(35))
            z = np.clip(np.random.uniform(params["z"][0], params["z"][1]), 0.36, 0.44)
            self.cube_base_pos = np.array([r * np.sin(theta), r * np.cos(theta), z], dtype=np.float64)
            self.cube_yaw = float(np.random.uniform(-params["yaw"], params["yaw"])) if params["yaw"] > 0 else 0.0
            self.v_target = self.v_target_fixed if self.v_target_fixed > 0 else 0.016
            self.arm_len = 0.035
            self.omega_cross = 2.0 * self.v_target / self.arm_len
            self.t_stroke = 2.0 * np.pi / self.omega_cross
            self.t_cycle = 4.0 * self.t_stroke
            self.start_stroke = int(np.random.randint(0, 4))
            self.wind_noise_state = np.zeros(2)
            self.swing_amp = 0.0
        elif self.dynamic_motion and self.motion_stage >= 4:
            # GIAI ĐOẠN 4 (> 120k steps): Toàn dải Level 3 [-35°, +35°] + Vận tốc ngẫu nhiên 1.2-2.0 cm/s + Nhiễu gió OU
            r = np.random.uniform(params["r"][0], params["r"][1])
            theta = np.random.uniform(-np.radians(35), np.radians(35))
            z = np.clip(np.random.uniform(params["z"][0], params["z"][1]), 0.36, 0.44)
            self.cube_base_pos = np.array([r * np.sin(theta), r * np.cos(theta), z], dtype=np.float64)
            self.cube_yaw = float(np.random.uniform(-params["yaw"], params["yaw"])) if params["yaw"] > 0 else 0.0
            self.v_target = float(np.random.uniform(self.speed_range[0], self.speed_range[1]))
            self.arm_len = 0.035
            self.omega_cross = 2.0 * self.v_target / self.arm_len
            self.t_stroke = 2.0 * np.pi / self.omega_cross
            self.t_cycle = 4.0 * self.t_stroke
            self.start_stroke = int(np.random.randint(0, 4))  # Ngẫu nhiên nhánh xuất phát
            self.wind_noise_state = np.zeros(2)
            self.swing_amp = 0.0
        else:
            # Chế độ tĩnh mặc định
            r = np.random.uniform(params["r"][0], params["r"][1])
            theta = np.random.uniform(-params["theta"], params["theta"])
            z = np.clip(np.random.uniform(params["z"][0], params["z"][1]), 0.36, 0.44)
            self.cube_base_pos = np.array([r * np.sin(theta), r * np.cos(theta), z], dtype=np.float64)
            self.cube_yaw = np.random.uniform(-params["yaw"], params["yaw"])
            self.v_target = 0.0
            self.arm_len = 0.0
            self.swing_amp = 0.0
            self.swing_phase = 0.0

        self.sim_time = 0.0
        self._update_cube_pose()

        # Góc khớp khởi tạo của robot: bắt đầu từ vị trí HOME chuẩn [0, 0, 0, 0]
        # (đồng bộ 100% với môi trường Deploy thực tế trên Gazebo)
        self.q_meas = np.zeros(4, dtype=np.float64)
        self.q_target = self.q_meas.copy()
        self.servo_model.reset(self.q_meas)
        self.gripper_meas = 0.0
        self.consec_handoff_steps = 0

        return self.q_meas.copy(), self.cube_pos.copy()

    def step(self, action_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Một bước mô phỏng (10 Hz).

        Parameters
        ----------
        action_target : ndarray(4)
            Target 4 khớp tay gửi từ policy.

        Returns
        -------
        q_meas : ndarray(4)
            Góc feedback khớp đo được từ servo model.
        cube_pos : ndarray(3)
            Vị trí thật của khối.
        cube_vel : ndarray(3)
            Vận tốc thật của khối.
        is_aligned : bool
            True nếu điều kiện bàn giao được thỏa mãn.
        """
        self.sim_time += self.dt
        self.q_target = action_target.copy()

        # Update servo response (vùng chết, rơ, trễ)
        self.q_meas = self.servo_model.step(self.q_target)

        # Update pendulum cube pose
        self._update_cube_pose()

        # Check alignment condition
        is_aligned = self._check_alignment()
        if is_aligned:
            self.consec_handoff_steps += 1
        else:
            self.consec_handoff_steps = 0

        handoff_ok = (self.consec_handoff_steps >= HANDOFF_CONSEC_STEPS)

        return self.q_meas.copy(), self.cube_pos.copy(), self.cube_vel.copy(), handoff_ok

    def set_motion_stage(self, stage: int) -> None:
        """Chuyển đổi giai đoạn: 1 (Cố định danh định) hoặc 2 (Ngẫu nhiên hóa + Nhiễu)."""
        self.motion_stage = int(stage)

    def set_target_velocity(self, v_mps: float) -> None:
        """Đặt tốc độ mục tiêu cố định cho motion_stage=1 (m/s)."""
        self.v_target_fixed = float(v_mps)

    def _update_cube_pose(self) -> None:
        """Cập nhật vị trí & vận tốc khối con lắc / chuyển động hình dấu cộng (+) 4 nhánh."""
        if self.dynamic_motion:
            t = self.sim_time
            t_mod = (t + self.start_stroke * self.t_stroke) % self.t_cycle
            stroke = int(t_mod // self.t_stroke)
            t_s = t_mod - stroke * self.t_stroke

            s = 0.5 * (1.0 - np.cos(self.omega_cross * t_s))
            s_dot = 0.5 * self.omega_cross * np.sin(self.omega_cross * t_s)

            if stroke == 0:    # Nhánh +X (Phải)
                dx = self.arm_len * s
                dy = 0.0
                vx = self.arm_len * s_dot
                vy = 0.0
            elif stroke == 1:  # Nhánh +Y (Vươn xa)
                dx = 0.0
                dy = self.arm_len * s
                vx = 0.0
                vy = self.arm_len * s_dot
            elif stroke == 2:  # Nhánh -X (Trái)
                dx = -self.arm_len * s
                dy = 0.0
                vx = -self.arm_len * s_dot
                vy = 0.0
            else:              # Nhánh -Y (Lùi gần)
                dx = 0.0
                dy = -self.arm_len * s
                vx = 0.0
                vy = -self.arm_len * s_dot

            cur_pos = self.cube_base_pos + np.array([dx, dy, 0.0])

            if self.motion_stage >= 4:
                # Giai đoạn 4: Bổ sung nhiễu rung lắc / gió nhẹ ngẫu nhiên (Ornstein-Uhlenbeck drift)
                noise_step = np.random.normal(0.0, 0.0005, size=2)
                self.wind_noise_state = 0.95 * self.wind_noise_state + noise_step
                self.wind_noise_state = np.clip(self.wind_noise_state, -0.005, 0.005)
                cur_pos[0] += self.wind_noise_state[0]
                cur_pos[1] += self.wind_noise_state[1]
                vx += self.wind_noise_state[0] * 0.5
                vy += self.wind_noise_state[1] * 0.5

            # Ràng buộc an toàn: luôn nằm trong vùng làm việc và Camera FOV
            r_cur = float(np.hypot(cur_pos[0], cur_pos[1]))
            theta_cur = float(np.arctan2(cur_pos[0], cur_pos[1]))
            r_clamped = float(np.clip(r_cur, 0.25, 0.33))
            theta_clamped = float(np.clip(theta_cur, -np.radians(35), np.radians(35)))

            clamped_x = r_clamped * np.sin(theta_clamped)
            clamped_y = r_clamped * np.cos(theta_clamped)
            self.cube_pos = np.array([clamped_x, clamped_y, cur_pos[2]])
            self.cube_vel = np.array([vx, vy, 0.0])
        elif self.swing_amp <= 1e-5:
            self.cube_pos = self.cube_base_pos.copy()
            self.cube_vel = np.zeros(3)
        else:
            t = self.sim_time
            dx = self.swing_amp * np.cos(self.swing_freq * t + self.swing_phase)
            dy = self.swing_amp * np.sin(self.swing_freq * t + self.swing_phase)

            self.cube_pos = self.cube_base_pos + np.array([dx, dy, 0.0])

            vx = -self.swing_amp * self.swing_freq * np.sin(self.swing_freq * t + self.swing_phase)
            vy = self.swing_amp * self.swing_freq * np.cos(self.swing_freq * t + self.swing_phase)
            self.cube_vel = np.array([vx, vy, 0.0])

    def _check_alignment(self) -> bool:
        """Kiểm tra điều kiện bàn giao (handoff) dựa trên pose chuẩn của sim."""
        p_cur = p_close(self.q_meas)
        pos_err_world = self.cube_pos - p_cur

        _, R_grip = fk_arm(self.q_meas)
        pos_err_grip = R_grip.T @ pos_err_world
        v_rel = np.linalg.norm(self.cube_vel)

        # Với khối trụ tròn: đối xứng xoay 360°, ngàm ôm mặt cong
        # Cần sai số 3D trong dung sai, vận tốc tương đối nhỏ và miệng ngàm phải chĩa vào khối trụ (cavity_align >= 0.70)
        pos_plate, _ = fk_arm(self.q_meas)
        v_cavity = -R_grip[:, 2]
        vec_to_obj = self.cube_pos - pos_plate
        dist_plate = np.linalg.norm(vec_to_obj)
        cavity_align = np.dot(v_cavity, vec_to_obj / dist_plate) if dist_plate > 1e-4 else 1.0

        return (
            abs(pos_err_grip[0]) <= HANDOFF_EX and
            abs(pos_err_grip[1]) <= HANDOFF_EY and
            abs(pos_err_grip[2]) <= HANDOFF_EZ and
            v_rel <= HANDOFF_REL_SPEED and
            cavity_align >= 0.70
        )
