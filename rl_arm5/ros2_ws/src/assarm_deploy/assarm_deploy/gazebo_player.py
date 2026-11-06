#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASSARM Gazebo Policy Player (ROS 2 Humble / Gazebo Fortress)
-------------------------------------------------------------
Nạp mô hình SAC đã huấn luyện và điều khiển robot ASSARM trong Gazebo GUI:
1. Tự động spawn khối cube ngẫu nhiên (Level 2/3 toàn dải) sau mỗi episode
2. Đưa cánh tay quay về vị trí HOME ([0,0,0,0]) mượt mà và mở kẹp
3. Chạy mạng SAC (10 Hz) tiếp cận khối lập phương theo thời gian thực
4. Kích hoạt bàn giao (Handoff) và đóng ngón kẹp gắp khối
5. Giữ khối (Hold) để người dùng quan sát trên màn hình GUI
6. Tự động lặp lại quy trình cho episode tiếp theo
"""

import subprocess
import os
import sys
import time
import argparse
import threading
import queue
import numpy as np

# Đảm bảo import được các module trong ros2_ws
ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for pkg in ["assarm_common", "assarm_kinematics", "assarm_rl_env", "assarm_rl_train"]:
    p = os.path.join(ws_root, "src", pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from stable_baselines3 import SAC

from assarm_common.config import (
    Q_LO_ARM, Q_HI_ARM, DT_CTRL, MAX_DELTA, EPS_ARM,
    HANDOFF_CONSEC_STEPS, HANDOFF_EX, HANDOFF_EY, HANDOFF_EZ,
    HANDOFF_DPSI, HANDOFF_TILT, HANDOFF_REL_SPEED,
    SPAWN_LEVELS
)
from assarm_common.fk import fk_arm, p_close, gripper_yaw, gripper_tilt, wrap_to_pm45
from assarm_common.obs_builder import build_obs
from assarm_common.perception_filter import PerceptionFilter
from assarm_common.grasp_script import GraspScript, GraspState


class AssarmGazeboPlayer(Node):
    def __init__(
        self,
        model_path: str,
        cube_pos: np.ndarray,
        num_episodes: int = 10,
        loop: bool = True,
        random_spawn: bool = True,
        spawn_level: int = 3,
        max_steps: int = 80,
        side: str = 'front',
        diverse_spawn: bool = True,
        dynamic_motion: bool = False
    ):
        super().__init__('assarm_gazebo_player')

        self.cube_pos = np.array(cube_pos, dtype=np.float64)
        self.cube_base_pos = self.cube_pos.copy()
        self.cube_vel = np.zeros(3)
        self.cube_yaw = 0.0
        self.num_episodes = num_episodes
        self.loop = loop
        self.random_spawn = random_spawn
        self.spawn_level = spawn_level
        self.max_steps = max_steps
        self.side = side.lower()
        self.diverse_spawn = diverse_spawn
        self.dynamic_motion = dynamic_motion
        self._prev_spawn_pos = None

        # Tham số quỹ đạo động Stage B+ (1.2 - 2.0 cm/s hình dấu cộng +)
        self.v_target = 0.016
        self.arm_len = 0.035
        self.omega_cross = 2.0 * self.v_target / self.arm_len
        self.t_stroke = 2.0 * np.pi / self.omega_cross
        self.t_cycle = 4.0 * self.t_stroke
        self.start_stroke = 0
        self.wind_noise_state = np.zeros(2)

        # Danh sách các sector chiến lược luân chuyển trong không gian phía trước (r=0.28-0.31m, z=0.38-0.42m)
        self._sectors_front = [
            {"name": "TRÁI - XA (+10%)", "theta": -np.radians(20), "r": 0.31, "z": 0.40},
            {"name": "GIỮA - XA (+10%)", "theta":  0.0,            "r": 0.31, "z": 0.40},
            {"name": "PHẢI - XA (+10%)", "theta": +np.radians(20), "r": 0.30, "z": 0.41},
            {"name": "TRÁI - RỘNG",      "theta": -np.radians(35), "r": 0.28, "z": 0.39},
            {"name": "PHẢI - RỘNG",      "theta": +np.radians(35), "r": 0.28, "z": 0.39},
            {"name": "GIỮA - CAO",       "theta":  0.0,            "r": 0.31, "z": 0.42},
        ]
        self._sectors_back = [
            {"name": "GIỮA - XA",    "theta":  0.0,            "r": 0.28, "z": 0.40},
            {"name": "TRÁI - XA",    "theta": -np.radians(20), "r": 0.27, "z": 0.41},
            {"name": "PHẢI - XA",    "theta": +np.radians(20), "r": 0.27, "z": 0.41},
            {"name": "TRÁI - RỘNG",  "theta": -np.radians(35), "r": 0.26, "z": 0.39},
            {"name": "PHẢI - RỘNG",  "theta": +np.radians(35), "r": 0.26, "z": 0.39},
            {"name": "GIỮA - CAO",   "theta":  0.0,            "r": 0.29, "z": 0.42},
        ]

        # Vị trí Home chuẩn của cánh tay ASSARM: thẳng đứng hướng xuống [0, 0, 0, 0]
        self.q_home = np.array([0.0, 0.0, 0.0, 0.0])
        self._reset_q_start = self.q_home.copy()

        # Nạp mô hình SAC
        self.get_logger().info(f"Đang nạp mô hình RL từ: {model_path}...")
        self.model = SAC.load(model_path)
        self.get_logger().info("Mô hình RL nạp thành công!")

        # Publishers
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_position_controller/commands',
            10
        )
        self.gripper_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_position_controller/commands',
            10
        )

        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # State tracking
        self.joint_names = [
            'Revolute_Joint_1',
            'Revolute_Joint_2',
            'Revolute_Joint_3',
            'Revolute_Joint_4'
        ]
        self.q_meas = self.q_home.copy()
        self.q_prev = self.q_home.copy()
        self.target_q = self.q_home.copy()
        self.has_joint_state = False

        self.perception_filter = PerceptionFilter(dt=DT_CTRL)
        self.grasp_script = GraspScript(dt=DT_CTRL)

        # Trạng thái FSM của Player: 'RESETTING', 'RUNNING_RL', 'GRASPING', 'HOLDING'
        self.fsm_state = 'RESETTING'
        self.state_timer = 0
        self.current_episode = 1
        self.step_in_ep = 0
        self.consec_handoff = 0

        # Tự động nhận diện thế giới Gazebo (empty hoặc pendulum_grasp_world) và chuẩn bị model
        self._detect_and_setup_world()

        # Luồng ngầm gửi lệnh cập nhật Gazebo (không block chu kỳ 10 Hz)
        self._pose_queue = queue.Queue(maxsize=1)
        self._pose_thread = threading.Thread(target=self._pose_worker, daemon=True)
        self._pose_thread.start()

        # Khởi tạo vị trí spawn ban đầu cho Episode 1
        if self.random_spawn:
            self._sample_next_cube_pose()
        else:
            self._update_gazebo_entity_pose()

        # Timer điều khiển 10 Hz
        self.timer = self.create_timer(DT_CTRL, self._control_loop)
        self.get_logger().info("Assarm Gazebo Player đã khởi động ở tần số 10 Hz.")

    def _sample_next_cube_pose(self):
        """
        Sinh vị trí mới cho khối cube:
        - Vùng phía trước arm (bên trái trong hình Gazebo): Y > 0 (side == 'front').
        - Thay đổi rõ rệt qua từng episode bằng cách luân chuyển giữa các sector (Trái/Giữa/Phải, Gần/Xa, Thấp/Cao).
        - Bảo đảm bước nhảy rõ rệt giữa các episode liên tiếp.
        """
        params = SPAWN_LEVELS.get(self.spawn_level, SPAWN_LEVELS[3])
        is_front = (self.side == 'front')

        if self.diverse_spawn:
            sectors = self._sectors_front if is_front else self._sectors_back
            sector = sectors[(self.current_episode - 1) % len(sectors)]
            sector_name = sector["name"]

            # Thêm nhiễu ngẫu nhiên nhỏ xung quanh sector
            d_theta = float(np.random.uniform(-np.radians(5), np.radians(5)))
            d_r = float(np.random.uniform(-0.01, 0.01))
            d_z = float(np.random.uniform(-0.01, 0.01))

            theta = float(np.clip(sector["theta"] + d_theta, -params["theta"], params["theta"]))
            r = float(np.clip(sector["r"] + d_r, params["r"][0], params["r"][1]))
            z = float(np.clip(sector["z"] + d_z, params["z"][0], params["z"][1]))
        else:
            sector_name = "NGẪU NHIÊN TOÀN DẢI"
            r = float(np.random.uniform(params["r"][0], params["r"][1]))
            theta = float(np.random.uniform(-params["theta"], params["theta"]))
            z = float(np.random.uniform(params["z"][0], params["z"][1]))

        # Tọa độ 3D:
        # Front (Phía trước arm, bên trái trong hình Gazebo): X = r * sin(theta), Y = +r * cos(theta), Z = z
        # Back  (Phía sau arm, bên phải trong hình Gazebo):   X = -r * sin(theta), Y = -r * cos(theta), Z = z
        if is_front:
            new_pos = np.array([r * np.sin(theta), r * np.cos(theta), z])
            side_desc = "Phía trước arm (bên trái trong hình Gazebo, Y > 0)"
        else:
            new_pos = np.array([-r * np.sin(theta), -r * np.cos(theta), z])
            side_desc = "Phía sau arm (bên phải trong hình Gazebo, Y < 0)"

        jump_cm = 0.0
        if self._prev_spawn_pos is not None:
            jump_cm = float(np.linalg.norm(new_pos - self._prev_spawn_pos) * 100.0)
        self._prev_spawn_pos = new_pos.copy()

        self.cube_pos = new_pos
        self.cube_yaw = float(np.random.uniform(-params["yaw"], params["yaw"]))

        jump_info = f" | Bước nhảy cự ly: {jump_cm:.1f} cm (Thay đổi rõ rệt)" if jump_cm > 0 else ""
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"[TỰ ĐỘNG SPAWN CUBE EPISODE {self.current_episode}] - SECTOR: [{sector_name}]\n"
            f" - Hướng vùng: {side_desc}\n"
            f" - Bán kính r: {r:.3f} m | Góc quét J1 (theta): {np.degrees(theta):+.1f}° | Độ cao z: {z:.3f} m\n"
            f" - Tọa độ 3D: [X: {self.cube_pos[0]:+.3f}, Y: {self.cube_pos[1]:+.3f}, Z: {self.cube_pos[2]:.3f}]{jump_info}\n"
            f" - Góc Yaw khối: {np.degrees(self.cube_yaw):+.1f}°\n"
            f"{'='*70}"
        )

    def _detect_and_setup_world(self):
        """Tự động kiểm tra thế giới Gazebo đang mở và thiết lập đồng bộ model."""
        world_name = "pendulum_grasp_world"
        model_name = "cube_pendulum"
        z_offset = 0.175
        try:
            res = subprocess.run(['ign', 'service', '-l'], capture_output=True, text=True, timeout=1.5)
            output = res.stdout
            if '/world/empty/' in output:
                world_name = "empty"
                model_name = "target_cylinder"
                z_offset = 0.0
            elif '/world/pendulum_grasp_world/' in output:
                world_name = "pendulum_grasp_world"
                model_name = "cube_pendulum"
                z_offset = 0.175
        except Exception:
            pass

        self.world_name = world_name
        self.model_name = model_name
        self.z_offset = z_offset
        self.get_logger().info(f"Đã kết nối Gazebo World: [{self.world_name}] | Model: [{self.model_name}] (z_offset={self.z_offset})")

        # Nếu là world empty, tự động spawn khối trụ nếu chưa có
        if self.world_name == "empty":
            sdf_str = f"""
            <sdf version="1.6">
              <model name="{self.model_name}">
                <static>true</static>
                <pose>{self.cube_pos[0]} {self.cube_pos[1]} {self.cube_pos[2]} 0 0 0</pose>
                <link name="link">
                  <visual name="visual_cylinder">
                    <geometry>
                      <cylinder><radius>0.025</radius><length>0.10</length></cylinder>
                    </geometry>
                    <material>
                      <ambient>0.0 0.8 1.0 1.0</ambient>
                      <diffuse>0.0 0.8 1.0 1.0</diffuse>
                    </material>
                  </visual>
                  <collision name="collision_cylinder">
                    <geometry>
                      <cylinder><radius>0.025</radius><length>0.10</length></cylinder>
                    </geometry>
                  </collision>
                </link>
              </model>
            </sdf>
            """
            escaped_sdf = sdf_str.replace('\n', ' ').replace('"', '\\"')
            req = f'sdf: "{escaped_sdf}"'
            subprocess.run(
                ['ign', 'service', '-s', f'/world/{self.world_name}/create',
                 '--reqtype', 'ignition.msgs.EntityFactory',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--req', req, '--timeout', '1000'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _joint_state_callback(self, msg: JointState):
        name_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        q_curr = []
        for jn in self.joint_names:
            if jn in name_map:
                q_curr.append(name_map[jn])
        if len(q_curr) == 4:
            self.q_meas = np.array(q_curr)
            self.has_joint_state = True

    def _publish_arm(self, q: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self.arm_cmd_pub.publish(msg)

    def _publish_gripper(self, val: float):
        msg = Float64MultiArray()
        msg.data = [float(val)]
        self.gripper_cmd_pub.publish(msg)

    def _pose_worker(self):
        """Worker thread ngầm thực hiện lệnh ign service để không block control loop chính."""
        while True:
            try:
                pos, yaw = self._pose_queue.get()
                z_model = float(pos[2] + self.z_offset)
                qz = float(np.sin(yaw / 2.0))
                qw = float(np.cos(yaw / 2.0))
                req = f'name: "{self.model_name}", position: {{x: {pos[0]:.4f}, y: {pos[1]:.4f}, z: {z_model:.4f}}}, orientation: {{z: {qz:.4f}, w: {qw:.4f}}}'
                subprocess.run(
                    ['ign', 'service', '-s', f'/world/{self.world_name}/set_pose',
                     '--reqtype', 'ignition.msgs.Pose',
                     '--reptype', 'ignition.msgs.Boolean',
                     '--req', req, '--timeout', '500'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

    def _update_gazebo_entity_pose(self):
        """Đưa vị trí mới vào hàng đợi để thread ngầm gửi sang Gazebo (không block nhịp 10 Hz)."""
        try:
            if not self._pose_queue.empty():
                try:
                    self._pose_queue.get_nowait()
                except queue.Empty:
                    pass
            self._pose_queue.put_nowait((self.cube_pos.copy(), float(self.cube_yaw)))
        except Exception:
            pass

    def _control_loop(self):
        if not self.has_joint_state:
            self.get_logger().info("Đang chờ topic /joint_states...", throttle_duration_sec=2.0)
            return

        if self.fsm_state == 'RESETTING':
            self.state_timer += 1

            if self.state_timer == 1:
                # Ghi nhận tư thế bắt đầu quay về Home
                self._reset_q_start = self.q_meas.copy()

                # Tự động spawn khối ở vị trí mới cho episode này
                if self.random_spawn:
                    self._sample_next_cube_pose()
                self.cube_base_pos = self.cube_pos.copy()
                if self.dynamic_motion:
                    self.v_target = float(np.random.uniform(0.012, 0.020))
                    self.arm_len = 0.035
                    self.omega_cross = 2.0 * self.v_target / self.arm_len
                    self.t_stroke = 2.0 * np.pi / self.omega_cross
                    self.t_cycle = 4.0 * self.t_stroke
                    self.start_stroke = int(np.random.randint(0, 4))
                    self.wind_noise_state = np.zeros(2)
                self._update_gazebo_entity_pose()

            # 1. Mở kẹp hoàn toàn
            self._publish_gripper(0.0)

            # 2. Điều khiển đưa cánh tay quay về vị trí HOME mượt mà (q_home = [0, 0, 0, 0])
            alpha = min(1.0, self.state_timer / 15.0)  # Nội suy êm trong 1.5 giây
            q_target_home = (1.0 - alpha) * self._reset_q_start + alpha * self.q_home
            self._publish_arm(q_target_home)

            if self.state_timer >= 25:  # 2.5 giây để hoàn toàn ổn định tại HOME
                dyn_info = f" [DYNAMIC {self.v_target*100:.1f} cm/s]" if self.dynamic_motion else " [STATIC]"
                self.get_logger().info(f"\n{'='*60}\n>>> BẮT ĐẦU EPISODE {self.current_episode} TỪ VỊ TRÍ HOME{dyn_info} <<<\n{'='*60}")
                self.fsm_state = 'RUNNING_RL'
                self.step_in_ep = 0
                self.consec_handoff = 0
                self.q_prev = self.q_home.copy()
                self.target_q = self.q_home.copy()
                self.perception_filter.reset(self.cube_pos)
                self.grasp_script.reset(self.q_meas)

        elif self.fsm_state == 'RUNNING_RL':
            self.step_in_ep += 1

            # 0. Cập nhật vị trí vật thể di động thời gian thực (nếu bật dynamic_motion)
            if self.dynamic_motion:
                t = self.step_in_ep * DT_CTRL
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

                # Nhiễu gió OU drift ngẫu nhiên
                noise_step = np.random.normal(0.0, 0.0005, size=2)
                self.wind_noise_state = 0.95 * self.wind_noise_state + noise_step
                self.wind_noise_state = np.clip(self.wind_noise_state, -0.005, 0.005)
                cur_pos[0] += self.wind_noise_state[0]
                cur_pos[1] += self.wind_noise_state[1]

                r_cur = float(np.hypot(cur_pos[0], cur_pos[1]))
                theta_cur = float(np.arctan2(cur_pos[0], cur_pos[1]))
                r_clamped = float(np.clip(r_cur, 0.25, 0.33))
                theta_clamped = float(np.clip(theta_cur, -np.radians(35), np.radians(35)))
                self.cube_pos[0] = r_clamped * np.sin(theta_clamped)
                self.cube_pos[1] = r_clamped * np.cos(theta_clamped)
                self.cube_pos[2] = cur_pos[2]
                self.cube_vel = np.array([vx, vy, 0.0])
                self._update_gazebo_entity_pose()

            # 1. Cập nhật bộ lọc tri giác PerceptionFilter đồng bộ chuẩn xác với môi trường RL
            self.perception_filter.update(self.cube_pos, self.cube_yaw)
            obj_pos_filtered = self.perception_filter.filter.position
            obj_vel_filtered = self.perception_filter.filter.velocity if self.dynamic_motion else np.zeros(3)

            # 2. Đo lường hình học thực tế
            p_cur = p_close(self.q_meas)
            pos_err_world = self.cube_pos - p_cur
            _, R_grip = fk_arm(self.q_meas)
            pos_err_grip = R_grip.T @ pos_err_world

            yaw_jaw = gripper_yaw(self.q_meas)
            dpsi = wrap_to_pm45(self.cube_yaw - yaw_jaw)
            tilt = gripper_tilt(self.q_meas)
            dist_mm = np.linalg.norm(pos_err_world) * 1000.0

            # 3. Xây dựng observation 28D chuẩn hóa giống hệt assarm_env.py
            obs = build_obs(
                q_arm=self.q_meas,
                q_prev=self.q_prev,
                target=self.target_q,
                p_close=p_cur,
                obj_pos=obj_pos_filtered,
                obj_vel_filtered=obj_vel_filtered,
                R_gripper=R_grip,
                yaw_obj=self.cube_yaw,
                yaw_jaw=yaw_jaw,
                tilt=tilt,
                pose_valid=True,
                dt=DT_CTRL,
            )

            # 4. Model SAC dự đoán action
            action, _ = self.model.predict(obs, deterministic=True)

            # [GIẢI PHÁP TỐI ƯU CƠ HỌC] Khối trụ tròn đối xứng 360° quanh trục đứng Z:
            # Khóa cứng action của khớp J4 = 0.0 để hai ngón kẹp LUÔN NẰM NGANG 100%
            # (Ngón trái - ngón phải ôm trọn chu vi thân trụ, triệt tiêu hoàn toàn lỗi xoay dọc 90°)
            action[3] = 0.0

            # 5. Cập nhật góc đích — Sim-to-Sim Compensated Action Scaling
            p_prev = p_close(self.q_meas)
            d_prev = float(np.linalg.norm(self.cube_pos - p_prev))
            scale = float(np.clip(d_prev / 0.03, 0.7, 1.0))
            delta_q = action * MAX_DELTA * scale
            self.target_q = np.clip(self.target_q + delta_q, Q_LO_ARM, Q_HI_ARM)
            self.target_q[3] = 0.0  # Luôn giữ mục tiêu J4 = 0.0 rad

            if d_prev >= 0.08:  # Xa hơn 80mm: giữ EPS_ARM bảo vệ an toàn
                self.target_q = np.clip(self.target_q, self.q_meas - EPS_ARM, self.q_meas + EPS_ARM)
                self.target_q[3] = 0.0

            self._publish_arm(self.target_q)
            self.q_prev = self.q_meas.copy()

            # 6. Kiểm tra điều kiện bàn giao NGHIÊM NGẶT THEO HỆ CƠ HỌC NGÀM KẸP (Gripper Local Frame)
            #    Khối trụ tròn d=50mm, khe há ngàm 70mm, dài ngón 65mm
            #    - ex: Trục đóng ngón (|ex| <= 15mm): Khối trụ phải nằm chính giữa 2 má kẹp cao su
            #    - ey: Trục sâu ngón (|ey| <= 22mm): Khối trụ phải lọt sâu vào trong lòng ngón kẹp
            #    - ez: Chiều cao ngàm (|ez| <= 25mm): Ngàm kẹp phải ngậm vào thân trụ (cao 100mm)
            #    - cavity_align >= 0.70: Miệng ngàm phải chĩa thẳng vào tâm khối trụ (góc lệch <= 45°)
            #    - |q_meas[3]| <= 12°: Khớp J4 phải nằm ngang chuẩn mực, không xoay vặn
            pos_plate, _ = fk_arm(self.q_meas)
            v_cavity = -R_grip[:, 2]
            vec_to_obj = self.cube_pos - pos_plate
            dist_plate = np.linalg.norm(vec_to_obj)
            cavity_align = np.dot(v_cavity, vec_to_obj / dist_plate) if dist_plate > 1e-4 else 1.0

            ex_mm = abs(pos_err_grip[0]) * 1000.0
            ey_mm = abs(pos_err_grip[1]) * 1000.0
            ez_mm = abs(pos_err_grip[2]) * 1000.0
            j4_deg = float(np.degrees(abs(self.q_meas[3])))

            is_aligned = (
                abs(pos_err_grip[0]) <= 0.015 and   # 15mm: tâm khối lọt khe 2 má
                abs(pos_err_grip[1]) <= 0.022 and   # 22mm: khối lọt sâu vào ngàm
                abs(pos_err_grip[2]) <= 0.025 and   # 25mm: ngang thân khối
                cavity_align >= 0.70 and            # Miệng ngàm chĩa thẳng vào khối
                abs(self.q_meas[3]) <= np.radians(12.0)  # Cổ tay J4 nằm ngang chuẩn xác (<= 12°)
            )

            if is_aligned:
                self.consec_handoff += 1
            else:
                self.consec_handoff = 0

            status_str = (
                f"STEP {self.step_in_ep:3d} | d={dist_mm:5.1f}mm "
                f"| align={cavity_align*100:3.0f}% "
                f"| J4={j4_deg:4.1f}° "
                f"| grip[ex={ex_mm:4.1f} ey={ey_mm:4.1f} ez={ez_mm:4.1f}]mm "
                f"| {'✓' if is_aligned else '✗'} {self.consec_handoff}/{HANDOFF_CONSEC_STEPS}"
            )
            self.get_logger().info(status_str)

            if self.consec_handoff >= HANDOFF_CONSEC_STEPS:
                self.get_logger().info(
                    f"\n{'='*60}\n"
                    f"[CHUẨN XÁC] >>> ĐẠT CHUẨN CƠ HỌC BÀN GIAO TẠI BƯỚC {self.step_in_ep}!\n"
                    f"Sai số ngàm: ex={ex_mm:.1f}mm, ey={ey_mm:.1f}mm, ez={ez_mm:.1f}mm | J4={j4_deg:.1f}° (Nằm ngang chuẩn)\n"
                    f"KÍCH HOẠT ĐÓNG KẸP ÔM KHÍCH THÂN KHỐI TRỤ!\n"
                    f"{'='*60}"
                )
                self.fsm_state = 'GRASPING'
                self.state_timer = 0
                self.grasp_script.trigger_handoff(self.q_meas)

            elif self.step_in_ep >= self.max_steps:
                self.get_logger().warn(
                    f"[HẾT GIỜ] {self.max_steps} bước ({self.max_steps*0.1:.1f}s). "
                    f"d_cuối={dist_mm:.1f}mm, align={cavity_align*100:.0f}%, "
                    f"grip[ex={ex_mm:.1f} ey={ey_mm:.1f} ez={ez_mm:.1f}]mm"
                )
                self._finish_episode()

        elif self.fsm_state == 'GRASPING':
            self.state_timer += 1
            # Đóng ngón kẹp góc 0.896 rad (ép chặt thân trụ)
            self._publish_gripper(0.896)
            # Giữ nguyên cánh tay ở tư thế hiện tại
            self._publish_arm(self.q_meas)
            # LƯU Ý QUAN TRỌNG: KHÔNG gọi _update_gazebo_entity_pose() ở đây
            # để Gazebo Physics hoàn toàn xử lý lực tiếp xúc và ma sát ngón kẹp!

            if self.state_timer >= 15:  # Đã kẹp chặt (1.5 giây)
                self.get_logger().info(">>> ĐÃ GẮP CHẶT THÂN KHỐI TRỤ! Đang giữ để quan sát tương tác vật lý... <<<")
                self.fsm_state = 'HOLDING'
                self.state_timer = 0

        elif self.fsm_state == 'HOLDING':
            self.state_timer += 1
            self._publish_gripper(0.896)
            self._publish_arm(self.q_meas)
            # KHÔNG ép vị trí khối trụ, để khối trụ được kẹp tự nhiên cùng cánh tay
            if self.state_timer >= 25:  # Giữ 2.5 giây
                self.get_logger().info(f">>> HOÀN THÀNH EPISODE {self.current_episode} <<<\n")
                self._finish_episode()

    def _finish_episode(self):
        self.current_episode += 1
        if not self.loop and self.current_episode > self.num_episodes:
            self.get_logger().info("Đã hoàn thành toàn bộ episode chỉ định. Tắt Node.")
            rclpy.shutdown()
        else:
            self.get_logger().info("--> Đang mở kẹp và đưa cánh tay quay về vị trí HOME...")
            self.fsm_state = 'RESETTING'
            self.state_timer = 0


def resolve_model_path(model_arg: str, ws_root: str) -> str:
    """Tự động phân giải đường dẫn mô hình, fallback thông minh sang checkpoint tốt nhất khả dụng."""
    candidates = [
        model_arg,
        os.path.join(ws_root, model_arg),
        os.path.abspath(model_arg),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)

    # Tự động tìm fallback trong thư mục models/
    models_dir = os.path.join(ws_root, "models")
    fallbacks = [
        os.path.join(models_dir, "stage_b_plus_final.zip"),
        os.path.join(models_dir, "stage_b_plus_best.zip"),
        os.path.join(models_dir, "stage_b_plus", "best_model.zip"),
        os.path.join(models_dir, "stage_b_best.zip"),
        os.path.join(models_dir, "stage_b_final.zip"),
        os.path.join(models_dir, "stage_b_interrupted.zip"),
        os.path.join(models_dir, "best_model.zip"),
        os.path.join(models_dir, "checkpoint_interrupted.zip"),
        os.path.join(models_dir, "latest_model.zip"),
    ]
    for fb in fallbacks:
        if os.path.exists(fb):
            print(f"\n[THÔNG BÁO] Chưa tìm thấy file '{model_arg}'.")
            print(f"--> Tự động sử dụng checkpoint khả dụng tốt nhất: {fb}\n")
            return fb

    raise FileNotFoundError(f"Không tìm thấy file mô hình tại '{model_arg}' và không có checkpoint khả dụng trong {models_dir}")


def main():
    parser = argparse.ArgumentParser(description="Chạy mô phỏng mô hình RL gắp khối trong Gazebo GUI")
    parser.add_argument("--model", type=str, default="models/stage_b_best.zip", help="Đường dẫn tới file .zip mô hình SAC (tự động fallback nếu chưa có)")
    parser.add_argument("--side", type=str, default="front", choices=["front", "back"], help="Vùng spawn: 'front' (phía trước arm, bên trái trong hình Gazebo, Y > 0) hoặc 'back' (phía sau arm, Y < 0)")
    parser.add_argument("--cube_x", type=float, default=0.0, help="Tọa độ X của khối ban đầu trong Gazebo (m, mặc định 0.0m)")
    parser.add_argument("--cube_y", type=float, default=0.28, help="Tọa độ Y của khối ban đầu trong Gazebo (m, mặc định +0.28m tâm làm việc)")
    parser.add_argument("--cube_z", type=float, default=0.40, help="Tọa độ Z của khối ban đầu trong Gazebo (m, mặc định +0.40m)")
    parser.add_argument("--fixed_spawn", action="store_true", help="Cố định vị trí khối (tắt tự động random spawn)")
    parser.add_argument("--no_diverse", action="store_true", help="Tắt luân chuyển sector đa dạng giữa các episode (dùng random thuần túy)")
    parser.add_argument("--level", type=int, default=3, help="Cấp độ spawn (0..3), mặc định Level 3 toàn dải")
    parser.add_argument("--episodes", type=int, default=20, help="Số episode chạy")
    parser.add_argument("--max_steps", type=int, default=120, help="Số bước tối đa mỗi episode (mặc định 120 bước = 12.0s)")
    parser.add_argument("--once", action="store_true", help="Chỉ chạy đúng số episode chỉ định rồi thoát")
    parser.add_argument("--dynamic", action="store_true", help="Kích hoạt chuyển động ngẫu nhiên động Stage B+ (1.2 - 2.0 cm/s)")
    args, unknown = parser.parse_known_args()

    model_abs = resolve_model_path(args.model, ws_root)

    rclpy.init()
    cube_pos = np.array([args.cube_x, args.cube_y, args.cube_z])
    player = AssarmGazeboPlayer(
        model_path=model_abs,
        cube_pos=cube_pos,
        num_episodes=args.episodes,
        loop=not args.once,
        random_spawn=not args.fixed_spawn,
        spawn_level=args.level,
        max_steps=args.max_steps,
        side=args.side,
        diverse_spawn=not args.no_diverse,
        dynamic_motion=args.dynamic
    )

    try:
        rclpy.spin(player)
    except KeyboardInterrupt:
        print("\nĐã dừng mô phỏng.")
    finally:
        player.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
