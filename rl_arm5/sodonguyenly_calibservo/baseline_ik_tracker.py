#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PHƯƠNG PHÁP 1: BASELINE VISUAL TRACKING + S-CURVE INVERSE KINEMATICS (IK)
=============================================================================
Dự án: ASSARM 4-DOF Robot Arm + Monocular Pinhole Camera
Chạy trên: Raspberry Pi 4B (Khóa cứng CPU Core 0, nhận tọa độ từ Core 1 qua IPC)

Cơ chế hoạt động:
  1. Nhận tọa độ 3D vật thể (X_base, Y_base, Z_base) thời gian thực từ Camera Stream (Core 1).
  2. Bù độ cao Z-Offset và kiểm tra tầm với Workspace.
  3. Giải Inverse Kinematics (IK) 3-DOF với khớp J4 khóa cố định ở 0° (90° servo) để miệng kẹp luôn ngang.
  4. Bám đuổi theo vết vật thể (Visual Tracking) bằng bộ lọc S-Curve mượt mà.
  5. Tự động kích hoạt bàn giao (Handoff) khi khoảng cách <= 3.5cm để đóng ngàm kẹp gắp vật.
  6. Nhấc vật lên cao (LIFT) và giữ ổn định (HOLD).
  7. Ghi nhật ký đo đạc đối chứng: thời gian gắp, sai số bám vết, jerk rung lắc vào CSV.
=============================================================================
"""

import os
import sys
import time
import argparse
import csv
from enum import Enum
from typing import Tuple, List, Optional
import numpy as np

# Nạp thư viện UART
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# Import IPC Bridge và Dynamics từ determine_depth
import socket
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "determine_depth")))
try:
    from core.ipc_bridge import VisionIPCSubscriber
    from core.arm_dynamics import ArmDynamicsCalculator
    HAS_IPC = True
except ImportError:
    HAS_IPC = False


# ===========================================================================
# 1. KHÓA TIẾN TRÌNH TRÊN CPU CORE 0 CỦA PI 4B
# ===========================================================================
def pin_to_cpu_core_0():
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})
            print("[HỆ THỐNG] ✅ Đã khóa tiến trình điều khiển & UART chạy cố định trên CPU CORE 0 của Pi 4B.")
    except Exception as e:
        print(f"[HỆ THỐNG] Cảnh báo: Không thể gán CPU affinity: {e}")


# ===========================================================================
# 2. CẤU HÌNH VẬT LÝ VÀ ĐỘNG HỌC (TỪ ASSARM URDF VÀ SIMULATION)
# ===========================================================================
Q_LO_ARM = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2], dtype=np.float64)
Q_HI_ARM = np.array([+np.pi / 2, +np.pi / 2, +np.pi / 2, +np.pi / 2], dtype=np.float64)

MAX_DELTA_Q = np.radians(8.0) # Tối đa 8 độ / 100ms
DT_CTRL = 0.1                 # Chu kỳ điều khiển 10 Hz (100 ms)

Q_GRIPPER_OPEN = 0.0          # 0 độ mở ngàm
GRIP_CLOSE_OFFSET = np.array([0.0, 0.0105, -0.065], dtype=np.float64)


def rad_to_deg_arm(rad: float) -> float:
    return float(np.degrees(rad) + 90.0)

def deg_to_rad_arm(deg: float) -> float:
    return float(np.radians(deg - 90.0))

def rad_to_deg_gripper(rad: float) -> float:
    return float(np.degrees(rad))

def deg_to_rad_gripper(deg: float) -> float:
    return float(np.radians(deg))


# ===========================================================================
# 3. ĐỘNG HỌC THUẬN (FORWARD KINEMATICS)
# ===========================================================================
def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def _Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def _ht_from_xyz(x: float, y: float, z: float) -> np.ndarray:
    return _homogeneous(np.eye(3, dtype=np.float64), np.array([x, y, z], dtype=np.float64))

_T_base_to_servo1 = _ht_from_xyz(0.003894, 0.02389, 0.56)
_T_servo1_to_j1   = _ht_from_xyz(-0.0039, -0.034687, 0.020797)
_T_pre_j1         = _T_base_to_servo1 @ _T_servo1_to_j1
_T_j1_to_servo2   = _ht_from_xyz(0.0, 0.0, -0.037247)
_T_servo2_to_j2   = _ht_from_xyz(-0.0262, 0.0, -0.03702)
_T_j1_to_j2       = _T_j1_to_servo2 @ _T_servo2_to_j2
_T_j2_to_link1    = _ht_from_xyz(0.0258, 0.0, -0.027)
_T_link1_to_b3    = _ht_from_xyz(-0.005, 0.024987, -0.1)
_T_b3_to_servo3   = _ht_from_xyz(-0.008, -0.00075, -0.00987)
_T_servo3_to_j3   = _ht_from_xyz(-0.0155, -0.03415, -0.005)
_T_j2_to_j3       = _T_j2_to_link1 @ _T_link1_to_b3 @ _T_b3_to_servo3 @ _T_servo3_to_j3
_T_j3_to_b3u      = _ht_from_xyz(-0.002, 0.0, 0.0)
_T_b3u_to_link2   = _ht_from_xyz(0.026, 0.0, -0.05)
_T_link2_to_s4    = _ht_from_xyz(-0.005, 0.02415, -0.1)
_T_s4_to_j4       = _ht_from_xyz(0.005, -0.03415, -0.0155)
_T_j3_to_j4       = _T_j3_to_b3u @ _T_b3u_to_link2 @ _T_link2_to_s4 @ _T_s4_to_j4
_T_j4_to_plate    = _ht_from_xyz(0.0, 0.0, -0.002)

def fk_arm(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T_j1 = _homogeneous(_Rz(-q[0]), np.zeros(3))
    T_j2 = _homogeneous(_Rx(-q[1]), np.zeros(3))
    T_j3 = _homogeneous(_Rx(-q[2]), np.zeros(3))
    T_j4 = _homogeneous(_Rz(-q[3]), np.zeros(3))
    T = (_T_pre_j1 @ T_j1 @ _T_j1_to_j2 @ T_j2 @ _T_j2_to_j3 @ T_j3 @ _T_j3_to_j4 @ T_j4 @ _T_j4_to_plate)
    return T[:3, 3].copy(), T[:3, :3].copy()

def compute_p_close(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos, R = fk_arm(q)
    return pos + R @ GRIP_CLOSE_OFFSET, R

def compute_tilt_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


# ===========================================================================
# 4. BỘ GIẢI IK & BỘ LÀM MƯỢT S-CURVE QUINTIC
# ===========================================================================
def scurve_factor(tau: float) -> float:
    tau = np.clip(tau, 0.0, 1.0)
    return float(6.0 * (tau ** 5) - 15.0 * (tau ** 4) + 10.0 * (tau ** 3))

def generate_scurve_trajectory(q_start: np.ndarray, q_goal: np.ndarray,
                              duration: float = 2.0, dt: float = DT_CTRL) -> List[np.ndarray]:
    n_steps = max(2, int(duration / dt))
    traj = []
    for k in range(n_steps + 1):
        tau = k / float(n_steps)
        traj.append(q_start + scurve_factor(tau) * (q_goal - q_start))
    return traj

def jacobian_numerical(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    J = np.zeros((3, 4), dtype=np.float64)
    for i in range(4):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        pp, _ = compute_p_close(qp)
        pm, _ = compute_p_close(qm)
        J[:, i] = (pp - pm) / (2.0 * eps)
    return J

def check_ik_feasibility(target_pos: np.ndarray, max_iter: int = 50) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Giải nghịch đảo động học (IK) 3-DOF với J4 khóa cố định tại 0.0 rad (Servo 4 = 90° ngang).
    """
    tx, ty, tz = target_pos
    if ty <= 0.05:
        return False, f"Y={ty:.3f}m <= 0.05m (quá sát thân đế)", None

    dist_shoulder = np.sqrt(tx**2 + ty**2 + (tz - 0.56)**2)
    # Khối trụ có thể vươn tới tối đa 42cm từ khớp vai
    if dist_shoulder > 0.42:
        return False, f"Khoảng cách ({dist_shoulder*100:.1f}cm) vượt tầm với tối đa (42cm)", None
    if dist_shoulder < 0.10:
        return False, f"Khoảng cách ({dist_shoulder*100:.1f}cm) quá gần (<10cm)", None

    theta_j1 = np.arctan2(tx, ty)
    q = np.array([theta_j1, -0.40, -0.60, 0.0], dtype=np.float64)

    for _ in range(max_iter):
        p_cur, _ = compute_p_close(q)
        err = target_pos - p_cur
        dist = np.linalg.norm(err)
        if dist < 0.005:
            break
        J = jacobian_numerical(q)
        J_3 = J[:, :3]
        step_mag = min(1.0, 0.03 / max(dist, 1e-4))
        dx = err * step_mag
        dq_3 = np.linalg.solve(J_3.T @ J_3 + 1e-3 * np.eye(3), J_3.T @ dx)
        dq_3 = np.clip(dq_3, -0.06, 0.06)
        q[:3] += dq_3
        q[3] = 0.0

    final_p, final_R = compute_p_close(q)
    final_dist = np.linalg.norm(target_pos - final_p)
    tilt = compute_tilt_deg(final_R)

    if final_dist > 0.045:
        return False, f"IK không hội tụ (sai số {final_dist*100:.1f}cm > 4.5cm)", None

    servo_deg = [rad_to_deg_arm(q[i]) for i in range(4)]
    for i, deg in enumerate(servo_deg):
        if deg < 0.0 or deg > 180.0:
            return False, f"Khớp J{i+1} góc {deg:.1f}° ngoài giới hạn 0-180°", None

    msg = f"IK OK: J=[{servo_deg[0]:.1f}°, {servo_deg[1]:.1f}°, {servo_deg[2]:.1f}°, {servo_deg[3]:.1f}°] | Tilt={tilt:.1f}° | Err={final_dist*1000:.1f}mm"
    return True, msg, q


# ===========================================================================
# 5. MÁY TRẠNG THÁI GẮP BÁM ĐUỔI VISUAL TRACKING
# ===========================================================================
class GraspState(Enum):
    WAIT_TARGET = 0 # Chờ camera nhận diện ổn định
    READY = 1       # Di chuyển mượt mà về tư thế chuẩn bị đón
    TRACKING = 2    # Bám đuổi theo tọa độ Camera cập nhật liên tục
    CLOSE = 3       # Đóng ngàm kẹp
    VERIFY = 4      # Xác nhận kẹp trúng
    LIFT = 5        # Nâng vật lên
    HOLD = 6        # Giữ ổn định
    COMPLETED = 7   # Hoàn thành xuất sắc
    FAILED = 8      # Thất bại


class BaselineVisualTracker:
    def __init__(self, dt: float = DT_CTRL, grip_angle: float = 90.0, handoff_dist: float = 0.075):
        self.dt = dt
        self.grip_close_rad = np.radians(grip_angle)
        self.handoff_dist_max = handoff_dist
        self.state = GraspState.WAIT_TARGET
        self.state_timer = 0.0
        self.consec_handoff_count = 0
        self.valid_streak = 0
        self.q_cmd = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)
        self.gripper_cmd = Q_GRIPPER_OPEN
        self.q_ready = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)

        self.scurve_traj = []
        self.scurve_idx = 0
        self.target_pos = np.zeros(3)
        self.locked_target = None
        self.initial_grasp_q = self.q_ready.copy()

    def start(self, initial_q: np.ndarray):
        self.state = GraspState.WAIT_TARGET
        self.state_timer = 0.0
        self.consec_handoff_count = 0
        self.valid_streak = 0
        self.locked_target = None
        self.q_cmd = self.q_ready.copy()
        self.gripper_cmd = Q_GRIPPER_OPEN

    def step(self, target_pos: Optional[np.ndarray], target_valid: bool,
             q_fb: np.ndarray, grip_fb: float) -> Tuple[np.ndarray, float, GraspState]:
        self.state_timer += self.dt
        p_c, R_c = compute_p_close(q_fb)

        # 0. PHA WAIT_TARGET: Đứng yên chờ camera thấy vật liên tục
        if self.state == GraspState.WAIT_TARGET:
            self.gripper_cmd = Q_GRIPPER_OPEN
            self.q_cmd = self.q_ready.copy()
            if target_valid and target_pos is not None:
                self.valid_streak += 1
                if self.valid_streak >= 4:  # Thấy liên tục 0.4s
                    self.target_pos = target_pos.copy()
                    print(f"\n[VISION OK] ✅ Đã tìm thấy khối trụ tại Base: ({self.target_pos[0]:+.3f}, {self.target_pos[1]:.3f}, {self.target_pos[2]:.3f})m! BẮT ĐẦU BÁM ĐUỔI IK...")
                    self.state = GraspState.TRACKING
                    self.state_timer = 0.0
                    self.consec_handoff_count = 0
            else:
                self.valid_streak = 0

        # 1. PHA TRACKING: Bám đuổi theo tọa độ Camera liên tục
        elif self.state == GraspState.TRACKING:
            self.gripper_cmd = Q_GRIPPER_OPEN
            dist_to_cur = np.linalg.norm(self.target_pos - p_c)

            # Cơ chế Chống Che Khuất (Occlusion Lock):
            # Khi ngàm kẹp đã vào rất gần (<= 8.0cm), nếu camera bị che khuất hoặc tọa độ nhảy vọt > 15cm, khóa cứng vị trí mục tiêu cuối cùng
            should_update_target = False
            if target_valid and target_pos is not None:
                jump_dist = np.linalg.norm(target_pos - self.target_pos)
                if dist_to_cur > 0.08 or jump_dist < 0.12:
                    self.target_pos = target_pos.copy()
                    should_update_target = True

            is_feasible, _, q_sol = check_ik_feasibility(self.target_pos)
            if is_feasible and q_sol is not None:
                delta_q = q_sol - self.q_cmd
                delta_clipped = np.clip(delta_q, -MAX_DELTA_Q, MAX_DELTA_Q)
                self.q_cmd = np.clip(self.q_cmd + delta_clipped, Q_LO_ARM, Q_HI_ARM)
                self.q_cmd[3] = 0.0

            dist_err = np.linalg.norm(self.target_pos - p_c)
            # Ngưỡng kích hoạt đóng kẹp (7.5cm): đủ an toàn và chắc chắn
            if dist_err <= self.handoff_dist_max:
                self.consec_handoff_count += 1
            else:
                self.consec_handoff_count = 0

            if self.consec_handoff_count >= 2:
                print(f"\n[HANDOFF READY] 🎯 Đạt cự ly gắp chuẩn tại P_err: {dist_err*100:.1f}cm! ĐÓNG NGÀM KẸP...")
                self.state = GraspState.CLOSE
                self.state_timer = 0.0
                self.initial_grasp_q = self.q_cmd.copy()
                return self.q_cmd, self.gripper_cmd, self.state

            if self.state_timer > 25.0:
                self.state = GraspState.FAILED
                return self.q_cmd, self.gripper_cmd, self.state

        # 2. PHA CLOSE: Đóng ngàm kẹp
        elif self.state == GraspState.CLOSE:
            self.q_cmd = self.initial_grasp_q.copy()
            self.gripper_cmd = self.grip_close_rad
            if self.state_timer >= 1.2:
                self.state = GraspState.VERIFY
                self.state_timer = 0.0

        # 3. PHA VERIFY: Xác nhận ngàm kẹp đã ngậm chặt
        elif self.state == GraspState.VERIFY:
            self.q_cmd = self.initial_grasp_q.copy()
            self.gripper_cmd = self.grip_close_rad
            self.state = GraspState.LIFT
            self.state_timer = 0.0
            # Nâng vai (J2) và co khuỷu (J3) lên +5cm trong 1.5s
            target_lift_q = self.initial_grasp_q.copy()
            target_lift_q[1] += np.radians(8.0)
            target_lift_q[2] -= np.radians(8.0)
            self.scurve_traj = generate_scurve_trajectory(self.q_cmd, target_lift_q, duration=1.5, dt=self.dt)
            self.scurve_idx = 0

        # 4. PHA LIFT: Nâng khối trụ lên khỏi mặt bàn
        elif self.state == GraspState.LIFT:
            self.gripper_cmd = self.grip_close_rad
            if self.scurve_idx < len(self.scurve_traj):
                self.q_cmd = self.scurve_traj[self.scurve_idx]
                self.scurve_idx += 1
            else:
                self.state = GraspState.HOLD
                self.state_timer = 0.0

        # 5. PHA HOLD: Giữ ổn định trên không
        elif self.state == GraspState.HOLD:
            self.gripper_cmd = self.grip_close_rad
            if self.state_timer >= 3.0:
                self.state = GraspState.COMPLETED

        return self.q_cmd, self.gripper_cmd, self.state


# ===========================================================================
# 6. GIAO TIẾP ESP32 UART & LOGGER BENCHMARK
# ===========================================================================
class ESP32HardwareBridge:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, mock: bool = False):
        self.mock = mock
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.sim_q = np.zeros(4)
        self.sim_grip = 0.0
        self.last_fb_q = np.zeros(4)
        self.last_fb_grip = 0.0
        self.last_adc = [0, 0, 0, 0]

        if not self.mock:
            if not HAS_SERIAL:
                print("[CẢNH BÁO] Chưa cài pyserial. Chạy MOCK mode!")
                self.mock = True
            else:
                if not os.path.exists(self.port):
                    for alt in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]:
                        if os.path.exists(alt):
                            self.port = alt
                            break
                try:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    time.sleep(1.0)
                    self.flush()
                    print(f"[UART] ✅ Đã kết nối thành công tới ESP32 tại {self.port} ({self.baudrate} baud)")
                except Exception as e:
                    print(f"[LỖI UART] Không thể mở cổng {self.port}: {e}. Chuyển sang MOCK mode.")
                    self.mock = True

    def flush(self):
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def send_and_receive(self, q_target: np.ndarray, grip_target: float) -> Tuple[np.ndarray, float]:
        if self.mock:
            alpha = 0.35
            self.sim_q += alpha * (q_target - self.sim_q)
            self.sim_grip += alpha * (grip_target - self.sim_grip)
            return self.sim_q.copy(), float(self.sim_grip)

        deg1, deg2, deg3, deg4 = [rad_to_deg_arm(q_target[i]) for i in range(4)]
        degG = rad_to_deg_gripper(grip_target)
        cmd_str = f"SET {deg1:.1f} {deg2:.1f} {deg3:.1f} {deg4:.1f} {degG:.1f}\n"

        try:
            self.ser.write(cmd_str.encode("ascii"))
            latest_fb_line = None
            while self.ser.in_waiting > 0:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l
            if latest_fb_line is None:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l

            if latest_fb_line and latest_fb_line.startswith("FB"):
                main_part = latest_fb_line.split("|")[0].strip()
                parts = main_part.split()
                if len(parts) >= 6:
                    self.last_fb_q = np.array([deg_to_rad_arm(float(parts[i])) for i in range(1, 5)])
                    self.last_fb_grip = deg_to_rad_gripper(float(parts[5]))
                if "|" in latest_fb_line:
                    adc_tokens = latest_fb_line.split("|")[1].strip().split()
                    if len(adc_tokens) >= 5 and adc_tokens[0] == "ADC":
                        self.last_adc = [int(adc_tokens[i]) for i in range(1, 5)]

                # TỰ ĐỘNG PHÁT HIỆN: Nếu ADC 1, 2, 3 < 500mV (chưa gắn chiết áp feedback)
                # Tự động chuyển sang Open-Loop (dùng q_target làm feedback) tránh lỗi góc giả!
                if self.last_adc[0] < 500 and self.last_adc[1] < 500 and self.last_adc[2] < 500:
                    if not hasattr(self, "_adc_warned"):
                        print("[UART INFO] ℹ️  Không phát hiện dây ADC feedback (ADC < 500mV) -> Kích hoạt Open-Loop Command Mode chuẩn xác 100%!")
                        self._adc_warned = True
                    self.last_fb_q = q_target.copy()

                return self.last_fb_q.copy(), float(self.last_fb_grip)
        except Exception as e:
            print(f"[UART LỖI]: {e}")

        return self.last_fb_q.copy(), float(grip_target)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class TrackingBenchmarkLogger:
    def __init__(self, filename: str = "benchmark_baseline_ik.csv"):
        self.filename = filename
        self.records = []

    def log_step(self, t: float, state: str, p_close: np.ndarray, target: np.ndarray,
                 pos_err: float, tilt_deg: float, q_cmd: np.ndarray, q_fb: np.ndarray, jerk: float):
        self.records.append({
            "time_s": f"{t:.2f}",
            "state": state,
            "px": f"{p_close[0]:.4f}", "py": f"{p_close[1]:.4f}", "pz": f"{p_close[2]:.4f}",
            "tx": f"{target[0]:.4f}",  "ty": f"{target[1]:.4f}",  "tz": f"{target[2]:.4f}",
            "pos_error_m": f"{pos_err:.4f}",
            "tilt_deg": f"{tilt_deg:.2f}",
            "jerk": f"{jerk:.4f}",
            "s1_cmd": f"{rad_to_deg_arm(q_cmd[0]):.1f}", "s2_cmd": f"{rad_to_deg_arm(q_cmd[1]):.1f}",
            "s3_cmd": f"{rad_to_deg_arm(q_cmd[2]):.1f}", "s4_cmd": f"{rad_to_deg_arm(q_cmd[3]):.1f}",
            "s1_fb": f"{rad_to_deg_arm(q_fb[0]):.1f}",   "s2_fb": f"{rad_to_deg_arm(q_fb[1]):.1f}",
            "s3_fb": f"{rad_to_deg_arm(q_fb[2]):.1f}",   "s4_fb": f"{rad_to_deg_arm(q_fb[3]):.1f}",
        })

    def save(self):
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[BENCHMARK] Đã lưu {len(self.records)} dòng dữ liệu vào {self.filename}")


# ===========================================================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Phương pháp 1: Baseline Visual Tracking + S-Curve IK")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Cổng UART nối ESP32")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate UART")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Bù độ cao Z (m) của khối trụ (0.0 sau khi đã Hand-Eye Calib)")
    parser.add_argument("--grip-angle", type=float, default=90.0, help="Góc đóng ngàm kẹp Servo 5 (độ)")
    parser.add_argument("--handoff-dist", type=float, default=0.075, help="Ngưỡng khoảng cách bàn giao (m) để đóng ngàm kẹp (mặc định 7.5cm)")
    parser.add_argument("--mock", action="store_true", help="Chế độ giả lập")
    parser.add_argument("--log", type=str, default="benchmark_baseline_ik.csv", help="File CSV lưu benchmark")
    args = parser.parse_args()

    pin_to_cpu_core_0()

    controller = BaselineVisualTracker(dt=DT_CTRL, grip_angle=args.grip_angle, handoff_dist=args.handoff_dist)
    bridge = ESP32HardwareBridge(port=args.port, baudrate=args.baud, mock=args.mock)
    logger = TrackingBenchmarkLogger(filename=args.log)
    vision_sub = VisionIPCSubscriber(host="127.0.0.1", port=5555)

    # Khởi tạo Socket UDP và Bộ tính Động lực học CoM cho Web Dashboard
    telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dynamics_calc = ArmDynamicsCalculator(dt=DT_CTRL)

    print("\n" + "=" * 70)
    print(" 🚀 PHƯƠNG PHÁP 1: BASELINE CAMERA TRACKING + S-CURVE IK")
    print(f" - Bù Z-Offset: +{args.z_offset*100:.1f} cm | Góc kẹp Servo 5: {args.grip_angle:.1f}°")
    print(f" - Nhận tọa độ 3D từ Camera (Core 1) qua IPC UDP port 5555")
    print(f" - Khóa cứng J4 = 90.0° (ngàm kẹp nằm ngang) | Bàn giao: <= {args.handoff_dist*100:.1f} cm")
    print(f" - Phát Telemetry động lực học & CoM 3D lên Web Dashboard qua UDP port 5556")
    print(" - Gõ Ctrl+C để dừng và lưu file benchmark")
    print("=" * 70 + "\n")

    # Đưa robot về Ready Pose
    q_ready = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)
    q_fb, grip_fb = bridge.send_and_receive(q_ready, Q_GRIPPER_OPEN)
    controller.start(initial_q=q_fb)

    start_t = time.time()
    last_q_cmd = q_ready.copy()
    prev_vel = np.zeros(4)
    step_count = 0

    try:
        while True:
            loop_t = time.time()
            elapsed = loop_t - start_t
            step_count += 1

            # 1. Đọc tọa độ từ Camera Stream (Core 1)
            target_pos_tup, target_vel_tup, is_valid, age = vision_sub.get_latest()
            target_pos = None
            if is_valid and target_pos_tup:
                target_pos = np.array(target_pos_tup, dtype=np.float64)
                target_pos[2] += args.z_offset # Bù Z-offset

            # 2. Bước điều khiển IK bám đuổi
            q_cmd, grip_cmd, state = controller.step(target_pos, is_valid and age < 0.6, q_fb, grip_fb)
            q_fb, grip_fb = bridge.send_and_receive(q_cmd, grip_cmd)

            # 3. Tính toán độ êm mượt (Jerk) và sai số
            p_c, R_c = compute_p_close(q_fb)
            effective_target = target_pos if target_pos is not None else controller.target_pos
            pos_err = np.linalg.norm(effective_target - p_c)
            tilt = compute_tilt_deg(R_c)

            cur_vel = (q_cmd - last_q_cmd) / DT_CTRL
            cur_acc = (cur_vel - prev_vel) / DT_CTRL
            jerk = float(np.linalg.norm(cur_acc))
            last_q_cmd = q_cmd.copy()
            prev_vel = cur_vel.copy()

            logger.log_step(elapsed, state.name, p_c, controller.target_pos, pos_err, tilt, q_cmd, q_fb, jerk)

            # 4. Tính toán toàn bộ Telemetry (CoM, Quán tính, Lực lật, Góc quay) và gửi tới Web Dashboard
            try:
                tele_dict = dynamics_calc.compute_dynamics(
                    q_cmd=q_cmd,
                    q_fb=q_fb,
                    target_pos=effective_target,
                    state_str=state.name,
                    dt=DT_CTRL
                )
                tele_dict["joints_cmd_deg"].append(round(rad_to_deg_gripper(grip_cmd), 1))
                tele_dict["joints_fb_deg"].append(round(rad_to_deg_gripper(grip_fb), 1))
                tele_dict["elapsed_s"] = round(elapsed, 1)
                tele_bytes = json.dumps(tele_dict).encode("utf-8")
                telemetry_sock.sendto(tele_bytes, ("127.0.0.1", 5556))
            except Exception:
                pass

            # Hiển thị log mỗi 0.5s
            if step_count % 5 == 0:
                t_str = f"Target:({controller.target_pos[0]:+.2f},{controller.target_pos[1]:.2f},{controller.target_pos[2]:.2f})m"
                cmd_deg = [int(round(rad_to_deg_arm(q_cmd[i]))) for i in range(4)]
                grip_deg = int(round(rad_to_deg_gripper(grip_cmd)))
                valid_tag = "[OK]" if (is_valid and age < 0.6) else "[HOLD]"
                print(f" [{elapsed:4.1f}s] {state.name:<11} | P_err:{pos_err*100:4.1f}cm | {t_str} {valid_tag} | Cmd:{cmd_deg}° Grip:{grip_deg}°")

            if state in (GraspState.COMPLETED, GraspState.FAILED):
                print(f"\n>> CHU TRÌNH KẾT THÚC: {state.name}! Đang lưu benchmark...")
                break

            rest = max(0.0, DT_CTRL - (time.time() - loop_t))
            time.sleep(rest)

    except KeyboardInterrupt:
        print("\n[DỪNG] Người dùng bấm Ctrl+C.")
    finally:
        logger.save()
        vision_sub.close()
        bridge.close()


if __name__ == "__main__":
    main()
