#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
BASELINE ĐIỀU KHIỂN GẮP THÔ BẰNG NGHỊCH ĐẢO ĐỘNG HỌC (IK) CHO RASPBERRY PI 4B
=============================================================================
Dự án: ASSARM 4-DOF Robot Arm + Parallel Gripper
Mục đích:
  - Cung cấp thuật toán Baseline điều khiển truyền thống (IK + S-Curve + Máy trạng thái).
  - Chạy trực tiếp trên Raspberry Pi 4B (khóa cố định trên CPU Core 0).
  - Kiểm tra vùng làm việc (Workspace) trước khi thực thi để bảo vệ servo.
  - Hỗ trợ nhập tọa độ 3D (X, Y, Z, Yaw) trực tiếp từ bàn phím.
  - Tạo quỹ đạo chuyển động mượt mà bằng đường cong S-Curve (Quintic Smoothstep).
  - Quy đổi góc chuẩn xác: 4 servo tay [0-180°], Home duỗi thẳng = 90°; Gripper [0-90°].
  - Giao tiếp thời gian thực với ESP32 Dual-Core qua UART (baud 115200).
=============================================================================
"""

import os
import sys
import time
import argparse
import csv
from enum import Enum
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

# Thử import serial (cho kết nối thực tế Pi 4B -> ESP32)
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ===========================================================================
# 1. KHÓA TIẾN TRÌNH TRÊN CPU CORE 0 CỦA RASPBERRY PI 4B
# ===========================================================================

def pin_to_cpu_core_0():
    """
    Khóa toàn bộ tiến trình điều khiển và giao tiếp UART chạy duy nhất trên Core 0 của Pi 4B.
    Giải phóng Core 1 cho Vision/Camera và Core 2, 3 cho các tác vụ khác.
    """
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})
            print("[HỆ THỐNG] ✅ Đã khóa tiến trình điều khiển & UART chạy cố định trên CPU CORE 0 của Pi 4B.")
        else:
            print("[HỆ THỐNG] ℹ️ Hệ điều hành không hỗ trợ sched_setaffinity (bỏ qua).")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể ghim CPU core: {e}")


# ===========================================================================
# 2. HẰNG SỐ KÍCH THƯỚC ĐỘNG HỌC VÀ HIỆU CHUẨN PHẦN CỨNG (URDF & CALIB)
# ===========================================================================

# Giới hạn góc radian của 4 khớp tay [-pi/2, +pi/2] tương ứng [0 deg, 180 deg]
# Khớp duỗi thẳng (Home) tại 0.0 rad tương ứng 90 deg trên servo
Q_LO_ARM = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2], dtype=np.float64)
Q_HI_ARM = np.array([+np.pi / 2, +np.pi / 2, +np.pi / 2, +np.pi / 2], dtype=np.float64)

# Vận tốc thay đổi góc tối đa mỗi bước điều khiển (10 Hz -> dt = 0.1s -> max ~2.8 deg/step)
MAX_DELTA_Q = 0.05  # rad/step

# Gripper dải hoạt động CHỈ 0 - 90 độ (0.0 rad = 0° mở, 0.896 rad ≈ 51.3° kẹp khối 5cm, 90° đóng kịch)
Q_GRIPPER_OPEN = 0.0          # rad (0 độ)
Q_GRIPPER_CLOSE = 0.896       # rad (~51.3 độ: khít khối 5cm)
Q_GRIPPER_MAX = np.pi / 2     # rad (90 độ: đóng kịch)
VERIFY_WINDOW_LO = 0.82       # rad (~47 độ)
VERIFY_WINDOW_HI = 0.97       # rad (~55.6 độ)

# Offset tâm lòng ngàm kẹp trong hệ tọa độ Plate_1 (m)
# Trục xoay ngón tại z=-0.022m, miếng cao su kẹp tại z=-0.065m, y=+0.0105m
GRIP_CLOSE_OFFSET = np.array([0.0, 0.0105, -0.065], dtype=np.float64)

# Điều kiện bàn giao (Handoff threshold) cho khối trụ tròn 10cm
HANDOFF_EX = 0.015    # ±1.5 cm sai số ngang
HANDOFF_EY = 0.015    # ±1.5 cm sai số tiến/lùi
HANDOFF_EZ = 0.015    # ±1.5 cm sai số cao độ
HANDOFF_TILT_DEV_MAX = np.radians(30.0)  # Góc nghiêng ngang thân trụ (70° đến 110°)
HANDOFF_CONSEC_REQUIRED = 3              # Cần ổn định liên tiếp 3 bước (0.3s)

# Thông số Calib ESP32 theo sơ đồ mới sodonguyenly_calibservo (Cầu chia áp 47k+47k, Ratio 2.0)
# - J1 (TD8120MG):           PWM = GPIO27 (Pin 12) | FB = GPIO35 (Pin 7)
# - J2 (RDS3120MG):          PWM = GPIO26 (Pin 11) | FB = GPIO34 (Pin 6)
# - J3 (MG995):              PWM = GPIO25 (Pin 10) | FB = GPIO39 / SENSOR_VN (Pin 5)
# - J4 (MG995):              PWM = GPIO33 (Pin 9)  | FB = GPIO36 / SENSOR_VP (Pin 4)
# - Servo 5 (MG996R độ 360): PWM = GPIO32 (Pin 8)  | Không có chân FB
SERVO_CALIB = {
    "J1": {"pulseMin": 490, "pulseMax": 2520, "adc0": 2533, "adcMid": 1616, "adc180": 705, "ratio": 2.0},
    "J2": {"pulseMin": 510, "pulseMax": 2440, "adc0": 756,  "adcMid": 3220, "adc180": 5790, "ratio": 2.0},
    "J3": {"pulseMin": 500, "pulseMax": 2500, "adc0": 494,  "adcMid": 3332, "adc180": 6186, "ratio": 2.0},
    "J4": {"pulseMin": 500, "pulseMax": 2500, "adc0": 494,  "adcMid": 3332, "adc180": 6186, "ratio": 2.0},
}


# ===========================================================================
# 3. QUY ĐỔI GÓC KHỚP VÀ SERVO THEO ĐÚNG ĐẶC TÍNH VẬT LÝ
# ===========================================================================

def rad_to_deg_arm(rad: float) -> float:
    """
    Quy đổi góc khớp tay: radian [-pi/2, +pi/2] -> độ servo [0, 180].
    Góc 0.0 rad (duỗi thẳng Home) tương ứng đúng 90.0 độ servo.
    """
    deg = float(np.degrees(rad)) + 90.0
    return float(np.clip(deg, 0.0, 180.0))

def deg_to_rad_arm(deg: float) -> float:
    """Quy đổi độ servo [0, 180] -> radian [-pi/2, +pi/2]."""
    return float(np.radians(np.clip(deg, 0.0, 180.0) - 90.0))

def rad_to_deg_gripper(rad: float) -> float:
    """
    Quy đổi góc gripper: radian [0, pi/2] -> độ servo [0, 90].
    0.0 rad = 0 độ (mở hoàn toàn).
    0.896 rad ≈ 51.3 độ (kẹp chặt khối 5cm).
    pi/2 rad = 90 độ (đóng tối đa).
    """
    deg = float(np.degrees(rad))
    return float(np.clip(deg, 0.0, 90.0))

def deg_to_rad_gripper(deg: float) -> float:
    """Quy đổi độ servo gripper [0, 90] -> radian [0, pi/2]."""
    return float(np.radians(np.clip(deg, 0.0, 90.0)))


# ===========================================================================
# 4. ĐỘNG HỌC THUẬN (FK) VÀ JACOBIAN SỐ
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

# Các ma trận biến đổi tĩnh trích từ URDF chuẩn của ASSARM
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
    """
    Tính động học thuận cho 4 khớp tay [J1, J2, J3, J4] (rad).
    Trả về:
      position: Tọa độ tâm Plate_1 trong hệ base (m)
      R: Ma trận xoay 3x3 của Plate_1
    """
    T_j1 = _homogeneous(_Rz(-q[0]), np.zeros(3))
    T_j2 = _homogeneous(_Rx(-q[1]), np.zeros(3))
    T_j3 = _homogeneous(_Rx(-q[2]), np.zeros(3))
    T_j4 = _homogeneous(_Rz(-q[3]), np.zeros(3))

    T = (_T_pre_j1 @ T_j1
         @ _T_j1_to_j2 @ T_j2
         @ _T_j2_to_j3 @ T_j3
         @ _T_j3_to_j4 @ T_j4
         @ _T_j4_to_plate)

    return T[:3, 3].copy(), T[:3, :3].copy()


def compute_p_close(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tính tọa độ tâm điểm gắp P_close (m) và ma trận xoay R của gripper."""
    pos, R = fk_arm(q)
    p_c = pos + R @ GRIP_CLOSE_OFFSET
    return p_c, R


def compute_tilt_deg(R: np.ndarray) -> float:
    """Độ nghiêng của trục gripper so với phương thẳng đứng (độ)."""
    cos_tilt = np.clip(R[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


def jacobian_numerical(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Tính ma trận Jacobian 3x4 (vị trí P_close theo 4 khớp) bằng đạo hàm số."""
    J = np.zeros((3, 4), dtype=np.float64)
    for i in range(4):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        p_plus, _ = compute_p_close(q_plus)
        p_minus, _ = compute_p_close(q_minus)
        J[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return J


# ===========================================================================
# 5. CƠ CHẾ S-CURVE (LÀM MƯỢT QUỸ ĐẠO BẰNG QUINTIC SMOOTHSTEP)
# ===========================================================================

def scurve_factor(tau: float) -> float:
    """
    Hàm S-Curve đa thức bậc 5 (Quintic Polynomial / Smoothstep):
      s(tau) = 6*tau^5 - 15*tau^4 + 10*tau^3,  tau in [0, 1]
    Đặc điểm:
      - s(0) = 0, s(1) = 1
      - Vận tốc tại 2 đầu = 0: s'(0) = s'(1) = 0
      - Gia tốc tại 2 đầu = 0: s''(0) = s''(1) = 0 (Triệt tiêu hiện tượng giật rung servo)
    """
    tau = np.clip(tau, 0.0, 1.0)
    return float(6.0 * (tau ** 5) - 15.0 * (tau ** 4) + 10.0 * (tau ** 3))


def generate_scurve_trajectory(q_start: np.ndarray, q_goal: np.ndarray,
                               duration: float = 2.0, dt: float = 0.1) -> List[np.ndarray]:
    """
    Tạo danh sách các waypoint góc khớp chuyển động mượt mà từ q_start -> q_goal.
    """
    n_steps = max(2, int(duration / dt))
    traj = []
    for k in range(n_steps + 1):
        tau = k / float(n_steps)
        s = scurve_factor(tau)
        q_k = q_start + s * (q_goal - q_start)
        traj.append(q_k)
    return traj


# ===========================================================================
# 6. BỘ KIỂM TRA WORKSPACE & KHẢ THI CỦA IK (PRE-CHECK)
# ===========================================================================

def check_ik_feasibility(target_pos: np.ndarray, target_yaw: float = 0.0,
                         max_iter: int = 40) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Kiểm tra xem tọa độ 3D (X, Y, Z) có nằm trong dải làm việc vật lý của 4 servo hay không.
    Trả về:
      is_valid: True nếu đạt yêu cầu, False nếu ngoài workspace.
      message: Mô tả lý do hoặc góc khớp nghiệm tính được.
      q_solution: Góc 4 khớp nếu hợp lệ, None nếu không hợp lệ.
    """
    tx, ty, tz = target_pos

    # 1. Kiểm tra hình học sơ bộ
    # Cánh tay nằm ở phía trước Y > 0
    if ty <= 0.05:
        return False, f"Tọa độ Y={ty:.3f}m <= 0.05m (nằm phía sau hoặc quá sát thân đế robot).", None

    # Khoảng cách 3D từ khớp vai (0, 0, 0.56) tới mục tiêu
    dist_shoulder = np.sqrt(tx**2 + ty**2 + (tz - 0.56)**2)
    if dist_shoulder > 0.35:
        return False, f"Khoảng cách tới mục tiêu ({dist_shoulder*100:.1f}cm) vượt quá tầm với tối đa của tay (35cm).", None
    if dist_shoulder < 0.12:
        return False, f"Khoảng cách tới mục tiêu ({dist_shoulder*100:.1f}cm) quá gần khớp đế (<12cm, va chạm cơ cấu).", None

    # 2. Giải thử bài toán IK bằng Damped Least Squares
    # Khởi tạo điểm đoán ban đầu: hướng J1 theo mục tiêu, J2 & J3 gập sẵn hướng về phía trước
    theta_j1 = np.arctan2(tx, ty)
    q = np.array([theta_j1, -0.40, -0.60, -(target_yaw + theta_j1)], dtype=np.float64)

    for _ in range(max_iter):
        p_cur, _ = compute_p_close(q)
        err = target_pos - p_cur
        dist = np.linalg.norm(err)
        if dist < 0.010:  # Đã hội tụ với sai số < 1 cm
            break

        J = jacobian_numerical(q)
        step_mag = min(1.0, 0.03 / max(dist, 1e-4))
        dx = err * step_mag
        dq = np.linalg.solve(J.T @ J + 1e-3 * np.eye(4), J.T @ dx)

        # Căn chỉnh J4 theo yaw
        dq[3] = 0.5 * (-(target_yaw + q[0]) - q[3])
        dq = np.clip(dq, -0.06, 0.06)
        q += dq

    # 3. Kiểm tra kết quả hội tụ
    final_p, final_R = compute_p_close(q)
    final_dist = np.linalg.norm(target_pos - final_p)
    tilt = compute_tilt_deg(final_R)

    if final_dist > 0.020:
        return False, f"Không thể hội tụ tới mục tiêu (Sai số còn lại {final_dist*100:.1f}cm > 2cm).", None

    # 4. Kiểm tra giới hạn góc từng servo (phải nằm trong [-pi/2, +pi/2] tương ứng [0, 180 deg])
    servo_deg = [rad_to_deg_arm(q[i]) for i in range(4)]
    for i, deg in enumerate(servo_deg):
        if deg < 0.0 or deg > 180.0:
            return False, f"Khớp J{i+1} yêu cầu góc {deg:.1f}° (vượt quá dải cho phép 0-180°).", None

    msg = (f"Vị trí hợp lệ trong Workspace! Nghiệm IK: "
           f"J1={servo_deg[0]:.1f}°, J2={servo_deg[1]:.1f}°, J3={servo_deg[2]:.1f}°, J4={servo_deg[3]:.1f}° | "
           f"Tilt={tilt:.1f}° | Sai số={final_dist*1000:.1f}mm")
    return True, msg, q


# ===========================================================================
# 7. MÁY TRẠNG THÁI GẮP KẾT HỢP S-CURVE
# ===========================================================================

class GraspState(Enum):
    IDLE = 0
    READY = 1       # Tư thế sẵn sàng vươn tới vùng làm việc phía trước
    APPROACH = 2    # IK tiếp cận chính xác tọa độ mục tiêu
    CLOSE = 3       # Đóng ngàm kẹp
    VERIFY = 4      # Xác thực kẹp trúng
    LIFT = 5        # Nâng vật lên
    HOLD = 6        # Giữ ổn định
    COMPLETED = 7   # Hoàn thành xuất sắc
    FAILED = 8      # Thất bại


class BaselineIKController:
    """
    Bộ điều khiển IK Baseline toàn trình tích hợp S-Curve và kiểm tra Workspace.
    """
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.state = GraspState.IDLE
        self.consec_handoff_count = 0
        self.state_timer = 0.0
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.q_cmd = np.zeros(4)
        self.gripper_cmd = Q_GRIPPER_OPEN
        self.initial_grasp_q = np.zeros(4)
        # Tư thế sẵn sàng vươn về phía trước (tránh điểm kỳ dị khi tay buông thẳng)
        self.q_ready = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)
        self.q_goal = None

        # Quỹ đạo S-Curve đang thực thi
        self.scurve_traj: List[np.ndarray] = []
        self.scurve_index: int = 0
        self.last_sent_q = np.zeros(4)

    def set_target(self, pos: np.ndarray, yaw: float = 0.0, current_q: Optional[np.ndarray] = None,
                   q_goal: Optional[np.ndarray] = None):
        """Đặt vị trí mục tiêu khối cần gắp và bắt đầu tiến trình."""
        self.target_pos = np.array(pos, dtype=np.float64)
        self.target_yaw = float(yaw)
        self.state = GraspState.READY
        self.consec_handoff_count = 0
        self.state_timer = 0.0
        if current_q is not None:
            self.q_cmd = current_q.copy()
            self.last_sent_q = current_q.copy()
        else:
            self.q_cmd = np.zeros(4)
            self.last_sent_q = np.zeros(4)
        self.gripper_cmd = Q_GRIPPER_OPEN
        self.q_goal = q_goal.copy() if q_goal is not None else None

        # 1. Quỹ đạo S-Curve Giai đoạn 1: Chuyển từ vị trí hiện tại sang Ready pose trong 1.5s
        theta_target = np.arctan2(self.target_pos[0], self.target_pos[1])
        ready_target = self.q_ready.copy()
        ready_target[0] = theta_target
        ready_target[3] = -(self.target_yaw + theta_target)

        self.scurve_traj = generate_scurve_trajectory(self.q_cmd, ready_target, duration=1.5, dt=self.dt)
        self.scurve_index = 0

    def step(self, q_feedback: np.ndarray, gripper_feedback: float) -> Tuple[np.ndarray, float, GraspState]:
        """
        Thực hiện một chu kỳ điều khiển (10 Hz).
        Trả về: (q_target_4d, gripper_target, current_state)
        """
        self.state_timer += self.dt
        p_cur, R_cur = compute_p_close(q_feedback)

        # -------------------------------------------------------------
        # 0. PHA READY (Mở tay từ Home về tư thế chuẩn bị qua S-Curve)
        # -------------------------------------------------------------
        if self.state == GraspState.READY:
            self.gripper_cmd = Q_GRIPPER_OPEN
            if self.scurve_index < len(self.scurve_traj):
                self.q_cmd = self.scurve_traj[self.scurve_index]
                self.scurve_index += 1
            else:
                self.state = GraspState.APPROACH
                self.state_timer = 0.0
                # Tạo quỹ đạo S-Curve Giai đoạn 2: Tiếp cận cực kỳ mượt mà tới nghiệm IK mục tiêu trong 2.0s
                goal_target = self.q_goal if self.q_goal is not None else self.q_cmd
                self.scurve_traj = generate_scurve_trajectory(self.q_cmd, goal_target, duration=2.0, dt=self.dt)
                self.scurve_index = 0

        # -------------------------------------------------------------
        # 1. PHA APPROACH (Tiếp cận mục tiêu theo S-Curve Quintic siêu êm)
        # -------------------------------------------------------------
        elif self.state == GraspState.APPROACH:
            self.gripper_cmd = Q_GRIPPER_OPEN

            if self.scurve_index < len(self.scurve_traj):
                self.q_cmd = self.scurve_traj[self.scurve_index]
                self.scurve_index += 1
            else:
                # Đã hoàn tất hành trình S-Curve tới điểm gắp -> Chuyển sang đóng kẹp
                self.state = GraspState.CLOSE
                self.state_timer = 0.0
                self.initial_grasp_q = self.q_cmd.copy()
                return self.q_cmd, self.gripper_cmd, self.state

            # Quá 15s tiếp cận mà không tới được -> FAILED
            if self.state_timer > 15.0:
                self.state = GraspState.FAILED
                return self.q_cmd, self.gripper_cmd, self.state

        # -------------------------------------------------------------
        # 2. PHA CLOSE (Đóng ngàm kẹp theo S-Curve)
        # -------------------------------------------------------------
        elif self.state == GraspState.CLOSE:
            self.q_cmd = self.initial_grasp_q.copy()
            # Khép gripper dần về vị trí kẹp Q_GRIPPER_CLOSE
            self.gripper_cmd = min(Q_GRIPPER_CLOSE, self.gripper_cmd + 0.15)

            if self.state_timer >= 1.2 or self.gripper_cmd >= Q_GRIPPER_CLOSE:
                self.state = GraspState.VERIFY
                self.state_timer = 0.0

        # -------------------------------------------------------------
        # 3. PHA VERIFY (Xác nhận kẹp chặt khối)
        # -------------------------------------------------------------
        elif self.state == GraspState.VERIFY:
            self.q_cmd = self.initial_grasp_q.copy()
            if VERIFY_WINDOW_LO <= gripper_feedback <= VERIFY_WINDOW_HI:
                self.state = GraspState.LIFT
                self.state_timer = 0.0
                # Tạo quỹ đạo S-Curve nâng vật lên cao +5cm trong 1.5s
                target_lift_q = self.initial_grasp_q.copy()
                target_lift_q[1] += np.radians(8.0)
                target_lift_q[2] -= np.radians(8.0)
                self.scurve_traj = generate_scurve_trajectory(self.q_cmd, target_lift_q, duration=1.5, dt=self.dt)
                self.scurve_index = 0
            elif self.state_timer >= 1.0:
                self.state = GraspState.FAILED

        # -------------------------------------------------------------
        # 4. PHA LIFT (Nâng vật lên cao theo S-Curve)
        # -------------------------------------------------------------
        elif self.state == GraspState.LIFT:
            if self.scurve_index < len(self.scurve_traj):
                self.q_cmd = self.scurve_traj[self.scurve_index]
                self.scurve_index += 1
            else:
                self.state = GraspState.HOLD
                self.state_timer = 0.0

        # -------------------------------------------------------------
        # 5. PHA HOLD (Giữ ổn định 3 giây)
        # -------------------------------------------------------------
        elif self.state == GraspState.HOLD:
            if self.state_timer >= 3.0:
                self.state = GraspState.COMPLETED

        return self.q_cmd, self.gripper_cmd, self.state


# ===========================================================================
# 8. GIAO TIẾP VỚI ESP32 QUA UART (HOẶC MOCK MODE)
# ===========================================================================

class ESP32HardwareBridge:
    """
    Cầu nối truyền nhận lệnh giữa Pi 4B và ESP32 qua UART.
    Hỗ trợ đọc liên tục feedback ADC & góc servo từ Core 0 của ESP32.
    """
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
                print("[CẢNH BÁO] Chưa cài thư viện pyserial. Tự động chuyển sang chế độ MOCK!")
                self.mock = True
            else:
                try:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    time.sleep(1.0)  # Chờ ESP32 khởi động
                    self.flush()
                    print(f"[UART] ✅ Đã kết nối thành công tới ESP32 tại {self.port} ({self.baudrate} baud)")
                except Exception as e:
                    print(f"[LỖI UART] Không thể mở cổng {self.port}: {e}. Chuyển sang MOCK mode.")
                    self.mock = True

    def flush(self):
        """Xóa sạch bộ đệm RX cũ để đảm bảo nhận số đo thời gian thực."""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def send_and_receive(self, q_target: np.ndarray, grip_target: float) -> Tuple[np.ndarray, float]:
        """
        Gửi lệnh góc khớp xuống ESP32 và nhận về góc feedback thực tế mới nhất.
        Quy ước góc gửi xuống:
          4 servo tay: [0, 180°], Home duỗi thẳng = 90°
          Gripper: [0, 90°], 0° = mở, 51° = kẹp 5cm, 90° = đóng kịch
        """
        if self.mock:
            # Mô phỏng phản hồi động học trễ nhẹ của servo vật lý
            alpha = 0.35
            self.sim_q += alpha * (q_target - self.sim_q)
            self.sim_grip += alpha * (grip_target - self.sim_grip)
            return self.sim_q.copy(), float(self.sim_grip)

        # Chế độ phần cứng thật: Gửi chuỗi lệnh qua Serial
        deg1 = rad_to_deg_arm(q_target[0])
        deg2 = rad_to_deg_arm(q_target[1])
        deg3 = rad_to_deg_arm(q_target[2])
        deg4 = rad_to_deg_arm(q_target[3])
        degG = rad_to_deg_gripper(grip_target)

        cmd_str = f"SET {deg1:.1f} {deg2:.1f} {deg3:.1f} {deg4:.1f} {degG:.1f}\n"
        try:
            self.ser.write(cmd_str.encode("ascii"))

            # Đọc xả hết các dòng cũ đọng trong bộ đệm để bắt đúng dòng mới nhất
            latest_fb_line = None
            while self.ser.in_waiting > 0:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l

            # Nếu đệm trống, đọc chờ 1 dòng mới từ ESP32
            if latest_fb_line is None:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l

            if latest_fb_line and latest_fb_line.startswith("FB"):
                # ESP32 Core 0 phản hồi: "FB <deg1> <deg2> <deg3> <deg4> <degGrip> | ADC <a1> <a2>..."
                main_part = latest_fb_line.split("|")[0].strip()
                parts = main_part.split()
                if len(parts) >= 6:
                    self.last_fb_q = np.array([
                        deg_to_rad_arm(float(parts[1])),
                        deg_to_rad_arm(float(parts[2])),
                        deg_to_rad_arm(float(parts[3])),
                        deg_to_rad_arm(float(parts[4]))
                    ])
                    self.last_fb_grip = deg_to_rad_gripper(float(parts[5]))

                if "|" in latest_fb_line:
                    adc_tokens = latest_fb_line.split("|")[1].strip().split()
                    if len(adc_tokens) >= 5 and adc_tokens[0] == "ADC":
                        self.last_adc = [int(adc_tokens[1]), int(adc_tokens[2]), int(adc_tokens[3]), int(adc_tokens[4])]

                return self.last_fb_q.copy(), float(self.last_fb_grip)

        except Exception as e:
            print(f"[UART LỖI]: {e}")

        return self.last_fb_q.copy() if hasattr(self, 'last_fb_q') else q_target.copy(), float(grip_target)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ===========================================================================
# 9. BỘ ĐO ĐẠC VÀ XUẤT DỮ LIỆU ĐÁNH GIÁ (BENCHMARK LOGGER)
# ===========================================================================

class BenchmarkLogger:
    """Ghi nhật ký chuyển động và chỉ số để so sánh với RL Policy."""
    def __init__(self, filename: str = "baseline_benchmark.csv"):
        self.filename = filename
        self.records = []

    def log_step(self, t: float, state: str, p_close: np.ndarray, target: np.ndarray,
                 pos_err: float, tilt_deg: float, q_cmd: np.ndarray, q_fb: np.ndarray):
        self.records.append({
            "time_s": f"{t:.2f}",
            "state": state,
            "px": f"{p_close[0]:.4f}", "py": f"{p_close[1]:.4f}", "pz": f"{p_close[2]:.4f}",
            "tx": f"{target[0]:.4f}",  "ty": f"{target[1]:.4f}",  "tz": f"{target[2]:.4f}",
            "pos_error_m": f"{pos_err:.4f}",
            "tilt_deg": f"{tilt_deg:.2f}",
            "s1_cmd_deg": f"{rad_to_deg_arm(q_cmd[0]):.1f}",
            "s2_cmd_deg": f"{rad_to_deg_arm(q_cmd[1]):.1f}",
            "s3_cmd_deg": f"{rad_to_deg_arm(q_cmd[2]):.1f}",
            "s4_cmd_deg": f"{rad_to_deg_arm(q_cmd[3]):.1f}",
            "s1_fb_deg": f"{rad_to_deg_arm(q_fb[0]):.1f}",
            "s2_fb_deg": f"{rad_to_deg_arm(q_fb[1]):.1f}",
            "s3_fb_deg": f"{rad_to_deg_arm(q_fb[2]):.1f}",
            "s4_fb_deg": f"{rad_to_deg_arm(q_fb[3]):.1f}",
        })

    def save(self):
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[BENCHMARK] Đã lưu {len(self.records)} dòng dữ liệu đối chứng vào file: {self.filename}")


# ===========================================================================
# 10. THỰC THI CHU TRÌNH GẮP VÀ CHẾ ĐỘ TƯƠNG TÁC BÀN PHÍM
# ===========================================================================

def execute_grasp_cycle(target_pos: np.ndarray, target_yaw: float,
                        controller: BaselineIKController, bridge: ESP32HardwareBridge,
                        logger: BenchmarkLogger, q_goal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Chạy 1 chu trình gắp hoàn chỉnh cho một tọa độ mục tiêu sau khi đã pass kiểm tra Workspace."""
    print(f"\n==================================================================")
    print(f" 🚀 BẮT ĐẦU CHU TRÌNH GẮP BASELINE IK (S-CURVE)")
    print(f" Mục tiêu: X={target_pos[0]:.3f}m | Y={target_pos[1]:.3f}m | Z={target_pos[2]:.3f}m | Yaw={np.degrees(target_yaw):.1f}°")
    print(f"==================================================================")

    bridge.flush()
    q_fb, grip_fb = bridge.send_and_receive(np.zeros(4), Q_GRIPPER_OPEN)
    controller.set_target(target_pos, target_yaw, current_q=q_fb, q_goal=q_goal)

    start_time = time.time()
    step_count = 0
    dt_ctrl = controller.dt
    handoff_time = None
    final_state = GraspState.FAILED

    while True:
        loop_start = time.time()
        step_count += 1
        elapsed = time.time() - start_time

        q_cmd, grip_cmd, state = controller.step(q_fb, grip_fb)
        q_fb, grip_fb = bridge.send_and_receive(q_cmd, grip_cmd)

        p_c, R_c = compute_p_close(q_fb)
        pos_err = np.linalg.norm(target_pos - p_c)
        tilt = compute_tilt_deg(R_c)

        logger.log_step(elapsed, state.name, p_c, target_pos, pos_err, tilt, q_cmd, q_fb)

        if state == GraspState.CLOSE and handoff_time is None:
            handoff_time = elapsed
            print(f" -> [HANDOFF ĐẠT] Tại t={handoff_time:.2f}s | Sai số: {pos_err*100:.2f}cm | Tilt: {tilt:.1f}°")

        if step_count % 5 == 0:
            c1, c2, c3, c4 = [rad_to_deg_arm(q_cmd[i]) for i in range(4)]
            f1, f2, f3, f4 = [rad_to_deg_arm(q_fb[i]) for i in range(4)]
            grip_deg = rad_to_deg_gripper(grip_fb)
            print(f" [{elapsed:4.1f}s] {state.name:<8} | P_err:{pos_err*100:4.1f}cm | Cmd:[{c1:4.1f},{c2:4.1f},{c3:4.1f},{c4:4.1f}] | Fb:[{f1:4.1f},{f2:4.1f},{f3:4.1f},{f4:4.1f}] | ADC:{bridge.last_adc}")

        if state in (GraspState.COMPLETED, GraspState.FAILED):
            final_state = state
            break

        if elapsed > 25.0:
            print("[TIMEOUT] Quá thời gian tối đa 25s!")
            final_state = GraspState.FAILED
            break

        comp_time = time.time() - loop_start
        sleep_time = max(0.0, dt_ctrl - comp_time)
        time.sleep(sleep_time)

    total_duration = time.time() - start_time
    is_success = (final_state == GraspState.COMPLETED)

    print(f"------------------------------------------------------------------")
    print(f" KẾT QUẢ: {'THÀNH CÔNG (SUCCESS) ✅' if is_success else 'THẤT BẠI (FAILED) ❌'}")
    print(f" Tổng thời gian: {total_duration:.2f} s")
    if handoff_time:
        print(f" Thời gian tiếp cận (Handoff): {handoff_time:.2f} s")
    print(f" Sai số vị trí cuối cùng: {pos_err*1000:.1f} mm")
    print(f"==================================================================\n")

    logger.save()

    return {
        "success": is_success,
        "total_time": total_duration,
        "handoff_time": handoff_time if handoff_time else -1.0,
        "final_error_mm": pos_err * 1000.0,
    }


def handle_user_command(target_pos: np.ndarray, target_yaw: float,
                        controller: BaselineIKController, bridge: ESP32HardwareBridge,
                        logger: BenchmarkLogger) -> bool:
    """Kiểm tra tính khả thi trước, nếu hợp lệ thì thực thi, nếu ngoài workspace thì thông báo từ chối."""
    is_valid, msg, q_sol = check_ik_feasibility(target_pos, target_yaw)
    if not is_valid:
        print(f"\n❌ [TỪ CHỐI THỰC THI] Điểm ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}) NẰM NGOÀI WORKSPACE!")
        print(f"   Lý do: {msg}")
        print(f"   => Cánh tay được giữ an toàn tại chỗ, không kích hoạt servo.\n")
        return False

    print(f"\n✅ {msg}")
    execute_grasp_cycle(target_pos, target_yaw, controller, bridge, logger, q_goal=q_sol)
    return True


def run_interactive_console(controller: BaselineIKController, bridge: ESP32HardwareBridge, logger: BenchmarkLogger):
    """Giao diện dòng lệnh tương tác cho phép người dùng gõ tọa độ mong muốn."""
    print("\n" + "=" * 65)
    print(" 🎮 CHẾ ĐỘ NHẬP TỌA ĐỘ TƯƠNG TÁC (INTERACTIVE MODE)")
    print(" - Nhập tọa độ mong muốn: X Y Z [Yaw_độ] (Ví dụ: 0.0 0.28 0.40)")
    print(" - Nhập 'home' để về vị trí ban đầu (duỗi thẳng 90°)")
    print(" - Nhập 'open' để mở ngàm kẹp, 'close' để đóng kẹp")
    print(" - Nhập 'exit' hoặc 'quit' để thoát và lưu dữ liệu benchmark")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("ASSARM (X Y Z) >>> ").strip().lower()
            if not user_input:
                continue
            if user_input in ("exit", "quit", "q"):
                print("[THOÁT] Kết thúc chế độ tương tác.")
                break

            if user_input == "home":
                print("[LỆNH] Đang đưa cánh tay về vị trí Home [0, 0, 0, 0] (Servos 90°)...")
                q_cur, _ = bridge.send_and_receive(np.zeros(4), Q_GRIPPER_OPEN)
                traj = generate_scurve_trajectory(q_cur, np.zeros(4), duration=1.5, dt=0.1)
                for q_step in traj:
                    bridge.send_and_receive(q_step, Q_GRIPPER_OPEN)
                    time.sleep(0.1)
                print(" -> Đã về vị trí Home an toàn.")
                continue

            if user_input == "open":
                print("[LỆNH] Mở ngàm kẹp (0°)...")
                q_cur, _ = bridge.send_and_receive(np.zeros(4), Q_GRIPPER_OPEN)
                bridge.send_and_receive(q_cur, Q_GRIPPER_OPEN)
                continue

            if user_input == "close":
                print("[LỆNH] Đóng ngàm kẹp kẹp khối 5cm (51.3°)...")
                q_cur, _ = bridge.send_and_receive(np.zeros(4), Q_GRIPPER_CLOSE)
                bridge.send_and_receive(q_cur, Q_GRIPPER_CLOSE)
                continue

            tokens = user_input.replace(",", " ").split()
            if len(tokens) < 3:
                print("⚠️ Cú pháp không đúng! Vui lòng nhập ít nhất 3 số: X Y Z (ví dụ: 0.0 0.28 0.40)")
                continue

            tx = float(tokens[0])
            ty = float(tokens[1])
            tz = float(tokens[2])
            tyaw = np.radians(float(tokens[3])) if len(tokens) >= 4 else 0.0

            handle_user_command(np.array([tx, ty, tz]), tyaw, controller, bridge, logger)

        except ValueError:
            print("⚠️ Lỗi định dạng số! Vui lòng nhập các giá trị số hợp lệ.")
        except KeyboardInterrupt:
            print("\n[DỪNG] Người dùng bấm Ctrl+C.")
            break


# ===========================================================================
# 11. HÀM MAIN KHỞI CHẠY
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline IK Controller cho ASSARM trên Pi 4B (Core 0 / S-Curve)")
    parser.add_argument("--mock", action="store_true", help="Chạy chế độ giả lập không cần kết nối UART")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Cổng UART nối ESP32 (mặc định: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Tốc độ Baud UART (mặc định: 115200)")
    parser.add_argument("--x", type=float, default=None, help="Tọa độ X mục tiêu (m)")
    parser.add_argument("--y", type=float, default=None, help="Tọa độ Y mục tiêu (m)")
    parser.add_argument("--z", type=float, default=None, help="Tọa độ Z mục tiêu (m)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Góc Yaw mục tiêu (độ, mặc định: 0.0)")
    parser.add_argument("--log", type=str, default="baseline_benchmark.csv", help="File CSV lưu log")
    parser.add_argument("--batch", action="store_true", help="Chạy kiểm thử trên lưới 10 điểm để lấy benchmark tổng quát")

    args = parser.parse_args()

    # 1. Khóa CPU Core 0 trên Raspberry Pi 4B
    pin_to_cpu_core_0()

    controller = BaselineIKController(dt=0.1)
    bridge = ESP32HardwareBridge(port=args.port, baudrate=args.baud, mock=args.mock)
    logger = BenchmarkLogger(filename=args.log)

    try:
        if args.batch:
            print("[BATCH MODE] Chạy đánh giá Baseline IK trên 10 kịch bản ngẫu nhiên...")
            np.random.seed(42)
            rs = np.random.uniform(0.26, 0.30, size=10)
            thetas = np.random.uniform(np.radians(-25), np.radians(25), size=10)
            zs = np.random.uniform(0.38, 0.42, size=10)
            yaws = np.random.uniform(np.radians(-20), np.radians(20), size=10)

            results = []
            for i in range(10):
                tx = float(rs[i] * np.sin(thetas[i]))
                ty = float(rs[i] * np.cos(thetas[i]))
                tz = float(zs[i])
                tyaw = float(yaws[i])
                res = handle_user_command(np.array([tx, ty, tz]), tyaw, controller, bridge, logger)
                results.append(res)
                time.sleep(1.0)

            n_success = sum(1 for r in results if r)
            print("\n================ TỔNG KẾT BENCHMARK BASELINE IK ================")
            print(f" Tỷ lệ thành công: {n_success}/10 ({n_success*10:.1f}%)")
            print("================================================================")

        elif args.x is not None and args.y is not None and args.z is not None:
            # Nhập trực tiếp qua cờ dòng lệnh CLI
            target_pos = np.array([args.x, args.y, args.z])
            target_yaw = np.radians(args.yaw)
            handle_user_command(target_pos, target_yaw, controller, bridge, logger)

        else:
            # Không truyền tham số -> Mở giao diện nhập tương tác bàn phím
            run_interactive_console(controller, bridge, logger)

        logger.save()

    except KeyboardInterrupt:
        print("\n[DỪNG] Người dùng bấm Ctrl+C để dừng chương trình.")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
