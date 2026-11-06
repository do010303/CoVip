# -*- coding: utf-8 -*-
"""
Module tính toán động học, động lực học và trọng tâm (Center of Mass - CoM)
cho cánh tay robot ASSARM 4-DOF + Gripper.

Tính toán chi tiết:
1. Vị trí 3D từng khớp xương: Base -> J1 -> J2 -> J3 -> J4 -> EE (P_close)
2. Tọa độ trọng tâm toàn cánh tay (CoM) trong hệ tọa độ Robot Base
3. Vận tốc và gia tốc CoM
4. Lực quán tính tổng thể F_inertial = M_arm * ||a_com|| (N)
5. Mô-men lật / uốn lên đế: M_pitch và M_roll (N.m)
6. Độ giật (Jerk) từng servo và tổng hợp RMS
7. Góc hướng ngàm kẹp: Pitch, Yaw, Roll, Tilt
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional


# ===========================================================================
# CẤU HÌNH KHỐI LƯỢNG VÀ THÔNG SỐ VẬT LÝ (TỪ CAD & ASSARM URDF)
# ===========================================================================
# Khối lượng từng phân đoạn (kg)
MASS_LINK0_BASE = 0.088   # Base mount + Servo 1 (TD8120MG)
MASS_LINK1_J1   = 0.083   # Bracket J1 + Servo 2 (RDS3120)
MASS_LINK2_J2   = 0.125   # Link 1 + Brackets + Servo 3 (MG995)
MASS_LINK3_J3   = 0.085   # Link 2 + Servo 4 (MG995)
MASS_LINK4_EE   = 0.127   # Khung kẹp Plate_1 + Servo 5 + 2 ngón kẹp
TOTAL_ARM_MASS  = MASS_LINK0_BASE + MASS_LINK1_J1 + MASS_LINK2_J2 + MASS_LINK3_J3 + MASS_LINK4_EE # ~0.508 kg

# Offset tĩnh từ URDF assarm.xacro
def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def _Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

def _Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def _ht_from_xyz(x: float, y: float, z: float) -> np.ndarray:
    return _homogeneous(np.eye(3, dtype=np.float64), np.array([x, y, z], dtype=np.float64))

# Chuỗi biến đổi hình học từ Base đến Gripper
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
GRIP_CLOSE_OFFSET = np.array([0.0, 0.0105, -0.065], dtype=np.float64)


class ArmDynamicsCalculator:
    """Bộ phân tích động lực học thời gian thực cho ASSARM."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.last_q = None
        self.last_vel = np.zeros(4, dtype=np.float64)
        self.last_acc = np.zeros(4, dtype=np.float64)
        self.last_com = None
        self.last_v_com = np.zeros(3, dtype=np.float64)
        self.last_err = None
        self.v_rel = 0.0

    def compute_kinematics_and_com(self, q: np.ndarray) -> Dict[str, Any]:
        """
        Tính toán forward kinematics chuỗi khớp và tọa độ trọng tâm (CoM).
        q: góc 4 khớp theo rad [q1, q2, q3, q4]
        """
        # Tọa độ gốc Base
        p_base = np.array([0.0, 0.0, 0.56], dtype=np.float64)

        # 1. J1 Origin
        T_j1_origin = _T_pre_j1
        p_j1 = T_j1_origin[:3, 3]

        # Áp dụng quay J1 quanh Z (-q1)
        T_j1_rotated = T_j1_origin @ _homogeneous(_Rz(-q[0]), np.zeros(3))

        # 2. J2 Origin
        T_j2_origin = T_j1_rotated @ _T_j1_to_j2
        p_j2 = T_j2_origin[:3, 3]

        # Áp dụng quay J2 quanh X (-q2)
        T_j2_rotated = T_j2_origin @ _homogeneous(_Rx(-q[1]), np.zeros(3))

        # 3. J3 Origin
        T_j3_origin = T_j2_rotated @ _T_j2_to_j3
        p_j3 = T_j3_origin[:3, 3]

        # Áp dụng quay J3 quanh X (-q3)
        T_j3_rotated = T_j3_origin @ _homogeneous(_Rx(-q[2]), np.zeros(3))

        # 4. J4 Origin
        T_j4_origin = T_j3_rotated @ _T_j3_to_j4
        p_j4 = T_j4_origin[:3, 3]

        # Áp dụng quay J4 quanh Z (-q4)
        T_j4_rotated = T_j4_origin @ _homogeneous(_Rz(-q[3]), np.zeros(3))

        # 5. Plate & Ngàm kẹp
        T_plate = T_j4_rotated @ _T_j4_to_plate
        p_plate = T_plate[:3, 3]
        R_plate = T_plate[:3, :3]
        p_ee = p_plate + R_plate @ GRIP_CLOSE_OFFSET

        # Tọa độ trọng tâm từng link:
        r0 = p_base
        r1 = p_j1 + np.array([0.0, 0.0, -0.015])
        r2 = 0.42 * p_j2 + 0.58 * p_j3
        r3 = 0.48 * p_j3 + 0.52 * p_j4
        r4 = 0.35 * p_j4 + 0.65 * p_ee

        # Tọa độ trọng tâm toàn bộ cánh tay:
        com = (MASS_LINK0_BASE * r0 +
               MASS_LINK1_J1   * r1 +
               MASS_LINK2_J2   * r2 +
               MASS_LINK3_J3   * r3 +
               MASS_LINK4_EE   * r4) / TOTAL_ARM_MASS

        # Tính góc định hướng ngàm kẹp (Pitch, Yaw, Roll, Tilt)
        # Vector hướng z của Plate trong hệ Base
        z_ee = R_plate[:, 2]
        x_ee = R_plate[:, 0]
        y_ee = R_plate[:, 1]

        # Yaw ngàm kẹp trên mặt phẳng XY
        ee_yaw = float(np.degrees(np.arctan2(x_ee[1], x_ee[0])))
        # Pitch: góc chúc xuống của ngàm kẹp
        ee_pitch = float(np.degrees(np.arctan2(-z_ee[1], np.sqrt(z_ee[0]**2 + z_ee[2]**2))))
        # Roll: góc xoay quanh trục vươn
        ee_roll = float(np.degrees(np.arctan2(y_ee[2], y_ee[1])))
        # Tilt: độ nghiêng so với phương thẳng đứng
        ee_tilt = float(np.degrees(np.arccos(np.clip(z_ee[2], -1.0, 1.0))))

        return {
            "p_base": p_base,
            "p_j1": p_j1,
            "p_j2": p_j2,
            "p_j3": p_j3,
            "p_j4": p_j4,
            "p_plate": p_plate,
            "p_ee": p_ee,
            "R_ee": R_plate,
            "com": com,
            "orientation": {
                "yaw": ee_yaw,
                "pitch": ee_pitch,
                "roll": ee_roll,
                "tilt": ee_tilt
            }
        }

    def compute_dynamics(self, q_cmd: np.ndarray, q_fb: np.ndarray,
                         target_pos: Optional[np.ndarray],
                         state_str: str,
                         dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Tính toán toàn bộ thông số động lực học, độ giật, quán tính và sai số.
        """
        dt_val = dt if dt is not None else self.dt
        dt_val = max(0.01, dt_val)

        # 1. Kinematics theo góc feedback thực tế
        k_fb = self.compute_kinematics_and_com(q_fb)
        p_ee = k_fb["p_ee"]
        com = k_fb["com"]

        # 2. Vận tốc, gia tốc, jerk từng servo (deg/s^3)
        q_cmd_deg = np.degrees(q_cmd) + 90.0
        q_fb_deg = np.degrees(q_fb) + 90.0

        if self.last_q is None:
            self.last_q = q_cmd.copy()
            self.last_com = com.copy()

        # Vận tốc khớp (rad/s)
        vel = (q_cmd - self.last_q) / dt_val
        # Gia tốc khớp (rad/s^2)
        acc = (vel - self.last_vel) / dt_val
        # Độ giật Jerk từng servo (deg/s^3)
        jerk_rad = (acc - self.last_acc) / dt_val
        jerk_deg = np.degrees(jerk_rad)
        jerk_total_rms = float(np.sqrt(np.mean(jerk_deg ** 2)))

        # 3. Động lực học trọng tâm CoM
        v_com = (com - self.last_com) / dt_val
        a_com = (v_com - self.last_v_com) / dt_val
        a_com_norm = float(np.linalg.norm(a_com))

        # Lực quán tính tổng thể do chuyển động của cánh tay: F = M * a
        f_inertial = float(TOTAL_ARM_MASS * a_com_norm)

        # Mô-men uốn/lật lên đế (hoặc drone): M = M_arm * g * d (N.m)
        g = 9.81
        m_pitch = float(TOTAL_ARM_MASS * g * com[1]) # Cánh tay đòn theo trục Y tiến
        m_roll = float(TOTAL_ARM_MASS * g * com[0])  # Cánh tay đòn theo trục X ngang

        # 4. Sai số và tốc độ tiếp cận
        pos_err_cm = 0.0
        if target_pos is not None:
            err = np.linalg.norm(target_pos - p_ee)
            pos_err_cm = float(err * 100.0)
            if self.last_err is not None:
                self.v_rel = float((pos_err_cm - self.last_err) / dt_val)
            self.last_err = pos_err_cm

        # Cập nhật trạng thái trước
        self.last_q = q_cmd.copy()
        self.last_vel = vel.copy()
        self.last_acc = acc.copy()
        self.last_com = com.copy()
        self.last_v_com = v_com.copy()

        # Đóng gói danh sách khớp 3D (đổi sang cm để trực quan hiển thị)
        joints_3d = [
            {"name": "Base", "pos": [float(k_fb["p_base"][0] * 100), float(k_fb["p_base"][1] * 100), float(k_fb["p_base"][2] * 100)]},
            {"name": "J1",   "pos": [float(k_fb["p_j1"][0] * 100), float(k_fb["p_j1"][1] * 100), float(k_fb["p_j1"][2] * 100)]},
            {"name": "J2",   "pos": [float(k_fb["p_j2"][0] * 100), float(k_fb["p_j2"][1] * 100), float(k_fb["p_j2"][2] * 100)]},
            {"name": "J3",   "pos": [float(k_fb["p_j3"][0] * 100), float(k_fb["p_j3"][1] * 100), float(k_fb["p_j3"][2] * 100)]},
            {"name": "J4",   "pos": [float(k_fb["p_j4"][0] * 100), float(k_fb["p_j4"][1] * 100), float(k_fb["p_j4"][2] * 100)]},
            {"name": "EE",   "pos": [float(p_ee[0] * 100), float(p_ee[1] * 100), float(p_ee[2] * 100)]},
        ]

        com_cm = [float(com[0] * 100), float(com[1] * 100), float(com[2] * 100)]
        target_cm = [float(target_pos[0] * 100), float(target_pos[1] * 100), float(target_pos[2] * 100)] if target_pos is not None else None

        return {
            "state": state_str,
            "success": state_str in ("HOLD", "COMPLETED", "LIFT"),
            "pos_err_cm": round(pos_err_cm, 2),
            "v_rel_cms": round(self.v_rel, 2),
            "com_cm": [round(c, 2) for c in com_cm],
            "f_inertial_N": round(f_inertial, 3),
            "m_pitch_Nm": round(m_pitch, 3),
            "m_roll_Nm": round(m_roll, 3),
            "jerk_each_degs3": [round(float(abs(j)), 1) for j in jerk_deg],
            "jerk_total_rms": round(jerk_total_rms, 1),
            "joints_cmd_deg": [round(float(d), 1) for d in q_cmd_deg],
            "joints_fb_deg": [round(float(d), 1) for d in q_fb_deg],
            "orientation": k_fb["orientation"],
            "joints_3d": joints_3d,
            "target_cm": target_cm,
            "ee_pos_cm": [round(float(p_ee[0] * 100), 2), round(float(p_ee[1] * 100), 2), round(float(p_ee[2] * 100), 2)],
        }
