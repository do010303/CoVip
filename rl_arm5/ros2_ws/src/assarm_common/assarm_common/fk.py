# -*- coding: utf-8 -*-
"""
assarm_common.fk — Forward Kinematics cho cánh tay ASSARM 4-DOF + gripper.

Chuỗi khớp (từ URDF assarm.xacro):
  world → base_link → Rigid_1 → Servo1_1
  → J1 (yaw, axis=[0,0,-1])  → Servo1_Pinion_Gear_1
  → Rigid_2 → Servo2_1
  → J2 (pitch, axis=[-1,0,0]) → Servo2_BracketU_1
  → Rigid_5 → Link1_1
  → Rigid_6 → Servo3_BracketM_1
  → Rigid_7 → Servo3_1
  → J3 (pitch, axis=[-1,0,0]) → Servo3_Pinion_Gear_1
  → Rigid_8 → Servo3_BracketU_1
  → Rigid_9 → Link2_1
  → Rigid_10 → Servo4_1
  → J4 (yaw, axis=[0,0,-1])  → Servo4_Pinion_Gear_1
  → Rigid_11 → Plate_1 (khung gripper)

Tất cả offset tĩnh lấy từ <origin xyz="..." rpy="..."> trong URDF.

LƯU Ý: Các giá trị offset ở đây trích thủ công từ URDF.
Cần unit test so sánh với TF Gazebo để xác nhận < 2 mm (Phase 0 bước 0.6).
"""

from typing import Tuple

import numpy as np


# ===========================================================================
# Ma trận quay cơ bản
# ===========================================================================

def _Rz(theta: float) -> np.ndarray:
    """Quay quanh trục z."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _Rx(theta: float) -> np.ndarray:
    """Quay quanh trục x."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Tạo ma trận 4x4 từ R 3x3 và t 3."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _ht_from_xyz(x: float, y: float, z: float) -> np.ndarray:
    """Ma trận dịch thuần."""
    return _homogeneous(np.eye(3), np.array([x, y, z]))


# ===========================================================================
# Offset tĩnh từ URDF (mỗi joint <origin xyz="...">)
# ===========================================================================

# base_link → Servo1_1 (Rigid_1)
_T_base_to_servo1 = _ht_from_xyz(0.003894, 0.02389, 0.56)

# Servo1_1 → J1 joint origin
_T_servo1_to_j1 = _ht_from_xyz(-0.0039, -0.034687, 0.020797)

# J1 child (Servo1_Pinion_Gear_1) → Servo2_1 (Rigid_2)
_T_j1_to_servo2 = _ht_from_xyz(0.0, 0.0, -0.037247)

# Servo2_1 → J2 joint origin
_T_servo2_to_j2 = _ht_from_xyz(-0.0262, 0.0, -0.03702)

# J2 child (Servo2_BracketU_1) → Link1_1 (Rigid_5)
_T_j2_to_link1 = _ht_from_xyz(0.0258, 0.0, -0.027)

# Link1_1 → Servo3_BracketM_1 (Rigid_6)
_T_link1_to_bracket3 = _ht_from_xyz(-0.005, 0.024987, -0.1)

# Servo3_BracketM_1 → Servo3_1 (Rigid_7)
_T_bracket3_to_servo3 = _ht_from_xyz(-0.008, -0.00075, -0.00987)

# Servo3_1 → J3 joint origin
_T_servo3_to_j3 = _ht_from_xyz(-0.0155, -0.03415, -0.005)

# J3 child (Servo3_Pinion_Gear_1) → Servo3_BracketU_1 (Rigid_8)
_T_j3_to_bracket3u = _ht_from_xyz(-0.002, 0.0, 0.0)

# Servo3_BracketU_1 → Link2_1 (Rigid_9)
_T_bracket3u_to_link2 = _ht_from_xyz(0.026, 0.0, -0.05)

# Link2_1 → Servo4_1 (Rigid_10)
_T_link2_to_servo4 = _ht_from_xyz(-0.005, 0.02415, -0.1)

# Servo4_1 → J4 joint origin
_T_servo4_to_j4 = _ht_from_xyz(0.005, -0.03415, -0.0155)

# J4 child (Servo4_Pinion_Gear_1) → Plate_1 (Rigid_11)
_T_j4_to_plate = _ht_from_xyz(0.0, 0.0, -0.002)

# Các offset tĩnh gộp (pre-compute)
# base → J1 origin
_T_pre_j1 = _T_base_to_servo1 @ _T_servo1_to_j1

# J1 child → J2 origin (qua Rigid_2 + Servo2_1)
_T_j1_to_j2 = _T_j1_to_servo2 @ _T_servo2_to_j2

# J2 child → J3 origin (qua Rigid_5 + Link1 + Rigid_6 + BracketM + Rigid_7 + Servo3)
_T_j2_to_j3 = _T_j2_to_link1 @ _T_link1_to_bracket3 @ _T_bracket3_to_servo3 @ _T_servo3_to_j3

# J3 child → J4 origin (qua Rigid_8 + BracketU + Rigid_9 + Link2 + Rigid_10 + Servo4)
_T_j3_to_j4 = _T_j3_to_bracket3u @ _T_bracket3u_to_link2 @ _T_link2_to_servo4 @ _T_servo4_to_j4

# J4 child → Plate_1 (Rigid_11)
_T_j4_to_plate_const = _T_j4_to_plate


# ===========================================================================
# Public API
# ===========================================================================

def fk_arm(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics cho 4 khớp tay.

    Parameters
    ----------
    q : ndarray(4)
        Góc 4 khớp [J1, J2, J3, J4] (rad).

    Returns
    -------
    position : ndarray(3)
        Vị trí tâm Plate_1 trong hệ tọa độ base (m).
    R : ndarray(3, 3)
        Ma trận quay của Plate_1 trong hệ base.
    """
    # J1: quay quanh axis=[0,0,-1] → Rz(-q1)
    R_j1 = _Rz(-q[0])
    T_j1 = _homogeneous(R_j1, np.zeros(3))

    # J2: quay quanh axis=[-1,0,0] → Rx(-q2) = Rx(-q2)
    R_j2 = _Rx(-q[1])
    T_j2 = _homogeneous(R_j2, np.zeros(3))

    # J3: quay quanh axis=[-1,0,0] → Rx(-q3)
    R_j3 = _Rx(-q[2])
    T_j3 = _homogeneous(R_j3, np.zeros(3))

    # J4: quay quanh axis=[0,0,-1] → Rz(-q4)
    R_j4 = _Rz(-q[3])
    T_j4 = _homogeneous(R_j4, np.zeros(3))

    # Chuỗi biến đổi: base → Plate_1
    T = (_T_pre_j1 @ T_j1
         @ _T_j1_to_j2 @ T_j2
         @ _T_j2_to_j3 @ T_j3
         @ _T_j3_to_j4 @ T_j4
         @ _T_j4_to_plate_const)

    return T[:3, 3].copy(), T[:3, :3].copy()


def p_close(q: np.ndarray) -> np.ndarray:
    """
    Tâm gắp khi đóng gripper.

    P_close = position(Plate_1) + R(Plate_1) · [0, 0, -0.0312]
    Đây là vị trí vật sẽ nằm khi gripper đóng, loại bỏ sai số Δz 3.12 cm.

    Parameters
    ----------
    q : ndarray(4)
        Góc 4 khớp tay.

    Returns
    -------
    ndarray(3)
        Tọa độ P_close trong hệ base (m).
    """
    from assarm_common.config import GRIP_CLOSE_OFFSET
    pos, R = fk_arm(q)
    return pos + R @ GRIP_CLOSE_OFFSET


def gripper_yaw(q: np.ndarray) -> float:
    """
    Yaw của trục đóng ngón (trục x của Plate_1) — FK đầy đủ.

    Returns
    -------
    float
        Góc yaw (rad) của trục x của Plate_1 chiếu lên mặt phẳng ngang.
    """
    _, R = fk_arm(q)
    # Trục x của Plate_1 trong hệ base
    x_axis = R[:, 0]
    return np.arctan2(x_axis[1], x_axis[0])


def gripper_tilt(q: np.ndarray) -> float:
    """
    Độ nghiêng gripper so với phương thẳng đứng.

    At home (q=0), R là ma trận đơn vị, z_axis = [0, 0, 1] trùng trục thẳng đứng +z base.

    Returns
    -------
    float
        Góc nghiêng (rad), 0 = thẳng đứng.
    """
    _, R = fk_arm(q)
    # Trục z của Plate_1 trong hệ base
    z_axis = R[:, 2]
    # Trục thẳng đứng +z
    up = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(z_axis, up), -1.0, 1.0)
    return np.arccos(cos_angle)


def wrap_to_pm45(angle: float) -> float:
    """
    Gập góc vào [-π/4, π/4] nhờ đối xứng 90° của khối lập phương.
    """
    # Gập vào [-π, π] trước
    a = (angle + np.pi) % (2 * np.pi) - np.pi
    # Gập vào [-π/4, π/4] (đối xứng 90°)
    a = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return a


def jacobian_numerical(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Jacobian số cho p_close.

    Returns
    -------
    ndarray(3, 4)
        dp_close / dq.
    """
    J = np.zeros((3, 4))
    p0 = p_close(q)
    for i in range(4):
        q_plus = q.copy()
        q_plus[i] += eps
        J[:, i] = (p_close(q_plus) - p0) / eps
    return J
