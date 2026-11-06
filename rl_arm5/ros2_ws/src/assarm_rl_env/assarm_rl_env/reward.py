# -*- coding: utf-8 -*-
"""
assarm_rl_env.reward — Hàm reward dense cho giai đoạn APPROACH.

Dùng thông tin đặc quyền của sim (pose thật, contact) cho reward.
Policy chỉ thấy observation (perception có nhiễu).

Xem rl_recommendation_v5_0.md mục 3.5.
"""

import numpy as np


def compute_reward(
    d: float,                    # khoảng cách P_close → khối (m)
    dpsi: float,                 # yaw tương đối đã wrap (rad)
    tilt_deg: float,             # độ nghiêng gripper (°)
    rel_speed: float,            # tốc độ tương đối khối−gripper (m/s)
    jaw_contact: bool,           # ngón mở chạm khối
    action: np.ndarray,          # (4,) action hiện tại
    prev_action: np.ndarray,     # (4,) action trước
    cavity_align: float = 1.0,   # Độ thẳng hàng vector miệng ngàm (-Z plate) và vector tới tâm khối [-1..1]
    dz_err: float = 0.0,         # Sai số độ cao |P_close,z - Z_cube| (m)
    v_match: float = 0.0,        # Độ đồng pha vận tốc giữa đầu gắp và khối (m^2/s^2)
    q_j4: float = 0.0,           # Góc khớp cổ tay J4 (rad)
    W_NEAR: float = 0.3,
    W_ALIGN: float = 0.6,        # Phạt lệch hướng miệng ngàm khi tiếp cận khối
    W_Z: float = 0.5,            # Phạt lệch cao độ Z (ép ngàm hạ đúng thân trụ)
    W_TILT: float = 0.25,        # [QUAN TRỌNG] Phạt ngửa/chúc ngàm kẹp, ép ngàm nằm ngang (tilt ~ 90°)
    W_J4: float = 0.02,          # [QUAN TRỌNG] Phạt vặn xoắn cổ tay J4 quá mức, giữ thế tay tự nhiên
    W_SPEED: float = 0.15,       # Phạt tốc độ tương đối khi gần
    W_CONTACT: float = 0.3,
    W_ACTION: float = 0.005,
    w_rate: float = 0.005,
    time_penalty: float = 0.02,  # Phạt thời gian để thúc đẩy kết thúc sớm
) -> float:
    """
    Tính reward dense thuần âm (<= 0) mỗi step cho bài toán gắp mặt tròn khối trụ:
    - -(d + 0.3 * tanh(d/0.03)): khoảng cách tiệm cận 0 khi d -> 0
    - Phạt cavity_align khi d < 0.15m: ép miệng ngàm kẹp phải há hướng thẳng vào khối trụ
    - Phạt dz_err khi d < 0.12m: ép ngàm hạ đúng cao độ thân trụ
    - Phạt TILT: ép ngàm kẹp nằm ngang vuông góc thân trụ (chống ngửa/chĩa xuống đất)
    - Phạt J4: ép cổ tay ở vị trí trung tính, không vặn xoắn kỳ dị
    - Thưởng velocity matching: khi d < 0.08m, gripper di chuyển đồng pha vận tốc với khối trụ
    - Phạt tốc độ tương đối, contact, action regularization
    """
    # Hệ số "gần" cho speed/contact/v_match: tăng từ 0 → 1 khi d giảm từ 0.08 → 0
    w_near = np.clip(1.0 - d / 0.08, 0.0, 1.0)
    # Hệ số hướng miệng ngàm: kích hoạt từ xa khi d < 0.15m để dẫn hướng sớm
    w_near_align = np.clip(1.0 - d / 0.15, 0.0, 1.0)
    # Hệ số độ cao Z: kích hoạt khi d < 0.12m
    w_near_z = np.clip(1.0 - d / 0.12, 0.0, 1.0)

    # Phạt lệch hướng miệng kẹp: cavity_align = 1.0 -> phạt = 0; cavity_align <= 0 -> phạt nặng
    align_penalty = W_ALIGN * w_near_align * max(0.0, 1.0 - cavity_align)

    # Phạt lệch cao độ Z: dung sai thân trụ ±20mm
    z_penalty = W_Z * w_near_z * max(0.0, dz_err - 0.020) / 0.030

    # Phạt góc nghiêng Tilt: ngàm kẹp phải hướng ngang (tilt_deg quanh 90° ± 15°)
    tilt_penalty = W_TILT * (max(0.0, abs(tilt_deg - 90.0) - 15.0) / 15.0)

    # Phạt vặn xoắn cổ tay J4: giữ J4 quanh 0.0 để cử động cánh tay tự nhiên
    j4_penalty = W_J4 * (abs(q_j4) / (np.pi / 2))

    # Thưởng đồng pha vận tốc đón đầu (Velocity Matching): tối đa +0.15 khi d < 0.08m
    vel_matching_bonus = 0.15 * w_near * float(np.clip(v_match / 0.00025, -1.0, 1.0))

    r = (
        # Distance shaping — thuần âm, tiệm cận 0 khi d -> 0
        - (d + W_NEAR * np.tanh(d / 0.03))
        # Phạt hướng mở ngàm — ép miệng ngàm chĩa thẳng vào khối trụ để nuốt trọn khối
        - align_penalty
        # Phạt cao độ Z — ép ngàm hạ ngang thân khối trụ
        - z_penalty
        # Phạt tư thế — giữ ngàm kẹp nằm ngang và cổ tay không bị vặn xoắn
        - tilt_penalty
        - j4_penalty
        # Phạt thời gian mỗi step — thúc đẩy hoàn thành càng sớm càng tốt
        - time_penalty
        # Speed penalty — phạt tốc độ tương đối giữa gripper và khối khi gần
        - W_SPEED * w_near * min(rel_speed / 0.05, 1.0)
        # Velocity matching bonus — thưởng bám cùng hướng vận tốc đón đầu
        + vel_matching_bonus
        # Contact penalty — ngón mở chạm khối (đẩy khối đi)
        - W_CONTACT * float(jaw_contact)
        # Action regularization
        - W_ACTION * np.sum(action**2)
        # Action rate penalty — giảm lắc giật
        - w_rate * np.sum((action - prev_action)**2)
    )

    return float(r)


def compute_terminal_reward(
    handoff_quality_good: bool,
) -> float:
    """
    Thưởng cuối khi bàn giao.

    handoff_quality_good: True nếu pose thật thỏa mọi ngưỡng.
    Nếu False, script kích hoạt bàn giao nhầm do nhiễu perception.

    Returns
    -------
    float
        +30.0 nếu tốt, -2.0 nếu bàn giao nhầm.
    """
    if handoff_quality_good:
        return 30.0
    else:
        return -2.0
