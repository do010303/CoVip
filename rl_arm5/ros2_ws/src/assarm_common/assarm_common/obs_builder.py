# -*- coding: utf-8 -*-
"""
assarm_common.obs_builder — Xây observation 28D dùng chung sim/thật.

DUY NHẤT MỘT BẢN — được import bởi cả Gymnasium env (sim) và deploy node (Pi 4).
Hai bản sẽ lệch nhau và gây sim-to-real lỗi âm thầm.

Xem rl_recommendation_v5_0.md mục 3.3 cho chi tiết từng thành phần.
"""

import numpy as np

from assarm_common.config import (
    NORM_JOINT, NORM_VEL, NORM_POS, NORM_OBJ_VEL, NORM_ERR, NORM_TILT,
    MAX_DELTA,
)
from assarm_common.fk import wrap_to_pm45


OBS_DIM = 28


def build_obs(
    q_arm: np.ndarray,            # (4,) góc 4 khớp tay hiện tại
    q_prev: np.ndarray,           # (4,) góc 4 khớp tay step trước
    target: np.ndarray,           # (4,) target hiện tại
    p_close: np.ndarray,          # (3,) tâm gắp khi đóng (FK)
    obj_pos: np.ndarray,          # (3,) vị trí khối (perception)
    obj_vel_filtered: np.ndarray, # (3,) vận tốc khối đã lọc
    R_gripper: np.ndarray,        # (3,3) rotation matrix Plate_1
    yaw_obj: float,               # yaw khối (perception)
    yaw_jaw: float,               # yaw trục đóng ngón (FK đầy đủ)
    tilt: float,                  # độ nghiêng gripper (rad)
    pose_valid: bool,             # cờ perception
    dt: float = 0.1,
) -> np.ndarray:
    """
    Xây observation 28D chuẩn hóa thủ công.

    Thứ tự các thành phần:
      [0:4]   — Góc khớp / giới hạn
      [4:8]   — Vận tốc khớp (sai phân)
      [8:12]  — target − q
      [12:15] — P_close (tâm gắp)
      [15:18] — Vị trí khối
      [18:21] — Vận tốc khối đã lọc
      [21:24] — Sai số trong khung gripper [e_x, e_y, e_z]
      [24:26] — sin(4Δψ), cos(4Δψ)
      [26]    — Độ nghiêng gripper
      [27]    — pose_valid

    Returns
    -------
    ndarray(28, dtype=float32)
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # 1–4: Góc khớp chuẩn hóa bằng giới hạn
    obs[0:4] = q_arm / NORM_JOINT

    # 5–8: Vận tốc khớp (sai phân, clip ±1)
    obs[4:8] = np.clip((q_arm - q_prev) / dt / NORM_VEL, -1.0, 1.0)

    # 9–12: Sai lệch target − q (clip ±1)
    obs[8:12] = np.clip((target - q_arm) / MAX_DELTA, -1.0, 1.0)

    # 13–15: P_close (tâm gắp)
    obs[12:15] = p_close / NORM_POS

    # 16–18: Vị trí khối
    obs[15:18] = obj_pos / NORM_POS

    # 19–21: Vận tốc khối đã lọc (clip ±1)
    obs[18:21] = np.clip(obj_vel_filtered / NORM_OBJ_VEL, -1.0, 1.0)

    # 22–24: Sai số trong khung gripper: e = R_gripper^T · (obj − P_close)
    e = R_gripper.T @ (obj_pos - p_close)
    obs[21:24] = e / NORM_ERR

    # 25–26: Yaw tương đối (4Δψ vì khối đối xứng 90°)
    dpsi = wrap_to_pm45(yaw_obj - yaw_jaw)
    obs[24] = np.sin(4 * dpsi)
    obs[25] = np.cos(4 * dpsi)

    # 27: Độ nghiêng gripper
    obs[26] = tilt / NORM_TILT

    # 28: Cờ perception
    obs[27] = float(pose_valid)

    return obs
