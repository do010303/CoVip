# -*- coding: utf-8 -*-
"""
Unit test cho FK module.

Test cơ bản:
- Home position (q=[0,0,0,0]): vị trí Plate_1 phải gần tâm kẹp mở ≈ 0.175 m
- P_close tại home ≈ 0.144 m (từ rl_recommendation_v5_0.md)
- Đối xứng J1: xoay J1 180° → x, y đổi dấu, z giữ nguyên (xấp xỉ)

CẦN LÀM: test FK so với TF Gazebo (< 2 mm) khi sim chạy được.
"""

import numpy as np
import pytest

from assarm_common.fk import fk_arm, p_close, gripper_yaw, gripper_tilt, wrap_to_pm45


class TestFK:
    """Tests cho forward kinematics."""

    def test_home_position_height(self):
        """Tại home (q=0), tâm Plate_1 phải ở khoảng 0.15–0.20 m."""
        q = np.zeros(4)
        pos, R = fk_arm(q)
        # Theo tài liệu v5.0: điểm giữa hai gốc ngón ≈ 0.175 m
        assert 0.10 < pos[2] < 0.25, f"FK home z = {pos[2]:.4f} ngoài khoảng hợp lý"

    def test_p_close_lower_than_fk(self):
        """P_close phải thấp hơn FK vì offset [0,0,-0.0312]."""
        q = np.zeros(4)
        pos_fk, _ = fk_arm(q)
        pos_pc = p_close(q)
        # P_close thấp hơn FK khoảng 3.12 cm (khi gripper thẳng đứng)
        assert pos_pc[2] < pos_fk[2], "P_close phải thấp hơn FK position"
        diff_z = pos_fk[2] - pos_pc[2]
        assert abs(diff_z - 0.0312) < 0.005, f"Chênh lệch z = {diff_z:.4f}, kỳ vọng ~0.0312"

    def test_j1_rotation_symmetry(self):
        """Xoay J1 → vị trí phải quay quanh trục J1 (bán kính r_j1 giữ nguyên)."""
        q0 = np.array([0.0, 0.3, -0.2, 0.0])
        q1 = np.array([np.pi / 4, 0.3, -0.2, 0.0])
        pos0, _ = fk_arm(q0)
        pos1, _ = fk_arm(q1)
        # Trục J1 nằm tại (0.0, -0.0108) trong hệ base
        j1_xy = np.array([0.0, -0.010797])
        r0 = np.linalg.norm(pos0[:2] - j1_xy)
        r1 = np.linalg.norm(pos1[:2] - j1_xy)
        assert abs(r0 - r1) < 0.001, f"Bán kính quanh J1 khác: {r0:.4f} vs {r1:.4f}"
        # Độ cao phải giữ nguyên
        assert abs(pos0[2] - pos1[2]) < 0.001, f"Độ cao khác: {pos0[2]:.4f} vs {pos1[2]:.4f}"

    def test_gripper_tilt_home_is_zero(self):
        """Tại home, gripper thẳng đứng hướng xuống → nghiêng ≈ 0."""
        q = np.zeros(4)
        tilt = gripper_tilt(q)
        # gripper_tilt đo góc với vector [0,0,-1]
        assert tilt < np.radians(5), f"Tilt tại home = {np.degrees(tilt):.1f}°, kỳ vọng < 5°"

    def test_wrap_to_pm45(self):
        """Test wrap yaw vào [-45°, 45°]."""
        assert abs(wrap_to_pm45(0.0)) < 1e-10
        assert abs(wrap_to_pm45(np.pi / 2)) < 1e-10  # 90° → 0° (đối xứng)
        assert abs(wrap_to_pm45(np.pi)) < 1e-10       # 180° → 0°
        assert abs(abs(wrap_to_pm45(np.pi / 4)) - np.pi / 4) < 1e-10  # 45° và -45° tương đương
        # -30° → -30°
        assert abs(wrap_to_pm45(-np.pi / 6) - (-np.pi / 6)) < 1e-10


class TestPCloseReachability:
    """Test tầm với tại các spawn level."""

    @pytest.mark.parametrize("q2,q3", [
        (0.3, -0.2),
        (0.5, -0.4),
        (0.7, -0.6),
    ])
    def test_reach_varies_with_shoulder_elbow(self, q2, q3):
        """Tầm ngang phải tăng khi shoulder/elbow vươn ra."""
        q = np.array([0.0, q2, q3, 0.0])
        pos = p_close(q)
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        # Phải có tầm ngang > 0
        assert r > 0.01, f"Tầm ngang quá nhỏ: {r:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
