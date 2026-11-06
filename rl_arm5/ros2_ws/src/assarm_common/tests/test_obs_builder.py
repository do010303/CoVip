# -*- coding: utf-8 -*-
"""
Unit test cho obs_builder.

Kiểm tra:
- Output shape đúng 28
- Không có NaN/Inf
- Chuẩn hóa nằm trong phạm vi hợp lý
"""

import numpy as np
import pytest

from assarm_common.obs_builder import build_obs, OBS_DIM


class TestObsBuilder:

    def _make_default_inputs(self):
        return dict(
            q_arm=np.array([0.1, -0.2, 0.3, 0.0]),
            q_prev=np.array([0.09, -0.21, 0.29, 0.01]),
            target=np.array([0.12, -0.18, 0.32, 0.0]),
            p_close=np.array([0.05, 0.1, 0.2]),
            obj_pos=np.array([0.06, 0.11, 0.19]),
            obj_vel_filtered=np.array([0.01, -0.02, 0.005]),
            R_gripper=np.eye(3),
            yaw_obj=0.1,
            yaw_jaw=0.05,
            tilt=0.15,
            pose_valid=True,
            dt=0.1,
        )

    def test_output_shape(self):
        obs = build_obs(**self._make_default_inputs())
        assert obs.shape == (OBS_DIM,), f"Shape {obs.shape}, kỳ vọng ({OBS_DIM},)"

    def test_output_dtype(self):
        obs = build_obs(**self._make_default_inputs())
        assert obs.dtype == np.float32

    def test_no_nan_inf(self):
        obs = build_obs(**self._make_default_inputs())
        assert not np.any(np.isnan(obs)), "Có NaN trong obs"
        assert not np.any(np.isinf(obs)), "Có Inf trong obs"

    def test_pose_valid_flag(self):
        inputs = self._make_default_inputs()
        inputs['pose_valid'] = True
        obs = build_obs(**inputs)
        assert obs[27] == 1.0

        inputs['pose_valid'] = False
        obs = build_obs(**inputs)
        assert obs[27] == 0.0

    def test_zero_inputs(self):
        """Obs tại home với khối ở gốc không crash."""
        obs = build_obs(
            q_arm=np.zeros(4),
            q_prev=np.zeros(4),
            target=np.zeros(4),
            p_close=np.zeros(3),
            obj_pos=np.zeros(3),
            obj_vel_filtered=np.zeros(3),
            R_gripper=np.eye(3),
            yaw_obj=0.0,
            yaw_jaw=0.0,
            tilt=0.0,
            pose_valid=True,
        )
        assert obs.shape == (OBS_DIM,)
        assert not np.any(np.isnan(obs))

    def test_velocity_clipping(self):
        """Vận tốc khớp rất lớn phải clip ±1."""
        inputs = self._make_default_inputs()
        inputs['q_arm'] = np.array([1.0, 0.0, 0.0, 0.0])
        inputs['q_prev'] = np.array([-1.0, 0.0, 0.0, 0.0])  # Δq = 2.0 trong 0.1s = 20 rad/s
        obs = build_obs(**inputs)
        # obs[4] = clip(20/2, -1, 1) = 1.0
        assert abs(obs[4]) <= 1.0 + 1e-7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
