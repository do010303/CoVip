# -*- coding: utf-8 -*-
"""
Unit test cho grasp_script.

Kiểm tra:
- State transitions cơ bản
- CLOSE → VERIFY flow (kẹp đúng, kẹp sai, retry)
- VERIFY window logic
- HOLD duration
"""

import numpy as np
import pytest

from assarm_common.grasp_script import GraspScript, GraspState


class TestGraspScript:

    def _make_script(self, **kwargs):
        return GraspScript(dt=0.1, **kwargs)

    def test_initial_state(self):
        gs = self._make_script()
        gs.reset(np.zeros(4))
        assert gs.state == GraspState.APPROACH

    def test_handoff_triggers_close(self):
        gs = self._make_script()
        gs.reset(np.zeros(4))
        gs.trigger_handoff(np.array([0.1, 0.2, -0.1, 0.0]))
        assert gs.state == GraspState.CLOSE

    def test_close_detects_stall(self):
        """Gripper bị chặn → chuyển VERIFY."""
        gs = self._make_script(stall_count_needed=3)
        gs.reset(np.zeros(4))
        gs.trigger_handoff(np.zeros(4))

        # Simulate: gripper_target tăng nhưng feedback đứng yên ở 0.9 rad
        for _ in range(20):
            cmd = gs.step(np.zeros(4), 0.9)
            if gs.state == GraspState.VERIFY:
                break

        assert gs.state == GraspState.VERIFY

    def test_verify_good_grip_transitions_to_hold(self):
        """Góc chặn trong cửa sổ → HOLD (dây đứt, giữ nguyên vị trí)."""
        gs = self._make_script()
        gs.reset(np.zeros(4))
        gs.state = GraspState.VERIFY
        gs._grip_stall_angle = 0.896  # q* — trong cửa sổ [0.82, 0.97]
        gs._step_in_state = 0

        # Chờ 2 bước ổn định
        gs.step(np.zeros(4), 0.896)
        gs.step(np.zeros(4), 0.896)

        assert gs.state == GraspState.HOLD

    def test_verify_bad_grip_retries(self):
        """Góc chặn ngoài cửa sổ → retry APPROACH."""
        gs = self._make_script()
        gs.reset(np.zeros(4))
        gs.state = GraspState.VERIFY
        gs._grip_stall_angle = 0.487  # kẹp chéo — ngoài cửa sổ
        gs._step_in_state = 0

        gs.step(np.zeros(4), 0.487)
        gs.step(np.zeros(4), 0.487)

        assert gs.state == GraspState.APPROACH
        assert gs.retry_count == 1

    def test_verify_exhausted_retries_fails(self):
        """Hết lượt retry → FAIL."""
        gs = self._make_script()
        gs.reset(np.zeros(4))
        gs.retry_count = 2  # max retries
        gs.state = GraspState.VERIFY
        gs._grip_stall_angle = 0.3  # ngoài cửa sổ
        gs._step_in_state = 0

        gs.step(np.zeros(4), 0.3)
        gs.step(np.zeros(4), 0.3)

        assert gs.state == GraspState.FAIL

    def test_hold_completes_after_duration(self):
        """HOLD → RELEASE sau hold_duration."""
        gs = self._make_script(hold_duration=0.5)
        gs.reset(np.zeros(4))
        gs.state = GraspState.HOLD
        gs._gripper_target = 0.896

        for _ in range(10):  # 10 × 0.1s = 1.0s > 0.5s
            cmd = gs.step(np.zeros(4), 0.896)
            if gs.state == GraspState.RELEASE:
                break

        assert gs.state == GraspState.RELEASE

    def test_is_terminal(self):
        gs = self._make_script()
        gs.reset(np.zeros(4))
        gs.state = GraspState.DONE
        assert gs.is_terminal
        assert gs.success

        gs.state = GraspState.FAIL
        assert gs.is_terminal
        assert not gs.success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
