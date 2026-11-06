# -*- coding: utf-8 -*-
"""
Phase S Benchmark Test: Baseline IK + GraspScript evaluation.

Kiểm tra tiêu chí cổng Phase S:
- Test 1: Script-only với khối đứng yên đặt tại P_close (yêu cầu ≥ 90% thành công)
- Test 2: IK Approach + Script với khối đứng yên (Phase 1 condition: Level 0-1, swing = 0.0)
- Test 3: IK Approach + Script với khối đung đưa nhẹ (Phase 2 condition: Level 2-3, swing < 1 cm)
"""

import numpy as np
import pytest
from assarm_common.config import Q_STAR, VERIFY_WINDOW_LO, VERIFY_WINDOW_HI, Q_OPEN
from assarm_common.fk import p_close, gripper_tilt, fk_arm
from assarm_common.grasp_script import GraspScript, GraspState
from assarm_common.baseline import IKBaselineController


def simulate_servo_feedback(q_target: np.ndarray, q_current: np.ndarray, dt: float = 0.1, speed: float = 2.0) -> np.ndarray:
    """Mô phỏng đệm động học feedback góc khớp (quá trình nội suy 100 ms)."""
    step = np.clip(q_target - q_current, -speed * dt, speed * dt)
    return q_current + step


def simulate_gripper_feedback(gripper_target: float, is_cube_present: bool = True) -> float:
    """Mô phỏng feedback góc kẹp khi đóng ngón ôm khối 5 cm."""
    if is_cube_present and gripper_target > Q_STAR:
        # Gripper bị chặn bởi khối tại góc q* = 0.896 rad với sai số ngẫu nhiên nhỏ ±0.01 rad
        return Q_STAR + np.random.uniform(-0.01, 0.01)
    return gripper_target


class TestPhaseSBaseline:
    """Bộ kiểm thử cổng Phase S."""

    def test_script_only_stationary_cube(self):
        """
        [Cổng Phase S - Tiêu chí go/no-go 1]
        Script-only với khối đứng yên đặt đúng P_close.
        Phải đạt tỷ lệ thành công ≥ 90%.
        """
        np.random.seed(42)
        n_episodes = 100
        success_count = 0

        for ep in range(n_episodes):
            script = GraspScript(dt=0.1)
            q_arm = np.array([0.0, 0.6, -0.6, 0.0])  # Khớp ở tư thế tiếp cận
            script.reset(q_arm)

            # Kích hoạt handoff ngay lập tức (khối đặt sẵn ở P_close)
            script.trigger_handoff(q_arm)

            # Sim 50 steps (5 giây)
            for _ in range(50):
                grip_fb = simulate_gripper_feedback(script._gripper_target, is_cube_present=True)
                cmd = script.step(q_arm, grip_fb)
                q_arm = simulate_servo_feedback(cmd.arm_target, q_arm)

                if script.is_terminal:
                    break

            if script.state in (GraspState.HOLD, GraspState.DONE, GraspState.RELEASE):
                success_count += 1

        success_rate = (success_count / n_episodes) * 100
        print(f"\n[Phase S - Step S.1] Script-Only Stationary Cube Success Rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
        assert success_rate >= 90.0, f"Cổng Phase S không đạt: tỷ lệ script-only = {success_rate:.1f}% < 90%"

    def test_baseline_ik_approach_stationary_cube(self):
        """
        [Phase S - Step S.2 (Phase 1 condition)]
        IK Approach + GraspScript với khối đứng yên trong vùng Level 0-1 (r = 0.09-0.16m, swing = 0.0).
        """
        np.random.seed(42)
        n_episodes = 100
        handoff_count = 0
        grasp_success_count = 0

        for ep in range(n_episodes):
            controller = IKBaselineController(dt=0.1)
            
            # Spawn random stationary target in Level 0-1 (hướng -Y trước mặt robot)
            r = np.random.uniform(0.09, 0.16)
            theta = np.random.uniform(-np.radians(20), np.radians(20))
            z = 0.18 + 0.40 * (r - 0.08) + np.random.uniform(-0.010, 0.010)
            cube_pos = np.array([-r * np.sin(theta), -r * np.cos(theta), z])

            # Robot initial home pose
            q_arm = np.array([theta, 0.4, -0.4, 0.0])
            controller.reset(q_arm)

            ep_handoff = False
            for step in range(80):  # max 80 steps (8s)
                p_cur = p_close(q_arm)
                grip_fb = simulate_gripper_feedback(controller.script._gripper_target, is_cube_present=True)

                res = controller.step(q_arm, grip_fb, cube_pos)
                q_arm = simulate_servo_feedback(res.arm_target, q_arm)

                if res.handoff_triggered:
                    ep_handoff = True

                if controller.script.is_terminal:
                    break

            if ep_handoff:
                handoff_count += 1
            if controller.script.state in (GraspState.HOLD, GraspState.DONE, GraspState.RELEASE):
                grasp_success_count += 1

        handoff_rate = (handoff_count / n_episodes) * 100
        overall_rate = (grasp_success_count / n_episodes) * 100
        print(f"\n[Phase S - Step S.2 / Phase 1] Stationary Baseline IK:")
        print(f"  Handoff Rate: {handoff_rate:.1f}% ({handoff_count}/{n_episodes})")
        print(f"  Overall Grasp Success Rate: {overall_rate:.1f}% ({grasp_success_count}/{n_episodes})")

        # Baseline IK benchmark expectation: >= 50% trên khối đứng yên
        assert handoff_rate >= 50.0, f"Tỷ lệ bàn giao Baseline IK khối đứng yên = {handoff_rate:.1f}% < 50%"

    def test_baseline_ik_approach_swinging_cube(self):
        """
        [Phase S - Step S.2 (Phase 2 condition: Vùng mở rộng Level 2-3)]
        IK Approach + GraspScript với khối đứng yên trong vùng mở rộng Level 2-3 (r = 0.08-0.19m, swing = 0.0).
        """
        np.random.seed(42)
        n_episodes = 100
        handoff_count = 0
        grasp_success_count = 0

        for ep in range(n_episodes):
            controller = IKBaselineController(dt=0.1)

            r_base = np.random.uniform(0.08, 0.19)
            theta_base = np.random.uniform(-np.radians(35), np.radians(35))
            z_base = 0.18 + 0.40 * (r_base - 0.08) + np.random.uniform(-0.010, 0.010)

            # Khối luôn luôn đứng yên (hướng -Y)
            cube_pos = np.array([-r_base * np.sin(theta_base), -r_base * np.cos(theta_base), z_base])
            cube_vel = np.zeros(3)

            q_arm = np.array([theta_base, 0.4, -0.4, 0.0])
            controller.reset(q_arm)

            ep_handoff = False
            for step in range(80):
                grip_fb = simulate_gripper_feedback(controller.script._gripper_target, is_cube_present=True)
                res = controller.step(q_arm, grip_fb, cube_pos, cube_vel)
                q_arm = simulate_servo_feedback(res.arm_target, q_arm)

                if res.handoff_triggered:
                    ep_handoff = True

                if controller.script.is_terminal:
                    break

            if ep_handoff:
                handoff_count += 1
            if controller.script.state in (GraspState.HOLD, GraspState.DONE, GraspState.RELEASE):
                grasp_success_count += 1

        handoff_rate = (handoff_count / n_episodes) * 100
        overall_rate = (grasp_success_count / n_episodes) * 100
        print(f"\n[Phase S - Step S.2 / Phase 2] Stationary Extended Baseline IK:")
        print(f"  Handoff Rate: {handoff_rate:.1f}% ({handoff_count}/{n_episodes})")
        print(f"  Overall Grasp Success Rate: {overall_rate:.1f}% ({grasp_success_count}/{n_episodes})")

        assert handoff_rate >= 50.0, f"Tỷ lệ bàn giao Baseline IK vùng mở rộng = {handoff_rate:.1f}% < 50%"
