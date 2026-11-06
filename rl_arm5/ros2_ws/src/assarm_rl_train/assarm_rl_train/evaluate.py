# -*- coding: utf-8 -*-
"""
assarm_rl_train.evaluate — Script đánh giá mô hình SAC chuẩn mực khoa học (Phase 4 / Benchmark).

Tính năng:
- Đánh giá 100/200/N episode độc lập với seed cố định
- Hỗ trợ cả môi trường tĩnh (Stage B) và môi trường động (Stage B+ Dynamic 1.2-2.0 cm/s)
- Tính toán khoảng tin cậy Wilson 95% CI (Wilson Score Interval)
- Thống kê chi tiết: Tỉ lệ bàn giao (Handoff Rate), Tỉ lệ gắp thành công (Overall Success),
  Số bước tiếp cận trung bình (Steps to handoff), Sai số căn chỉnh (ex, ey, ez, yaw).
"""

import argparse
import os
import sys
import numpy as np

# Đảm bảo đường dẫn import trong ros2_ws
current_dir = os.path.dirname(os.path.abspath(__file__))
ws_src = os.path.abspath(os.path.join(current_dir, "../../"))
for pkg in ["assarm_common", "assarm_rl_env", "assarm_rl_train"]:
    p = os.path.join(ws_src, pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from stable_baselines3 import SAC

from assarm_common.config import MAX_STEPS_STAGE_B
from assarm_common.grasp_script import GraspScript, GraspState
from assarm_rl_env.assarm_env import AssarmGraspEnv


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Tính khoảng tin cậy Wilson 95% CI cho tỉ lệ nhị thức.
    Returns: (tỉ_lệ_phần_trăm, cận_dưới_phần_trăm, cận_trên_phần_trăm)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    margin = (z / denominator) * np.sqrt((p * (1.0 - p) / trials) + (z**2 / (4.0 * trials**2)))
    low = max(0.0, (center - margin) * 100.0)
    high = min(100.0, (center + margin) * 100.0)
    return p * 100.0, low, high


def evaluate_policy(
    model_path: str,
    n_episodes: int = 200,
    level: int = 3,
    dynamic_motion: bool = False,
    motion_stage: int = 2,
    deterministic: bool = True,
    max_steps: int = MAX_STEPS_STAGE_B,
    seed_base: int = 1000,
    verbose: int = 1,
):
    mode_str = "Deterministic" if deterministic else "Stochastic"
    env_str = f"Dynamic Motion Stage {motion_stage} (1.2 - 2.0 cm/s + OU Wind)" if dynamic_motion else "Static"

    print("=" * 80)
    print(f" ĐÁNH GIÁ CHUẨN MỰC KHOA HỌC MÔ HÌNH SAC [{mode_str.upper()}]")
    print(f" - Mô hình: {model_path}")
    print(f" - Môi trường: {env_str} | Spawn Level: {level}")
    print(f" - Số Episodes: {n_episodes} | Max steps/ep: {max_steps} ({(max_steps*0.1):.1f}s)")
    print("=" * 80)

    env = AssarmGraspEnv(
        spawn_level=level,
        max_steps=max_steps,
        dynamic_motion=dynamic_motion,
        motion_stage=motion_stage,
    )
    model = SAC.load(model_path, env=env)

    handoff_count = 0
    good_handoff_count = 0
    grasp_success_count = 0
    false_handoff_count = 0

    ep_rewards = []
    end_distances = []
    handoff_steps = []
    errors_ex = []
    errors_ey = []
    errors_ez = []
    errors_yaw_deg = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep)
        script = GraspScript(dt=0.1, stall_threshold=0.12)
        script.reset(env.sim.q_meas)

        ep_reward = 0.0
        handoff_trig = False
        in_script_phase = False
        step_at_handoff = None

        for step in range(env.max_steps):
            if in_script_phase:
                grip_fb = 0.896 if script._gripper_target > 0.896 else script._gripper_target
                cmd = script.step(env.sim.q_meas, grip_fb)
                obs, reward, terminated, truncated, info = env.step((cmd.arm_target - env.sim.q_meas) / 0.08)
                if script.is_terminal:
                    break
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                if info.get("handoff_ok", False):
                    handoff_trig = True
                    in_script_phase = True
                    step_at_handoff = step + 1
                    script.trigger_handoff(env.sim.q_meas)

                    # Ghi nhận sai số căn chỉnh tại thời điểm bàn giao
                    if "e_grip" in info:
                        errors_ex.append(abs(info["e_grip"][0]) * 1000.0)
                        errors_ey.append(abs(info["e_grip"][1]) * 1000.0)
                        errors_ez.append(abs(info["e_grip"][2]) * 1000.0)
                    if "dpsi" in info:
                        errors_yaw_deg.append(abs(np.degrees(info["dpsi"])))

                if terminated or truncated:
                    break

        end_distances.append(info.get("distance", 0.0))
        ep_rewards.append(ep_reward)

        if handoff_trig:
            handoff_count += 1
            handoff_steps.append(step_at_handoff)
            if info.get("good_handoff", True):
                good_handoff_count += 1
            else:
                false_handoff_count += 1

        if script.state in (GraspState.HOLD, GraspState.DONE, GraspState.RELEASE) or (handoff_trig and not script.is_terminal):
            grasp_success_count += 1

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f" -> Đã đánh giá {ep + 1}/{n_episodes} episodes | Handoff hiện tại: {handoff_count}/{ep + 1} ({(handoff_count/(ep+1)*100):.1f}%)")

    # Tính toán chỉ số thống kê & khoảng tin cậy Wilson 95%
    h_rate, h_low, h_high = wilson_score_interval(handoff_count, n_episodes)
    g_rate, g_low, g_high = wilson_score_interval(good_handoff_count, n_episodes)
    s_rate, s_low, s_high = wilson_score_interval(grasp_success_count, n_episodes)
    false_rate = (false_handoff_count / max(handoff_count, 1)) * 100.0

    mean_steps = float(np.mean(handoff_steps)) if handoff_steps else float(max_steps)
    std_steps = float(np.std(handoff_steps)) if handoff_steps else 0.0
    median_steps = float(np.median(handoff_steps)) if handoff_steps else float(max_steps)

    mean_dist_mm = float(np.mean(end_distances) * 1000.0)
    mean_rew = float(np.mean(ep_rewards))
    std_rew = float(np.std(ep_rewards))

    print("\n" + "=" * 80)
    print(f" BÁO CÁO THỐNG KÊ KHOA HỌC CHUẨN MỰC [{mode_str.upper()}] ({n_episodes} Episodes)")
    print("=" * 80)
    print(f"  • Handoff Rate (Tỉ lệ bàn giao):       {h_rate:6.2f}%  [95% CI: {h_low:.1f}% — {h_high:.1f}%] ({handoff_count}/{n_episodes})")
    print(f"  • Good Handoff Rate (Đạt chuẩn gắp):    {g_rate:6.2f}%  [95% CI: {g_low:.1f}% — {g_high:.1f}%] ({good_handoff_count}/{n_episodes})")
    print(f"  • Overall Success (Gắp & Giữ vững):    {s_rate:6.2f}%  [95% CI: {s_low:.1f}% — {s_high:.1f}%] ({grasp_success_count}/{n_episodes})")
    print(f"  • False Handoff Rate (Bàn giao lỗi):   {false_rate:6.2f}%  ({false_handoff_count}/{max(handoff_count, 1)})")
    print("-" * 80)
    print(f"  • Thời gian tiếp cận (Steps to Handoff): {mean_steps:4.1f} ± {std_steps:3.1f} steps (Median: {median_steps:.1f} steps ~ {(median_steps*0.1):.2f}s)")
    if errors_ex:
        print(f"  • Sai số căn chỉnh ngàm kẹp tại Handoff:")
        print(f"    - Trục đóng ngàm |e_x|:  {np.mean(errors_ex):4.2f} ± {np.std(errors_ex):4.2f} mm (Ngưỡng cho phép: 12.0 mm)")
        print(f"    - Dọc thân ngón  |e_y|:  {np.mean(errors_ey):4.2f} ± {np.std(errors_ey):4.2f} mm (Ngưỡng cho phép: 18.0 mm)")
        print(f"    - Chiều cao thân |e_z|:  {np.mean(errors_ez):4.2f} ± {np.std(errors_ez):4.2f} mm (Ngưỡng cho phép: 25.0 mm)")
    print(f"  • Khoảng cách tâm cuối (End Distance):  {mean_dist_mm:4.1f} mm")
    print(f"  • Phần thưởng Episode (Mean Reward):    {mean_rew:5.2f} ± {std_rew:4.2f}")
    print("=" * 80 + "\n")

    return {
        "handoff_rate": h_rate,
        "handoff_ci": (h_low, h_high),
        "good_handoff_rate": g_rate,
        "overall_success": s_rate,
        "success_ci": (s_low, s_high),
        "false_handoff_rate": false_rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_rew,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình SAC chuẩn mực khoa học (Wilson 95% CI)")
    parser.add_argument("--model", type=str, default="./models/stage_b_plus_final.zip", help="Đường dẫn file mô hình .zip")
    parser.add_argument("--episodes", type=int, default=200, help="Số episode đánh giá (mặc định 200)")
    parser.add_argument("--level", type=int, default=3, help="Level spawn (0..3, mặc định 3: toàn dải)")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Bật môi trường động Stage B+ (1.2-2.0 cm/s)")
    parser.add_argument("--static", dest="dynamic", action="store_false", help="Chạy trên môi trường tĩnh")
    parser.add_argument("--motion_stage", type=int, default=2, help="1: Cố định tâm | 2: Ngẫu nhiên Level 3 + Nhiễu gió OU")
    parser.add_argument("--stochastic", action="store_true", help="Chạy đánh giá ở chế độ stochastic")
    parser.add_argument("--seed", type=int, default=42, help="Seed gốc khởi tạo")
    args = parser.parse_args()

    # Fallback model nếu không tìm thấy mặc định
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = "./ros2_ws/models/stage_b_plus_final.zip"
        if os.path.exists(fallback):
            model_path = fallback

    evaluate_policy(
        model_path=model_path,
        n_episodes=args.episodes,
        level=args.level,
        dynamic_motion=args.dynamic,
        motion_stage=args.motion_stage,
        deterministic=not args.stochastic,
        seed_base=args.seed,
    )


if __name__ == "__main__":
    main()
