# -*- coding: utf-8 -*-
"""
assarm_rl_train.train_stage_b — Huấn luyện SAC Phase 2 Stage B (Full Workspace & DR).

- Kế thừa (Warm-start) từ mô hình hội tụ tốt nhất Stage A (models/best_model.zip).
- Mở rộng toàn dải không gian làm việc Level 2 & Level 3:
    + Bán kính r: [0.08, 0.19] m (từ cực cận đến cực viễn)
    + Góc quét đế J1: ±35° đến ±60° (hình quạt 120° trước mặt robot)
    + Góc xoay Yaw khối: ±25° đến ±45°
    + Độ cao z: [0.18, 0.26] m
    + Khối luôn đứng yên (swing = 0.0) theo cam kết thiết kế
- Hỗ trợ Curriculum Learning tự động:
    + Giai đoạn B1 (0 - 50.000 steps): Level 2
    + Giai đoạn B2 (50.000 - 150.000 steps): Level 3 toàn dải
- Đánh giá định kỳ trên Level 3 (Cổng C1a: handoff_rate >= 90%).
"""

import argparse
import os
import sys
import numpy as np

# Tự động nạp đường dẫn packages trong ros2_ws/src vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
ws_src = os.path.abspath(os.path.join(current_dir, "../../"))
for pkg in ["assarm_common", "assarm_rl_env", "assarm_rl_train"]:
    pkg_path = os.path.join(ws_src, pkg)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from assarm_common.config import MAX_STEPS_STAGE_B
from assarm_rl_env.assarm_env import AssarmGraspEnv
from assarm_rl_train.callbacks import HandoffEvalCallback


class CurriculumCallback(BaseCallback):
    """
    Callback chuyển cấp độ spawn theo tiến trình huấn luyện (Curriculum).
    Ví dụ: < 50k steps: Level 2 -> >= 50k steps: Level 3 toàn dải.
    """
    def __init__(self, train_env, switch_step: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.switch_step = switch_step
        self.switched = False

    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.switch_step:
            if hasattr(self.train_env, "spawn_level"):
                self.train_env.spawn_level = 3
            elif hasattr(self.train_env, "envs"):
                for e in self.train_env.envs:
                    e.spawn_level = 3
            self.switched = True
            if self.verbose > 0:
                print("\n" + "=" * 70)
                print(f"[CURRICULUM] >>> ĐẠT MỐC {self.num_timesteps} STEPS! NÂNG CẤP SPAWN LÊN LEVEL 3 TOÀN DẢI (±60°) <<<")
                print("=" * 70 + "\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện SAC Phase 2 Stage B (Full Workspace)")
    parser.add_argument("--warm_start", type=str, default=None, help="Đường dẫn checkpoint để warm-start (mặc định None - train mới từ đầu cho kẹp ngang)")
    parser.add_argument("--timesteps", type=int, default=150000, help="Tổng timesteps huấn luyện Stage B")
    parser.add_argument("--level", type=int, default=2, help="Cấp độ ban đầu (2 hoặc 3, mặc định 2)")
    parser.add_argument("--no_curriculum", action="store_true", help="Tắt curriculum, giữ cố định cấp độ chỉ định")
    parser.add_argument("--switch_step", type=int, default=50000, help="Số step chuyển từ Level 2 lên Level 3")
    parser.add_argument("--save_dir", type=str, default="./models/", help="Thư mục lưu mô hình")
    parser.add_argument("--tb_log", type=str, default="./tb_logs/", help="Thư mục TensorBoard")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.zip) để huấn luyện tiếp tục")
    args = parser.parse_args()

    # Tối ưu CPU 4 cores
    torch.set_num_threads(2)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 75)
    print(f" KHỞI CHẠY HUẤN LUYỆN SAC STAGE B (PHASE 2 - FULL WORKSPACE EXPANSION)")
    print(f" - Tổng timesteps: {args.timesteps}")
    print(f" - Max steps/episode: {MAX_STEPS_STAGE_B} (8.0 giây)")
    print(f" - Cấp độ khởi đầu: Level {args.level}")
    print(f" - Curriculum chuyển Level 3 tại step: {args.switch_step if not args.no_curriculum else 'TẮT'}")
    print(f" - Khối Cube: ĐỨNG YÊN (swing = 0.0), ngẫu nhiên hóa vị trí toàn dải")
    if args.resume:
        print(f" - RESUME từ: {args.resume}")
    elif args.warm_start and os.path.exists(args.warm_start):
        print(f" - WARM-START từ mô hình Stage A: {args.warm_start}")
    print("=" * 75)

    # Môi trường train & eval
    # Stage B max_steps = 80
    train_env = AssarmGraspEnv(spawn_level=args.level, max_steps=MAX_STEPS_STAGE_B)
    eval_env = AssarmGraspEnv(spawn_level=3, max_steps=MAX_STEPS_STAGE_B)

    # Nạp hoặc khởi tạo Model
    if args.resume and os.path.exists(args.resume):
        print(f"--> Đang resume mô hình từ: {args.resume}")
        model = SAC.load(args.resume, env=train_env, tensorboard_log=args.tb_log)
        model_dir = os.path.dirname(args.resume)
        buf_candidates = [
            os.path.join(model_dir, "stage_b_replay_buffer.pkl"),
            os.path.join(model_dir, "replay_buffer.pkl"),
            os.path.join(model_dir, "best_replay_buffer.pkl"),
        ]
        for bp in buf_candidates:
            if os.path.exists(bp):
                print(f"--> Nạp replay buffer từ: {bp}")
                model.load_replay_buffer(bp)
                break
    elif args.warm_start and os.path.exists(args.warm_start):
        print(f"--> Nạp trọng số Warm-Start từ: {args.warm_start}")
        model = SAC.load(
            args.warm_start,
            env=train_env,
            tensorboard_log=args.tb_log,
            custom_objects={"learning_rate": 2e-4},
        )
        # Reset entropy parameters post-load để tránh lỗi BaseClass set_parameters
        # và khôi phục khả năng exploration cho 4-DOF trong Stage B:
        target_initial_ent = 0.1
        with torch.no_grad():
            model.log_ent_coef.fill_(float(np.log(target_initial_ent)))
        model.target_entropy = -4.0
        # Reset optimizer cho log_ent_coef để xóa momentum cũ từ Stage A
        model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=2e-4)

        print("--> Đã kế thừa thành công chính sách tiếp cận và bù trọng trường từ Stage A!")
        print(f"--> [FIX] Reset ent_coef={target_initial_ent}, target_entropy=-4.0 để khôi phục exploration Stage B.")
    else:
        print("--> Khởi tạo SAC mới từ đầu (Cold-start).")
        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            gamma=0.98,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=300_000,
            learning_starts=2000,
            train_freq=1,
            gradient_steps=4,
            ent_coef="auto",
            target_entropy=-2.0,
            tensorboard_log=args.tb_log,
            verbose=1,
        )

    # Callback đánh giá cổng C1a (Level 3 toàn dải)
    eval_callback = HandoffEvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=25,
        save_path=args.save_dir,
        verbose=1,
    )

    callbacks = [eval_callback]
    if not args.no_curriculum:
        curriculum_cb = CurriculumCallback(train_env=train_env, switch_step=args.switch_step, verbose=1)
        callbacks.append(curriculum_cb)

    current_steps = model.num_timesteps if args.resume else 0
    target_timesteps = args.timesteps
    remaining_timesteps = max(0, target_timesteps - current_steps)

    print(f"--> Bắt đầu huấn luyện {remaining_timesteps} timesteps (Mục tiêu tổng: {target_timesteps})...\n")

    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                reset_num_timesteps=False if args.resume else True,
            )

        final_path = os.path.join(args.save_dir, "stage_b_final.zip")
        final_buffer = os.path.join(args.save_dir, "stage_b_replay_buffer.pkl")
        model.save(final_path)
        model.save_replay_buffer(final_buffer)
        print(f"\n[HOÀN TẤT] Huấn luyện Stage B thành công! Đã lưu tại: {final_path}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(" [PAUSE] ĐÃ NHẬN TÍN HIỆU NGẮT (Ctrl+C)! ĐANG LƯU CHECKPOINT AN TOÀN...")
        print("=" * 70)
        interrupted_path = os.path.join(args.save_dir, "stage_b_interrupted.zip")
        buffer_path = os.path.join(args.save_dir, "stage_b_replay_buffer.pkl")
        model.save(interrupted_path)
        try:
            model.save_replay_buffer(buffer_path)
            print(f"--> Đã lưu Replay Buffer tại: {buffer_path}")
        except Exception as e:
            print(f"--> Cảnh báo buffer: {e}")
        print(f"--> Đã lưu Checkpoint tại: {interrupted_path}")
        print(f"\nĐể tiếp tục huấn luyện bất cứ lúc nào, chạy:")
        print(f"  python3 src/assarm_rl_train/assarm_rl_train/train_stage_b.py --resume {interrupted_path} --timesteps {args.timesteps}")
        print("=" * 70)


if __name__ == "__main__":
    main()
