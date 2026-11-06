# -*- coding: utf-8 -*-
"""
assarm_rl_train.train — Script huấn luyện SAC cho Phase 1 Stage A.

Huấn luyện mô hình SAC gắp khối đứng yên (Level 0-1) đạt handoff_rate >= 90%.
"""

import argparse
import os
import sys

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

from assarm_rl_env.assarm_env import AssarmGraspEnv
from assarm_rl_train.callbacks import HandoffEvalCallback


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện SAC Phase 1 Stage A")
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file cấu hình YAML (tùy chọn)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Tổng số timesteps huấn luyện")
    parser.add_argument("--level", type=int, default=0, help="Cấp độ spawn (0 hoặc 1 cho Stage A)")
    parser.add_argument("--save_dir", type=str, default="./models/", help="Thư mục lưu mô hình")
    parser.add_argument("--tb_log", type=str, default="./tb_logs/", help="Thư mục Tensorboard")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn mô hình (.zip) để huấn luyện tiếp tục")
    args = parser.parse_args()

    # Nạp cấu hình từ YAML nếu có
    if args.config and os.path.exists(args.config):
        print(f"--> Đang nạp cấu hình từ: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "timesteps" in cfg:
            args.timesteps = cfg["timesteps"]
        if "spawn_level" in cfg:
            args.level = cfg["spawn_level"]

    # Tối ưu cho CPU i5-1035G4 (4 cores)
    torch.set_num_threads(2)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 70)
    print(f" KHỞI CHẠY HUẤN LUYỆN SAC STAGE A (Level {args.level}, {args.timesteps} timesteps)")
    if args.resume:
        print(f" RESUME TỪ CHECKPOINT: {args.resume}")
    print("=" * 70)

    # Môi trường train & eval
    env = AssarmGraspEnv(spawn_level=args.level)
    eval_env = AssarmGraspEnv(spawn_level=args.level)

    if args.resume and os.path.exists(args.resume):
        print(f"--> Đang nạp mô hình từ: {args.resume}")
        model = SAC.load(args.resume, env=env, tensorboard_log=args.tb_log)

        # Tìm replay buffer tương ứng
        model_dir = os.path.dirname(args.resume)
        possible_buffers = [
            os.path.join(model_dir, "replay_buffer.pkl"),
            os.path.join(model_dir, "latest_replay_buffer.pkl"),
            os.path.join(model_dir, "best_replay_buffer.pkl"),
            os.path.join(model_dir, "interrupted_replay_buffer.pkl"),
        ]
        buffer_found = False
        for buf_path in possible_buffers:
            if os.path.exists(buf_path):
                print(f"--> Đang nạp Replay Buffer từ: {buf_path}")
                model.load_replay_buffer(buf_path)
                buffer_found = True
                break
        if not buffer_found:
            print("--> Thông báo: Không tìm thấy file replay_buffer.pkl, sẽ khởi tạo Replay Buffer mới.")
    else:
        # Khởi tạo mô hình SAC mới theo thiết lập v5.0
        model = SAC(
            "MlpPolicy",
            env,
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

    # Callback đánh giá
    eval_callback = HandoffEvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=20,
        save_path=args.save_dir,
        verbose=1,
    )

    # Tính toán timesteps còn lại nếu resume
    current_steps = model.num_timesteps
    target_timesteps = args.timesteps
    remaining_timesteps = max(0, target_timesteps - current_steps) if args.resume else target_timesteps

    print(f"--> Số timesteps hiện tại: {current_steps}, sẽ train thêm: {remaining_timesteps} timesteps (Tổng mục tiêu: {target_timesteps})")

    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=eval_callback,
                reset_num_timesteps=False if args.resume else True,
            )
        # Lưu model hoàn tất
        final_path = os.path.join(args.save_dir, "stage_a_final.zip")
        final_buffer = os.path.join(args.save_dir, "replay_buffer.pkl")
        model.save(final_path)
        model.save_replay_buffer(final_buffer)
        print(f"\n[DONE] Đã hoàn thành huấn luyện Stage A và lưu tại: {final_path}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(" [PAUSE] ĐÃ NHẬN TÍN HIỆU NGẮT (Ctrl+C)! ĐANG LƯU CHECKPOINT VÀ REPLAY BUFFER...")
        print("=" * 70)
        interrupted_path = os.path.join(args.save_dir, "checkpoint_interrupted.zip")
        buffer_path = os.path.join(args.save_dir, "replay_buffer.pkl")
        model.save(interrupted_path)
        try:
            model.save_replay_buffer(buffer_path)
            print(f"--> Đã lưu Replay Buffer tại: {buffer_path}")
        except Exception as e:
            print(f"--> Cảnh báo lưu buffer: {e}")

        print(f"--> Đã lưu Model Checkpoint tại: {interrupted_path}")
        print("\nBạn có thể tắt máy an toàn! Khi bật lại máy, tiếp tục train bằng lệnh:")
        print(f"  python3 -m assarm_rl_train.train --resume {interrupted_path} --timesteps {args.timesteps}")
        print("=" * 70)


if __name__ == "__main__":
    main()
