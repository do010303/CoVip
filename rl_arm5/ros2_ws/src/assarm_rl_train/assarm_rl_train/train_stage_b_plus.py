# -*- coding: utf-8 -*-
"""
assarm_rl_train.train_stage_b_plus — Huấn luyện SAC Phase 2+ Stage B+ (Dynamic Tracking).

- Kế thừa (Warm-start) từ mô hình hội tụ Stage B (models/best_model.zip hoặc stage_b_final.zip).
- Môi trường động: Vật thể di chuyển ngẫu nhiên chậm (v = 1.2 - 2.0 cm/s) trong toàn vùng làm việc Level 3.
- Quỹ đạo luôn được ràng buộc an toàn bên trong trường nhìn của Camera (FOV) và tầm vươn của cánh tay.
- Mạng SAC khai thác 3 chiều vận tốc obj_vel [18:21] trong vector quan sát để chủ động đón đầu mục tiêu (Predictive Interception).
- Đánh giá định kỳ bằng HandoffEvalCallback trên môi trường động.
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
from stable_baselines3 import SAC

from assarm_common.config import MAX_STEPS_STAGE_B
from assarm_rl_env.assarm_env import AssarmGraspEnv
from assarm_rl_train.callbacks import HandoffEvalCallback, CurriculumMotionCallback


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện SAC Phase 2+ Stage B+ v3 (Dynamic Motion Tracking với Ràng Buộc Tư Thế Chuẩn)")
    parser.add_argument("--warm_start", type=str, default="./models/stage_b_best.zip", help="Đường dẫn checkpoint để warm-start (mặc định stage_b_best.zip)")
    parser.add_argument("--actor_only", action="store_true", help="Chỉ kế thừa Actor weights (dùng khi warm-start từ Stage B tĩnh để tránh Critic Value Shock)")
    parser.add_argument("--timesteps", type=int, default=180000, help="Tổng timesteps huấn luyện Stage B+ v3")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate tinh chỉnh (fine-tuning)")
    parser.add_argument("--target_entropy", type=float, default=-2.0, help="Target entropy duy trì khả năng khám phá (mặc định -2.0)")
    parser.add_argument("--buffer_size", type=int, default=500000, help="Kích thước Replay Buffer")
    parser.add_argument("--learning_starts", type=int, default=3000, help="Số steps thu thập dữ liệu động trước khi cập nhật mạng")
    parser.add_argument("--gradient_steps", type=int, default=2, help="Số bước gradient mỗi env step")
    parser.add_argument("--save_dir", type=str, default="./models/", help="Thư mục lưu mô hình")
    parser.add_argument("--tb_log", type=str, default="./tb_logs/stage_b_plus_v3/", help="Thư mục TensorBoard Stage B+ v3")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.zip) để huấn luyện tiếp tục")
    parser.add_argument("--motion_stage", type=int, default=0, help="0: Auto 4-Phase Curriculum | 1..4: Cố định stage tương ứng")
    parser.add_argument("--p1_steps", type=int, default=30000, help="Step kết thúc Phase 1 (Stage 1: theta [-10°, +10°], v = 0.8 cm/s)")
    parser.add_argument("--p2_steps", type=int, default=70000, help="Step kết thúc Phase 2 (Stage 2: theta [-22°, +22°], v = 1.4 cm/s)")
    parser.add_argument("--p3_steps", type=int, default=120000, help="Step kết thúc Phase 3 (Stage 3: theta [-35°, +35°], v = 1.6 cm/s) -> Sang Phase 4 (Random Level 3 + OU wind)")
    args = parser.parse_args()

    # Tối ưu đa luồng CPU
    torch.set_num_threads(2)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    # Fallback warm_start nếu không tìm thấy file mặc định
    warm_start_path = args.warm_start
    if not (warm_start_path and os.path.exists(warm_start_path)):
        fallback_path = "./models/best_model.zip"
        if os.path.exists(fallback_path):
            warm_start_path = fallback_path

    print("=" * 80)
    print(" KHỞI CHẠY HUẤN LUYỆN SAC STAGE B+ (PHASE 2+ DYNAMIC MOTION TRACKING)")
    print(f" - Tổng timesteps: {args.timesteps}")
    print(f" - Chiến lược khởi tạo: {'Actor-Only Warm-Start (Fresh Critic)' if args.actor_only else 'Full Model Warm-Start'}")
    print(f" - Target Entropy: {args.target_entropy} (Chống Entropy Collapse)")
    print(f" - Learning Rate: {args.lr} | Gradient Steps: {args.gradient_steps} | Replay Buffer: {args.buffer_size}")
    print(f" - Learning Starts: {args.learning_starts} steps thu thập mẫu")
    print(f" - Max steps/episode: {MAX_STEPS_STAGE_B} (8.0 giây)")
    if args.resume:
        print(f" - RESUME từ: {args.resume}")
    elif warm_start_path and os.path.exists(warm_start_path):
        print(f" - WARM-START từ: {warm_start_path}")
    else:
        print(" - CẢNH BÁO: Không tìm thấy checkpoint warm-start, sẽ khởi tạo mới hoàn toàn!")
    print("=" * 80)

    # 1. Khởi tạo môi trường Động (dynamic_motion = True)
    initial_stage = 1 if (args.motion_stage in (0, 1)) else 2
    train_env = AssarmGraspEnv(spawn_level=3, max_steps=MAX_STEPS_STAGE_B, dynamic_motion=True, motion_stage=initial_stage)
    eval_env = AssarmGraspEnv(spawn_level=3, max_steps=MAX_STEPS_STAGE_B, dynamic_motion=True, motion_stage=initial_stage)

    # 2. Nạp hoặc khởi tạo mô hình
    if args.resume and os.path.exists(args.resume):
        print(f"--> Đang resume mô hình từ: {args.resume}")
        model = SAC.load(args.resume, env=train_env, tensorboard_log=args.tb_log)
    elif warm_start_path and os.path.exists(warm_start_path):
        if args.actor_only:
            print(f"--> Áp dụng ACTOR-ONLY WARM-START từ: {warm_start_path}")
            print("--> Khởi tạo SAC với Fresh Critic & Target Entropy để chống Critic Value Shock...")
            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=dict(net_arch=[256, 256, 256]),
                gamma=0.98,
                learning_rate=args.lr,
                batch_size=256,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                train_freq=1,
                gradient_steps=args.gradient_steps,
                ent_coef="auto",
                target_entropy=args.target_entropy,
                tensorboard_log=args.tb_log,
                verbose=1,
            )
            pretrained_sac = SAC.load(warm_start_path, device="cpu")
            model.policy.actor.load_state_dict(pretrained_sac.policy.actor.state_dict())
            del pretrained_sac
            print("--> [THÀNH CÔNG] Đã chuyển giao toàn bộ weights Actor từ Stage B! Critic được khởi tạo độc lập.")
        else:
            print(f"--> Nạp toàn bộ trọng số (Actor + Critic) từ Stage B: {warm_start_path}")
            model = SAC.load(
                warm_start_path,
                env=train_env,
                tensorboard_log=args.tb_log,
                custom_objects={"learning_rate": args.lr},
            )
            model.target_entropy = args.target_entropy
    else:
        print("--> Khởi tạo SAC mới từ đầu (Cold-start).")
        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            gamma=0.98,
            learning_rate=args.lr,
            batch_size=256,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            train_freq=1,
            gradient_steps=args.gradient_steps,
            ent_coef="auto",
            target_entropy=args.target_entropy,
            tensorboard_log=args.tb_log,
            verbose=1,
        )

    # 3. Callback đánh giá định kỳ trên môi trường động
    save_stage_b_plus = os.path.join(args.save_dir, "stage_b_plus")
    os.makedirs(save_stage_b_plus, exist_ok=True)
    eval_callback = HandoffEvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=25,
        save_path=save_stage_b_plus,
        verbose=1,
    )

    callbacks = [eval_callback]

    current_steps = model.num_timesteps if args.resume else 0

    # Nếu cấu hình Auto Curriculum (args.motion_stage == 0)
    if args.motion_stage == 0:
        curriculum_cb = CurriculumMotionCallback(
            train_env=train_env,
            eval_env=eval_env,
            p1_steps=args.p1_steps,
            p2_steps=args.p2_steps,
            p3_steps=args.p3_steps,
            verbose=1,
        )
        callbacks.append(curriculum_cb)

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

        final_path = os.path.join(args.save_dir, "stage_b_plus_final.zip")
        final_buffer = os.path.join(args.save_dir, "stage_b_plus_replay_buffer.pkl")
        model.save(final_path)
        model.save_replay_buffer(final_buffer)
        print(f"\n[HOÀN TẤT] Huấn luyện Stage B+ thành công! Đã lưu tại: {final_path}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 75)
        print(" [PAUSE] ĐÃ NHẬN TÍN HIỆU NGẮT (Ctrl+C)! ĐANG LƯU CHECKPOINT AN TOÀN...")
        print("=" * 75)
        interrupted_path = os.path.join(args.save_dir, "stage_b_plus_interrupted.zip")
        buffer_path = os.path.join(args.save_dir, "stage_b_plus_replay_buffer.pkl")
        model.save(interrupted_path)
        try:
            model.save_replay_buffer(buffer_path)
            print(f"--> Đã lưu Replay Buffer tại: {buffer_path}")
        except Exception as e:
            print(f"--> Cảnh báo buffer: {e}")
        print(f"--> Đã lưu Checkpoint tại: {interrupted_path}")
        print(f"\nĐể tiếp tục huấn luyện bất cứ lúc nào, chạy:")
        print(f"  python3 src/assarm_rl_train/assarm_rl_train/train_stage_b_plus.py --resume {interrupted_path} --timesteps {args.timesteps}")
        print("=" * 75)


if __name__ == "__main__":
    main()
