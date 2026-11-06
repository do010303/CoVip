# -*- coding: utf-8 -*-
"""
assarm_rl_train.callbacks — Callbacks cho quá trình huấn luyện SAC.

Bao gồm:
- EvaluationCallback: Đánh giá định kỳ tỉ lệ bàn giao (handoff_rate)
- CheckpointCallback: Lưu checkpoint mô hình tốt nhất
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HandoffEvalCallback(BaseCallback):
    """
    Callback đánh giá tỉ lệ bàn giao (handoff_rate) định kỳ mỗi eval_freq steps.

    Parameters
    ----------
    eval_env : Gymnasium Env
        Môi trường đánh giá.
    eval_freq : int
        Tần suất đánh giá (mẫu step). Mặc định 5000.
    n_eval_episodes : int
        Số episode mỗi đợt đánh giá. Mặc định 20.
    save_path : str
        Thư mục lưu mô hình tốt nhất.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 5000,
        n_eval_episodes: int = 20,
        save_path: str = "./models/",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_handoff_rate = 0.0

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 1. Đánh giá Deterministic
            handoff_count_det = 0
            good_handoff_count_det = 0
            total_rewards_det = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                    if info.get("handoff_ok", False):
                        handoff_count_det += 1
                    if info.get("good_handoff", False):
                        good_handoff_count_det += 1
                total_rewards_det.append(ep_reward)

            # 2. Đánh giá Stochastic
            handoff_count_stoch = 0
            good_handoff_count_stoch = 0
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    if info.get("handoff_ok", False):
                        handoff_count_stoch += 1
                    if info.get("good_handoff", False):
                        good_handoff_count_stoch += 1

            handoff_rate_det = (handoff_count_det / self.n_eval_episodes) * 100.0
            good_handoff_rate_det = (good_handoff_count_det / self.n_eval_episodes) * 100.0
            handoff_rate_stoch = (handoff_count_stoch / self.n_eval_episodes) * 100.0
            mean_reward = float(np.mean(total_rewards_det))

            if self.verbose > 0:
                print(
                    f"\n[Eval @ Step {self.num_timesteps}] "
                    f"Handoff Det: {handoff_rate_det:.1f}% ({handoff_count_det}/{self.n_eval_episodes}) | "
                    f"Good Handoff: {good_handoff_rate_det:.1f}% | "
                    f"Stoch: {handoff_rate_stoch:.1f}% | "
                    f"Mean Rew: {mean_reward:.2f}"
                )

            self.logger.record("eval/handoff_rate", handoff_rate_det)
            self.logger.record("eval/good_handoff_rate", good_handoff_rate_det)
            self.logger.record("eval/handoff_rate_stoch", handoff_rate_stoch)
            self.logger.record("eval/mean_reward", mean_reward)

            # Luôn lưu checkpoint mới nhất cùng replay buffer
            latest_model_path = os.path.join(self.save_path, "latest_model.zip")
            latest_buffer_path = os.path.join(self.save_path, "latest_replay_buffer.pkl")
            self.model.save(latest_model_path)
            try:
                self.model.save_replay_buffer(latest_buffer_path)
            except Exception as e:
                if self.verbose > 0:
                    print(f"--> Warning: Could not save replay buffer: {e}")

            if handoff_rate_det > self.best_handoff_rate:
                self.best_handoff_rate = handoff_rate_det
                best_model_path = os.path.join(self.save_path, "best_model.zip")
                best_buffer_path = os.path.join(self.save_path, "best_replay_buffer.pkl")
                self.model.save(best_model_path)
                try:
                    self.model.save_replay_buffer(best_buffer_path)
                except Exception:
                    pass
                if self.verbose > 0:
                    print(f"--> Saved new best model & buffer to {best_model_path} (handoff_rate_det = {handoff_rate_det:.1f}%)")

        return True


class CurriculumMotionCallback(BaseCallback):
    """
    Callback điều phối Curriculum 4-Phase Mở rộng Không gian & Vận tốc cho Stage B+ v2:
    - Phase 1 (0 -> p1): Stage 1 (theta [-10°, +10°], v = 0.8 cm/s) - Bám vết trung tâm ổn định
    - Phase 2 (p1 -> p2): Stage 2 (theta [-22°, +22°], v = 1.4 cm/s) - Mở rộng góc trung bình
    - Phase 3 (p2 -> p3): Stage 3 (theta [-35°, +35°], v = 1.6 cm/s) - Toàn dải góc với tốc độ chuẩn
    - Phase 4 (> p3): Stage 4 (Toàn dải Level 3 [-35°, +35°], v = 1.2-2.0 cm/s ngẫu nhiên + Nhiễu gió OU)
    """

    def __init__(
        self,
        train_env,
        eval_env,
        p1_steps: int = 30000,
        p2_steps: int = 70000,
        p3_steps: int = 120000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.p1_steps = p1_steps
        self.p2_steps = p2_steps
        self.p3_steps = p3_steps
        self.current_phase = 0

    def _set_env_stage(self, env, stage: int):
        if hasattr(env, "set_motion_stage"):
            env.set_motion_stage(stage)
        elif hasattr(env, "env_method"):
            env.env_method("set_motion_stage", stage)

    def _set_env_velocity(self, env, v_mps: float):
        if hasattr(env, "set_target_velocity"):
            env.set_target_velocity(v_mps)
        elif hasattr(env, "env_method"):
            env.env_method("set_target_velocity", v_mps)

    def _on_training_start(self) -> None:
        self._update_phase(self.num_timesteps, force=True)

    def _update_phase(self, step: int, force: bool = False):
        new_phase = 1
        if step >= self.p3_steps:
            new_phase = 4
        elif step >= self.p2_steps:
            new_phase = 3
        elif step >= self.p1_steps:
            new_phase = 2

        if new_phase != self.current_phase or force:
            self.current_phase = new_phase
            if new_phase == 1:
                self._set_env_stage(self.train_env, 1)
                self._set_env_stage(self.eval_env, 1)
                self._set_env_velocity(self.train_env, 0.008)
                self._set_env_velocity(self.eval_env, 0.008)
                desc = "PHASE 1 (Stage 1): theta in [-10°, +10°], v = 0.8 cm/s (Tâm & Lân cận gần)"
            elif new_phase == 2:
                self._set_env_stage(self.train_env, 2)
                self._set_env_stage(self.eval_env, 2)
                self._set_env_velocity(self.train_env, 0.014)
                self._set_env_velocity(self.eval_env, 0.014)
                desc = "PHASE 2 (Stage 2): Mở rộng theta in [-22°, +22°], v = 1.4 cm/s (Góc trung bình)"
            elif new_phase == 3:
                self._set_env_stage(self.train_env, 3)
                self._set_env_stage(self.eval_env, 3)
                self._set_env_velocity(self.train_env, 0.016)
                self._set_env_velocity(self.eval_env, 0.016)
                desc = "PHASE 3 (Stage 3): Toàn dải theta in [-35°, +35°], v = 1.6 cm/s (Biên rộng chuẩn)"
            else:
                self._set_env_stage(self.train_env, 4)
                self._set_env_stage(self.eval_env, 4)
                self._set_env_velocity(self.train_env, 0.0)  # Reset fixed speed -> random [1.2, 2.0] cm/s
                self._set_env_velocity(self.eval_env, 0.0)
                desc = "PHASE 4 (Stage 4): Toàn dải theta in [-35°, +35°] + v in [1.2, 2.0] cm/s + Nhiễu gió OU"

            if self.verbose > 0:
                print("\n" + "=" * 80)
                print(f" [CURRICULUM STAGE B+ v2] CẬP NHẬT GIAI ĐOẠN TẠI STEP {step}:")
                print(f" -> {desc}")
                print("=" * 80 + "\n")

    def _on_step(self) -> bool:
        self._update_phase(self.num_timesteps)
        return True

