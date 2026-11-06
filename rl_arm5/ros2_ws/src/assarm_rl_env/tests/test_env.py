# -*- coding: utf-8 -*-
"""
Unit tests cho Gymnasium Environment assarm_rl_env.assarm_env.
"""

import numpy as np
import pytest
from assarm_rl_env.assarm_env import AssarmGraspEnv


class TestAssarmEnv:
    """Bộ kiểm thử tính hợp lệ của AssarmGraspEnv."""

    def test_env_reset(self):
        """Kiểm tra reset trả về obs (28,) và info dict."""
        env = AssarmGraspEnv(spawn_level=0)
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (28,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert "spawn_level" in info
        assert info["spawn_level"] == 0

    def test_env_step_returns(self):
        """Kiểm tra step trả về đúng tuple 5 phần tử Gymnasium API."""
        env = AssarmGraspEnv(spawn_level=0)
        obs, info = env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        obs_next, reward, terminated, truncated, info_step = env.step(action)

        assert obs_next.shape == (28,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info_step, dict)

    def test_env_episode_truncation(self):
        """Kiểm tra episode bị truncate khi đủ max_steps."""
        max_steps = 10
        env = AssarmGraspEnv(spawn_level=0, max_steps=max_steps)
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        truncated = False
        for step in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert truncated or terminated

    def test_random_policy_rollout(self):
        """Rollout 50 steps ngẫu nhiên mà không bị crash hay NaN."""
        env = AssarmGraspEnv(spawn_level=1)
        obs, info = env.reset(seed=42)

        np.random.seed(42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()
            assert not np.isnan(reward)
            if terminated or truncated:
                obs, info = env.reset()
