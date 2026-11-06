# -*- coding: utf-8 -*-
"""assarm_rl_env — Gymnasium environment cho ASSARM robot."""

from gymnasium.envs.registration import register

register(
    id='AssarmGrasp-v0',
    entry_point='assarm_rl_env.assarm_env:AssarmEnv',
)
