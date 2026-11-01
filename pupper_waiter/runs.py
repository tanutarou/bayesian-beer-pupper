"""Load a trained run (``logs/<exp>/``) for evaluation and demos."""

from __future__ import annotations

import os
import pickle

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from pupper_waiter.config import EnvCfg


def load_config(exp_name: str) -> tuple:
    """(env_cfg, obs_cfg, reward_cfg, rsl_cfg) as saved by ``scripts/train.py``."""
    with open(os.path.join("logs", exp_name, "cfgs.pkl"), "rb") as f:
        return pickle.load(f)


def use_nominal_robot(env_cfg: EnvCfg) -> None:
    """Evaluate one nominal robot: no observation or start-pose noise, servo stiffness and damping at the middle
    of the training ranges, nominal friction and mass."""
    env_cfg.obs_noise.gravity = env_cfg.obs_noise.ang_vel = env_cfg.obs_noise.dof_pos = 0.0
    env_cfg.domain_rand.init_joint_noise = 0.0
    dr = env_cfg.domain_rand
    dr.kp_range = (sum(dr.kp_range) / 2,) * 2
    dr.kv_range = (sum(dr.kv_range) / 2,) * 2
    dr.friction_range = (1.0, 1.0)
    dr.mass_scale_range = (1.0, 1.0)


def make_runner(env, rsl_cfg: dict, exp_name: str) -> OnPolicyRunner:
    return OnPolicyRunner(env, rsl_cfg, os.path.join("logs", exp_name), device=gs.device)


def load_actor(runner: OnPolicyRunner, exp_name: str, ckpt: int):
    """Load the actor of checkpoint ``ckpt`` and return the deterministic policy.

    Only the actor is needed: the critic's input depends on the training set-up (e.g. the cup stage's
    privileged cup state), which an evaluation need not reproduce.
    """
    runner.load(
        os.path.join("logs", exp_name, f"model_{ckpt}.pt"),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
    )
    return runner.get_inference_policy(device=gs.device)


def load_policy(env, rsl_cfg: dict, exp_name: str, ckpt: int):
    return load_actor(make_runner(env, rsl_cfg, exp_name), exp_name, ckpt)
