"""Train a walking policy with PPO (rsl_rl).

Usage (the three stages, see README.md; the fine-tuning stages add ``--max-iterations 300``):
  uv run python scripts/train.py -e pretrain --task pretrain
  uv run python scripts/train.py -e level-walker --task walk --resume logs/pretrain/model_599.pt
  uv run python scripts/train.py -e cup-walker --task cupwalk --resume logs/level-walker/model_898.pt

``--reward NAME=SCALE`` overrides reward weights (e.g. ``level=0 tilt=0 ang_vel_xy=0`` for a plain walker).
Checkpoints, the configuration (``cfgs.pkl`` for the scripts, ``cfgs.json`` for people) and the TensorBoard log
go to ``logs/<exp>/``. A new run refuses to start if that directory already exists, so earlier results are
never overwritten by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from pupper_waiter.config import TASKS, TrainCfg
from pupper_waiter.env import PupperEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("-b", "--num-envs", type=int, default=TrainCfg.num_envs)
    parser.add_argument("--max-iterations", type=int, default=TrainCfg.max_iterations)
    parser.add_argument("--seed", type=int, default=TrainCfg.seed)
    parser.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
    parser.add_argument("--reward", nargs="*", default=[], metavar="NAME=SCALE", help="override reward weights")
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg = TASKS[args.task]()
    for item in args.reward:
        name, value = item.split("=")
        if name not in reward_cfg.scales:
            raise SystemExit(f"unknown reward term {name!r}; known: {sorted(reward_cfg.scales)}")
        reward_cfg.scales[name] = float(value)
    rsl_cfg = TrainCfg(num_envs=args.num_envs, max_iterations=args.max_iterations, seed=args.seed).to_rsl_rl(
        args.exp_name
    )

    log_dir = os.path.join("logs", args.exp_name)
    if os.path.exists(log_dir) and args.resume is None:
        raise SystemExit(f"{log_dir} already exists; choose another --exp-name or remove it yourself")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, rsl_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = PupperEnv(args.num_envs, env_cfg, obs_cfg, reward_cfg)
    with open(os.path.join(log_dir, "cfgs.json"), "w") as f:
        json.dump({**env.config_dict(), "train": rsl_cfg}, f, indent=2)

    runner = OnPolicyRunner(env, rsl_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        # The cup stage's critic sees more privileged inputs than the walking critic, so across stages only the
        # actor is carried over and the critic and optimizer start fresh.
        checkpoint = torch.load(args.resume, weights_only=False, map_location=gs.device)
        critic = runner.alg._raw_critic.state_dict()
        if all(checkpoint["critic_state_dict"][k].shape == v.shape for k, v in critic.items()):
            runner.load(args.resume)
        else:
            runner.load(args.resume, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
            print("resumed the actor only; the critic and the optimizer start fresh")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
