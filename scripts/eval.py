"""Evaluate a walking policy on the nominal robot and print gait and cup metrics.

Usage:
  uv run python scripts/eval.py -e level-walker --ckpt 898 --volume 100          # fixed 105 g weight on the tray
  uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --seed 1      # free cup on the bare tray
  uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --holder --seed 1

The robot walks at a fixed command (default 0.15 m/s forward) and is pushed as in training; ``--seed`` fixes
the push sequence.
"""

from __future__ import annotations

import argparse

import genesis as gs
import numpy as np
import torch

from pupper_waiter.env import PupperEnv
from pupper_waiter.metrics import RolloutMetrics, format_summary
from pupper_waiter.payload import volume_to_payload
from pupper_waiter.runs import load_config, load_policy, use_nominal_robot

URDF_DIR = "assets/mini_pupper_2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--command", type=float, nargs=3, default=(0.15, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=None, help="seed of the push sequence (default: random)")
    parser.add_argument("--volume", type=float, default=None, help="fixed weight on the tray: a 5 g cup plus this [ml]")
    parser.add_argument("--cup", type=float, default=None, help="free cylinder cup of this mass [g] on the tray")
    parser.add_argument("--holder", action="store_true", help="stand the free cup in the 35 mm deep holder")
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, rsl_cfg = load_config(args.exp_name)
    use_nominal_robot(env_cfg)
    env_cfg.episode_length_s = args.seconds + 1.0
    env_cfg.free_cup.enabled = False  # no training cup and no cup randomisation; the cup is the one asked for
    free_cup = None
    if args.cup is not None:
        free_cup = {"mass": args.cup * 1e-3, "friction": 0.5}
        holder = "_tray_holder" if args.holder else "_tray_bare"
        env_cfg.robot_urdf = f"{URDF_DIR}/mini_pupper_2{holder}.urdf"
        env_cfg.payload.enabled = False
    elif args.volume is not None:
        env_cfg.robot_urdf = f"{URDF_DIR}/mini_pupper_2_tray.urdf"
        env_cfg.payload.enabled = True

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    env = PupperEnv(1, env_cfg, obs_cfg, reward_cfg, free_cup=free_cup)
    if args.volume is not None:
        env.fixed_payload = volume_to_payload(torch.tensor([args.volume], device=gs.device))
    env.fixed_command = torch.tensor(args.command, device=gs.device)
    env.reset()
    policy = load_policy(env, rsl_cfg, args.exp_name, args.ckpt)

    obs = env.reset()
    metrics = RolloutMetrics(env.dt)
    velocities = []
    with torch.no_grad():
        for _ in range(int(args.seconds / env.dt)):
            obs, _, dones, _ = env.step(policy(obs))
            metrics.update(env, bool(dones[0]))
            velocities.append(torch.cat((env.base_lin_vel[0, :2], env.base_ang_vel[0, 2:])).tolist())

    v = np.array(velocities)
    print("metrics (env 0):\n" + format_summary(metrics.summary()))
    print(f"mean forward speed {v[:, 0].mean():.3f} m/s (commanded {args.command[0]:.3f} m/s)")


if __name__ == "__main__":
    main()
