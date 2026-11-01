"""Carry a glass of beer: the glass stands in the holder on the tray and holds MPM liquid. Reports how much
beer was spilled and how much the surface sloshed.

The robot first stands still while the poured beer settles (``--fill-seconds``), then walks at the commanded
speed; the metrics cover the walk only.

Usage: uv run python scripts/demo_beer.py -e cup-walker --ckpt 299 --volume 62 --command 0.05 0 0 --seed 1
"""

from __future__ import annotations

import argparse

import genesis as gs
import numpy as np
import torch

from pupper_waiter import liquid
from pupper_waiter.env import PupperEnv
from pupper_waiter.metrics import RolloutMetrics, format_summary, surface_angle_deg
from pupper_waiter.runs import load_config, load_policy, use_nominal_robot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--volume", type=float, default=62.0, help="beer poured into the glass [ml]")
    parser.add_argument("--command", type=float, nargs=3, default=(0.05, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seconds", type=float, default=12.0, help="walking time [s]")
    parser.add_argument("--fill-seconds", type=float, default=3.0, help="standing time while the beer settles [s]")
    parser.add_argument("--grid", type=int, default=200, help="MPM grid density (cells per metre)")
    parser.add_argument("--substeps", type=int, default=300, help="physics substeps per 20 ms control step")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, rsl_cfg = load_config(args.exp_name)
    use_nominal_robot(env_cfg)
    env_cfg.robot_urdf = liquid.HOLDER_URDF
    env_cfg.payload.enabled = False
    env_cfg.free_cup.enabled = False
    env_cfg.push.force_range = (0.0, 0.0)
    env_cfg.substeps = args.substeps
    env_cfg.episode_length_s = args.seconds + args.fill_seconds + 5.0

    entities = {}
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    env = PupperEnv(
        1,
        env_cfg,
        obs_cfg,
        reward_cfg,
        # The particle coupling does not carry the static weight of the liquid to the glass, so the glass
        # body is given the liquid's mass (1 g/ml).
        free_cup=liquid.glass(liquid.GLASS_MASS_KG + args.volume * 1e-3),
        add_entities=liquid.beer_adder(env_cfg, args.volume, entities),
        scene_kwargs=liquid.mpm_scene_kwargs(env_cfg, args.command[0], args.seconds, args.grid),
        robot_material=gs.materials.Rigid(needs_coup=False),
        floor_material=gs.materials.Rigid(needs_coup=False),
    )
    beer = entities["beer"]
    command = torch.tensor(args.command, device=gs.device)
    env.fixed_command = command
    policy = load_policy(env, rsl_cfg, args.exp_name, args.ckpt)

    def glass_pose():
        return env.cup.get_pos()[0], env.cup.get_quat()[:1]

    obs = env.reset()
    # Particle positions are valid after the first step; the column has hardly moved in 20 ms.
    obs, _, _, _ = env.step(torch.zeros_like(env.actions))
    print(f"poured: column heights scaled by {liquid.pour(beer, *glass_pose()):.2f} to start below the rim")
    metrics = RolloutMetrics(env.dt)
    surface, guarded = [], 0
    n_fill = int(round(args.fill_seconds / env.dt))
    n_walk = int(args.seconds / env.dt)
    with torch.no_grad():
        for step in range(n_fill + n_walk):
            filling = step < n_fill
            env.fixed_command = torch.zeros(3, device=gs.device) if filling else command
            env.commands[:] = env.fixed_command
            obs, _, dones, _ = env.step(policy(obs))
            moved = liquid.guard_walls(beer, env.robot.get_vel()[0], *glass_pose(), funnel=filling)
            if not filling:
                guarded += moved
                metrics.update(env, bool(dones[0]))
                center, quat = glass_pose()
                surface.append(surface_angle_deg(beer.get_state().pos.reshape(-1, 3), center, quat[0]))

    result = liquid.measure(beer, *glass_pose(), args.grid)
    summary = metrics.summary()
    summary["surface_angle_max_deg"] = float(np.nanmax(surface))
    summary["surface_angle_rms_deg"] = float(np.sqrt(np.nanmean(np.square(surface))))
    print(f"liquid: {result['particles']} particles = {result['volume_ml']:.0f} ml (poured {args.volume:.0f} ml)")
    print(f"liquid level after the walk: {result['level_m'] * 100:.1f} cm above the glass floor")
    print(f"spilled: {result['spilled'] * 100:.1f}% of the particles")
    print(f"wall guard: {guarded / n_walk:.2f} particles moved back per control step")
    print("metrics (env 0):\n" + format_summary(summary))


if __name__ == "__main__":
    main()
