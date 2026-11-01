"""Beer-carrying demo: weigh the glass standing still, then walk at a speed chosen from the estimate.

The glass of beer stands in the holder. For ``--weigh-seconds`` the robot stands while the estimator updates
its belief about the volume; then it walks at

    v = v_min + (v_max - v_min) (1 - q90 / v_full),  clipped to [v_min, v_max],

where q90 is the 90% upper quantile of the belief. An empty glass is carried at v_max and a full one at
v_min, and uncertainty counts as "maybe full".

Usage: uv run python scripts/demo_serve.py -e cup-walker --ckpt 299 --volume 62 --seed 1 --walk-seconds 30
The belief at every control step is written to ``logs/<exp>/serve_<volume>ml_posterior.csv``.
"""

from __future__ import annotations

import argparse
import math
import os

import genesis as gs
import numpy as np
import torch

from pupper_waiter import liquid
from pupper_waiter.env import PupperEnv
from pupper_waiter.estimator import FeaturePipeline, VolumeKalman, load_estimator
from pupper_waiter.metrics import RolloutMetrics, format_summary, surface_angle_deg
from pupper_waiter.runs import load_config, load_policy, use_nominal_robot


def speed_from_belief(q90_ml: float, v_full_ml: float, v_max: float, v_min: float) -> float:
    return float(np.clip(v_min + (v_max - v_min) * (1.0 - q90_ml / v_full_ml), v_min, v_max))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--volume", type=float, default=62.0, help="beer poured into the glass [ml]")
    parser.add_argument("--estimator", default=None, help="default: logs/<exp>/sim_estimator.npz")
    parser.add_argument("--weigh-seconds", type=float, default=6.0)
    parser.add_argument("--walk-seconds", type=float, default=30.0)
    parser.add_argument("--v-max", type=float, default=0.05, help="speed for an empty glass [m/s]")
    parser.add_argument("--v-min", type=float, default=0.02, help="speed for a full glass [m/s]")
    parser.add_argument("--v-full", type=float, default=91.0, help="volume of a full glass [ml]")
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-liquid", action="store_true", help="empty glass with the beer's mass (quick check)")
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, rsl_cfg = load_config(args.exp_name)
    use_nominal_robot(env_cfg)
    env_cfg.robot_urdf = liquid.HOLDER_URDF
    env_cfg.payload.enabled = False
    env_cfg.free_cup.enabled = False
    env_cfg.push.force_range = (0.0, 0.0)
    env_cfg.substeps = args.substeps
    total_seconds = args.weigh_seconds + args.walk_seconds
    env_cfg.episode_length_s = total_seconds + 5.0

    entities = {}
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    env = PupperEnv(
        1,
        env_cfg,
        obs_cfg,
        reward_cfg,
        free_cup=liquid.glass(liquid.GLASS_MASS_KG + (0.0 if args.no_liquid else args.volume * 1e-3)),
        add_entities=None if args.no_liquid else liquid.beer_adder(env_cfg, args.volume, entities),
        scene_kwargs=liquid.mpm_scene_kwargs(env_cfg, args.v_max, args.walk_seconds, args.grid),
        robot_material=gs.materials.Rigid(needs_coup=False),
        floor_material=gs.materials.Rigid(needs_coup=False),
    )
    beer = entities.get("beer")
    env.fixed_command = torch.zeros(3, device=gs.device)
    policy = load_policy(env, rsl_cfg, args.exp_name, args.ckpt)

    estimator_path = args.estimator or os.path.join("logs", args.exp_name, "sim_estimator.npz")
    estimator, blocks, bias_var, noise_var = load_estimator(estimator_path)
    kalman = VolumeKalman(bias_var=bias_var)
    pipeline = FeaturePipeline()

    def glass_pose():
        return env.cup.get_pos()[0], env.cup.get_quat()[:1]

    obs = env.reset()
    if beer is not None:
        obs, _, _, _ = env.step(torch.zeros_like(env.actions))
        print(f"poured: column heights scaled by {liquid.pour(beer, *glass_pose()):.2f} to start below the rim")
    posterior_log, surface, guarded, out_of_range = [], [], 0, 0
    metrics = RolloutMetrics(env.dt)
    speed = None
    with torch.no_grad():
        for step in range(int(total_seconds / env.dt)):
            t = step * env.dt
            weighing = t < args.weigh_seconds
            if not weighing and speed is None:
                q90 = kalman.upper(0.9)
                speed = speed_from_belief(q90, args.v_full, args.v_max, args.v_min)
                print(f"belief after weighing: {kalman.mean:.0f} ml, upper 90% {q90:.0f} ml -> speed {speed:.3f} m/s")
                env.fixed_command = torch.tensor([speed, 0.0, 0.0], device=gs.device)
            env.commands[:] = env.fixed_command
            obs, _, dones, _ = env.step(policy(obs))
            if beer is not None:
                # While the beer settles (the first second) the glass wall reaches above the rim like a funnel.
                guarded += liquid.guard_walls(beer, env.robot.get_vel()[0], *glass_pose(), funnel=t < 1.0)
            metrics.update(env, bool(dones[0]))

            # The estimator's features, low-passed as in the calibration: servo sag (target minus measured
            # angle) and load. The filter restarts until the robot has settled after the drop at the start.
            target = env.filtered_actions * env_cfg.action_scale + env.default_dof_pos
            raw = {"sag": (target - env.dof_pos)[0].cpu().numpy(), "load": env.torques[0].cpu().numpy()}
            if t < 1.0:
                pipeline.reset()
            features = pipeline.update(np.concatenate([raw[b] for b in blocks]), env.dt)
            mean, var = estimator.predict(features)
            # Outside the calibration set the GP only returns its prior mean; such frames do not update the
            # belief. Updates run at 5 Hz, the rate of the calibration frames.
            in_range = bool(estimator.in_calibration_range(var)[0])
            out_of_range += int(weighing and not in_range)
            if weighing and in_range and t >= 2.0 and step % 10 == 0:
                kalman.update(float(mean[0]), noise_var, 0.2, float(env.base_ang_vel[0].norm()))
            lo, hi = kalman.interval(0.9)
            tilt_deg = math.degrees(math.asin(min(1.0, float(env.projected_gravity[0, :2].norm()))))
            posterior_log.append((t, kalman.mean, kalman.std, lo, hi, float(mean[0]), tilt_deg, args.volume))
            if beer is not None:
                center, quat = glass_pose()
                surface.append(surface_angle_deg(beer.get_state().pos.reshape(-1, 3), center, quat[0]))

    csv_path = os.path.join("logs", args.exp_name, f"serve_{int(args.volume)}ml_posterior.csv")
    header = "t,mean,std,lo,hi,frame,tilt_deg,true_volume"
    np.savetxt(csv_path, np.array(posterior_log), delimiter=",", fmt="%.4f", header=header, comments="")

    summary = metrics.summary()
    print(f"posterior log: {csv_path}")
    print(f"frames outside the calibration range while weighing: {out_of_range}")
    if beer is not None:
        result = liquid.measure(beer, *glass_pose(), args.grid)
        summary["surface_angle_max_deg"] = float(np.nanmax(surface))
        summary["surface_angle_rms_deg"] = float(np.sqrt(np.nanmean(np.square(surface))))
        print(f"liquid: {result['particles']} particles = {result['volume_ml']:.0f} ml (poured {args.volume:.0f} ml)")
        print(f"spilled: {result['spilled'] * 100:.1f}% of the particles")
        print(f"wall guard: {guarded / len(posterior_log):.2f} particles moved back per control step")
    print("metrics (env 0):\n" + format_summary(summary))


if __name__ == "__main__":
    main()
