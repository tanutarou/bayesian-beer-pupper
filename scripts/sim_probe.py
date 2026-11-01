"""Calibrate the volume estimator in simulation, in the configuration of the beer-carrying demo.

The walking policy stands still with the glass in the holder while the glass's mass is set to an empty glass
plus a known volume (1 g/ml). Per frame the script records the servo sag (commanded minus measured joint
angle), the servo load (joint torque) and, for comparison, the actor's last hidden layer, low-passed as in
the demo. It then

* compares feature sets frame by frame (leave one volume out),
* splits the held-out error of the shipped features into bias and frame noise for the interval,
* scores the whole method (regression + filter) on held-out volumes,
* and saves the estimator fitted on all volumes to ``logs/<exp>/sim_estimator.npz``.

Usage:
  uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --no-obs-noise    # calibrate
  uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --domain-rand --eval-only --repeats 3

``--eval-only`` records new trajectories (e.g. with observation noise and randomised physics) and scores the
saved estimator on them without refitting. ``--from-saved`` recalibrates from the last recording.
"""

from __future__ import annotations

import argparse
import json
import os

import genesis as gs
import numpy as np
import torch

from pupper_waiter import liquid
from pupper_waiter.env import PupperEnv
from pupper_waiter.estimator import (
    FeaturePipeline,
    GaussianProcessRegression,
    bias_noise_split,
    held_out_weighings,
    leave_one_group_out,
    load_estimator,
    save_estimator,
    score_weighings,
)
from pupper_waiter.runs import load_actor, load_config, make_runner

FEATURE_SETS = {
    "hidden layer": ("hidden",),
    "sag + load": ("sag", "load"),
    "hidden layer + sag + load": ("hidden", "sag", "load"),
}
SHIPPED = FEATURE_SETS["sag + load"]
SAMPLE_HZ = 5.0


def attach_hidden_hook(actor: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Keep the output of the actor's last hidden activation under ``store["h"]`` after every forward pass."""
    store: dict[str, torch.Tensor] = {}
    last = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.ELU)][-1]
    last.register_forward_hook(lambda module, inputs, output: store.__setitem__("h", output.detach()))
    return store


def record(args, volumes_ml: list[float]) -> dict[str, np.ndarray]:
    """Let the robot stand with each volume and record the low-passed features at 5 Hz after it settled."""
    env_cfg, obs_cfg, reward_cfg, rsl_cfg = load_config(args.exp_name)
    env_cfg.robot_urdf = liquid.HOLDER_URDF
    env_cfg.payload.enabled = False
    env_cfg.free_cup.enabled = False
    env_cfg.push.force_range = (0.0, 0.0)
    env_cfg.domain_rand.init_joint_noise = 0.0
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.substeps = args.substeps
    dr = env_cfg.domain_rand
    if not args.domain_rand:
        # One nominal robot: randomised stiffness would be confounded with the payload.
        dr.kp_range = (sum(dr.kp_range) / 2,) * 2
        dr.kv_range = (sum(dr.kv_range) / 2,) * 2
        dr.friction_range = (1.0, 1.0)
        dr.mass_scale_range = (1.0, 1.0)
    if args.no_obs_noise:
        env_cfg.obs_noise.gravity = env_cfg.obs_noise.ang_vel = env_cfg.obs_noise.dof_pos = 0.0

    num_envs = len(volumes_ml) * args.repeats
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = PupperEnv(
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        free_cup={"urdf": liquid.GLASS_MJCF, "mass": 0.05, "friction": 0.5, "height": liquid.GLASS_HEIGHT},
    )
    volumes = torch.tensor(volumes_ml, device=gs.device).repeat_interleave(args.repeats)
    env.cup.set_mass(liquid.GLASS_MASS_KG + volumes / 1000.0)
    env.fixed_command = torch.zeros(3, device=gs.device)
    actor = load_actor(make_runner(env, rsl_cfg, args.exp_name), args.exp_name, args.ckpt)
    hidden = attach_hidden_hook(actor)

    stride = max(1, int(round(1.0 / (SAMPLE_HZ * env.dt))))
    records = {"hidden": [], "sag": [], "load": [], "volume": []}
    pipelines = [FeaturePipeline() for _ in range(num_envs)]
    obs = env.reset()
    with torch.no_grad():
        for step in range(int(args.seconds / env.dt)):
            t = step * env.dt
            obs, _, _, _ = env.step(actor(obs))
            target = env.filtered_actions * env_cfg.action_scale + env.default_dof_pos
            frame = np.concatenate(
                (hidden["h"].cpu().numpy(), (target - env.dof_pos).cpu().numpy(), env.torques.cpu().numpy()), axis=1
            )
            if t < 1.0:
                for pipe in pipelines:
                    pipe.reset()
            filtered = np.stack([pipe.update(frame[i], env.dt) for i, pipe in enumerate(pipelines)])
            # Skip the first two seconds (settling after the drop) and keep 5 frames per second.
            if t < 2.0 or step % stride:
                continue
            h, sag, load = np.split(filtered, [64, 76], axis=1)
            records["hidden"].append(h)
            records["sag"].append(sag)
            records["load"].append(load)
            records["volume"].append(volumes.cpu().numpy())
    return {k: np.concatenate(v) for k, v in records.items()}


def features(data: dict[str, np.ndarray], blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([data[b] for b in blocks], axis=1)


def calibrate(data: dict[str, np.ndarray], num_envs: int, log_dir: str) -> None:
    volume = data["volume"]
    results = {}
    for name, blocks in FEATURE_SETS.items():
        results[name] = leave_one_group_out(features(data, blocks), volume, volume)
        r = results[name]
        print(
            f"frame level, {name:26s}: MAE {r['mae']:.1f} ml, coverage {r['coverage']:.2f}, std {r['pred_std']:.1f} ml"
        )

    x = features(data, SHIPPED)
    bias_var, noise_var = bias_noise_split(x, volume, num_envs)
    held_out = held_out_weighings(x, volume, num_envs, 1.0 / SAMPLE_HZ)
    results["shipped"] = held_out | {"bias_var_ml2": bias_var, "noise_var_ml2": noise_var}
    print(
        f"interval: bias std {np.sqrt(bias_var):.1f} ml, frame noise std {np.sqrt(noise_var):.1f} ml; held-out "
        f"weighings: MAE {held_out['kf_mae_ml']:.1f} ml, error q90 {held_out['kf_q90_ml']:.1f} ml, "
        f"coverage {held_out['coverage']:.2f}, half-width {held_out['half_width_ml']:.0f} ml"
    )
    save_estimator(
        os.path.join(log_dir, "sim_estimator.npz"),
        GaussianProcessRegression().fit(x, volume),
        SHIPPED,
        bias_var,
        noise_var,
    )
    with open(os.path.join(log_dir, "sim_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {log_dir}/sim_estimator.npz and sim_probe.json")


def evaluate(data: dict[str, np.ndarray], num_envs: int, log_dir: str) -> None:
    """Score the saved estimator on new trajectories, as the demo runs it (regression + filter)."""
    estimator, blocks, bias_var, noise_var = load_estimator(os.path.join(log_dir, "sim_estimator.npz"))
    mean, _ = estimator.predict(features(data, blocks))
    n_frames = len(data["volume"]) // num_envs
    truth = data["volume"].reshape(n_frames, num_envs)[0]
    r = score_weighings(mean.reshape(n_frames, num_envs), truth, bias_var, noise_var, 1.0 / SAMPLE_HZ)
    print(
        f"saved estimator on new weighings: MAE {r['kf_mae_ml']:.1f} ml, error q90 {r['kf_q90_ml']:.1f} ml, "
        f"coverage {r['coverage']:.2f}, half-width {r['half_width_ml']:.0f} ml"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--volumes", default="0,15,30,45,60,75,90", help="comma-separated volumes [ml]")
    parser.add_argument("--repeats", type=int, default=4, help="environments per volume")
    parser.add_argument("--seconds", type=float, default=17.0)
    parser.add_argument("--substeps", type=int, default=300, help="as in the liquid demos")
    parser.add_argument("--no-obs-noise", action="store_true")
    parser.add_argument("--domain-rand", action="store_true", help="randomise the physics as in training")
    parser.add_argument("--eval-only", action="store_true", help="score the saved estimator on new trajectories")
    parser.add_argument("--from-saved", action="store_true", help="recalibrate from the last recording")
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.exp_name)
    volumes = [float(v) for v in args.volumes.split(",")]
    num_envs = len(volumes) * args.repeats
    recording = os.path.join(log_dir, "sim_probe_eval.npz" if args.eval_only else "sim_probe.npz")
    if args.from_saved:
        with np.load(recording) as d:
            data = {k: d[k] for k in ("hidden", "sag", "load", "volume")}
    else:
        data = record(args, volumes)
        np.savez(recording, **data)
    if args.eval_only:
        evaluate(data, num_envs, log_dir)
    else:
        calibrate(data, num_envs, log_dir)


if __name__ == "__main__":
    main()
