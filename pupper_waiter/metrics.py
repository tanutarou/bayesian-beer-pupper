"""Rollout metrics shared by the evaluation and demo scripts.

The quantities follow what liquid-transport papers report: horizontal acceleration and jerk of the carrier
(what sloshes the liquid), body orientation, falls, plus simple gait statistics (duty factor, step rate,
body height) that tell whether the policy walks differently with a glass on its back.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class RolloutMetrics:
    """Accumulate per-step state of environment 0 and summarise it at the end of a rollout."""

    def __init__(self, dt: float):
        self.dt = dt
        self.vel: list[list[float]] = []  # world-frame base velocity
        self.ang_vel: list[list[float]] = []  # body-frame angular velocity
        self.gravity: list[list[float]] = []  # projected gravity (body frame)
        self.height: list[float] = []
        self.contacts: list[list[bool]] = []
        self.falls = 0
        self.cup_slip: list[float] = []
        self.cup_tilt: list[float] = []
        self.cup_fallen = False
        self.cup_rattles = 0

    def update(self, env, done: bool) -> None:
        self.vel.append(env.robot.get_vel()[0].tolist())
        self.ang_vel.append(env.base_ang_vel[0].tolist())
        self.gravity.append(env.projected_gravity[0].tolist())
        self.height.append(float(env.height[0]))
        self.contacts.append(env.feet_contact[0].tolist())
        self.falls += int(done)
        if env.cup is not None:
            slip, tilt, fallen = env.cup_state()
            self.cup_slip.append(float(slip[0].norm()))
            self.cup_tilt.append(math.degrees(float(tilt[0])))
            self.cup_fallen |= bool(fallen[0])
            # A sideways contact force on the cup means it hit the holder ring (or something else sideways).
            self.cup_rattles += int(float(env.cup_rattle()[0]) > 0.2)

    def summary(self) -> dict[str, float]:
        v = np.array(self.vel)[:, :2]
        acc = np.diff(v, axis=0) / self.dt
        jerk = np.diff(acc, axis=0) / self.dt
        g = np.array(self.gravity)
        # Roll about the body x axis, pitch about y, from the gravity direction in the body frame.
        roll = np.degrees(np.arctan2(g[:, 1], -g[:, 2]))
        pitch = np.degrees(np.arctan2(-g[:, 0], np.hypot(g[:, 1], g[:, 2])))
        tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(g[:, :2], axis=1), 0.0, 1.0)))
        w = np.array(self.ang_vel)
        out = {
            "tilt_rms_deg": float(np.sqrt(np.mean(tilt**2))),
            "tilt_max_deg": float(tilt.max()),
            "roll_mean_abs_deg": float(np.mean(np.abs(roll))),
            "roll_max_abs_deg": float(np.abs(roll).max()),
            "pitch_mean_abs_deg": float(np.mean(np.abs(pitch))),
            "pitch_max_abs_deg": float(np.abs(pitch).max()),
            "ang_vel_xy_rms_rad_s": float(np.sqrt(np.mean(np.sum(w[:, :2] ** 2, axis=1)))),
            "acc_xy_max_m_s2": float(np.linalg.norm(acc, axis=1).max()) if len(acc) else 0.0,
            # The 90th percentile is less at the mercy of a single disturbance than the maximum.
            "acc_xy_p90_m_s2": float(np.percentile(np.linalg.norm(acc, axis=1), 90)) if len(acc) else 0.0,
            "jerk_xy_max_m_s3": float(np.linalg.norm(jerk, axis=1).max()) if len(jerk) else 0.0,
            "height_mean_m": float(np.mean(self.height)),
            "falls": float(self.falls),
        }
        if self.contacts:
            c = np.array(self.contacts, dtype=bool)
            touchdowns = np.sum(c[1:] & ~c[:-1])  # contact events over all feet
            seconds = len(c) * self.dt
            out["duty_factor"] = float(c.mean())
            out["step_rate_hz"] = float(touchdowns / 4 / seconds)
        if self.cup_slip:
            out["cup_slip_max_m"] = float(max(self.cup_slip))
            out["cup_tilt_max_deg"] = float(max(self.cup_tilt))
            out["cup_fallen"] = float(self.cup_fallen)
            out["cup_rattle_steps"] = float(self.cup_rattles)
        return out


def surface_angle_deg(pos: torch.Tensor, cup_center: torch.Tensor, cup_quat: torch.Tensor) -> float:
    """Tilt of the free surface relative to the cup [deg].

    The surface is sampled as the highest particle in each cell of a small horizontal grid over the cup and a
    plane is fitted through those samples. A level surface in the world shows up as ``-tilt`` of the cup, so the
    number is what a camera looking at the glass sees; it is also the quantity that slosh-free planners bound.
    """
    from genesis.utils.geom import inv_quat, transform_by_quat

    rel = transform_by_quat(pos - cup_center, inv_quat(cup_quat).expand(pos.shape[0], 4))
    rel = rel[rel[:, :2].norm(dim=1) < 0.03]
    if rel.shape[0] < 20:
        return float("nan")
    cells, radius = 6, 0.022
    ij = ((rel[:, :2] + radius) / (2 * radius) * cells).long().clamp(0, cells - 1)
    key = ij[:, 0] * cells + ij[:, 1]
    top = torch.full((cells * cells,), -1.0, device=rel.device).scatter_reduce(0, key, rel[:, 2], "amax")
    keep = top > -1.0
    centers = (torch.arange(cells, device=rel.device).float() + 0.5) / cells * 2 * radius - radius
    xy = torch.cartesian_prod(centers, centers)[keep]
    design = torch.cat((xy, torch.ones((xy.shape[0], 1), device=rel.device)), dim=1)
    coef = torch.linalg.lstsq(design, top[keep, None]).solution.squeeze()
    return math.degrees(math.atan(math.hypot(float(coef[0]), float(coef[1]))))


def format_summary(summary: dict[str, float]) -> str:
    """One line per metric, aligned, for the console."""
    width = max(len(k) for k in summary)
    return "\n".join(f"  {k:<{width}}  {v:8.3f}" for k, v in summary.items())
