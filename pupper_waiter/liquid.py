"""Beer in the glass: an MPM liquid in the free glass model, which stands in the holder on the tray.

Shared by ``scripts/demo_beer.py`` and ``scripts/demo_serve.py``. The liquid is Genesis's MPM liquid, two-way
coupled with the glass only (the robot, tray and holder do not touch the particles). Three numerical measures
keep it in the glass; their size is reported next to every result:

* the glass has thick liquid-only shells (``free_glass.xml``), because the particle-grid coupling cannot hold a
  wall thinner than a grid cell;
* the coupling softness is small (0.002 m), since larger values turn the beer into jelly;
* ``guard_walls`` puts particles that sank into the floor or the wall back inside once per control step.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from pupper_waiter.config import CUP_X, TRAY_TOP_Z, EnvCfg

HOLDER_URDF = "assets/mini_pupper_2/mini_pupper_2_tray_holder_glass.urdf"
GLASS_MJCF = "assets/mini_pupper_2/free_glass.xml"
GLASS_MASS_KG = 0.05
GLASS_HEIGHT = 0.06
CUP_INNER_RADIUS = 0.022
LIQUID_RADIUS = CUP_INNER_RADIUS - 0.002  # the poured column starts 2 mm inside the wall
COUPLING_SOFTNESS = 0.002


def glass(mass_kg: float, friction: float = 0.5) -> dict:
    """The ``free_cup`` argument of ``PupperEnv`` for the glass model coupled with the liquid."""
    return {
        "urdf": GLASS_MJCF,
        "mass": mass_kg,
        "friction": friction,
        "height": GLASS_HEIGHT,
        "material": gs.materials.Rigid(
            needs_coup=True, coup_friction=0.1, coup_softness=COUPLING_SOFTNESS, friction=friction
        ),
    }


def beer_adder(env_cfg: EnvCfg, volume_ml: float, entities: dict):
    """Callback for ``PupperEnv(add_entities=...)`` that adds the beer as a column standing in the glass.

    The entity is stored as ``entities["beer"]``.
    """
    base = np.array(env_cfg.base_init_pos)
    height = volume_ml * 1e-6 / (math.pi * LIQUID_RADIUS**2)
    floor_z = TRAY_TOP_Z + 0.002 + 0.001  # glass floor: 1 mm above the tray plus the 2 mm floor disk
    pos = (base[0] + CUP_X, base[1], base[2] + floor_z + 0.003 + height / 2)

    def add(scene, env) -> None:
        entities["beer"] = scene.add_entity(
            material=gs.materials.MPM.Liquid(rho=1000.0, sampler="regular"),
            morph=gs.morphs.Cylinder(pos=pos, radius=LIQUID_RADIUS, height=height),
        )

    return add


def mpm_scene_kwargs(env_cfg: EnvCfg, speed: float, seconds: float, grid: int) -> dict:
    """``scene_kwargs`` of ``PupperEnv``: the MPM domain along the robot's straight path.

    Particles that reach the domain boundary stick to it, so the domain reaches down to the floor (spilled beer
    falls and lands) and covers the path of the walking robot with a margin.
    """
    x0, y0, z0 = env_cfg.base_init_pos
    margin = 0.35
    lower = (x0 + min(0.0, speed * seconds) - margin, y0 - margin, 0.0)
    upper = (x0 + max(0.0, speed * seconds) + margin, y0 + margin, z0 + 0.3)
    return {"mpm_options": gs.options.MPMOptions(grid_density=grid, lower_bound=lower, upper_bound=upper)}


def to_glass_frame(points: torch.Tensor, center: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Express world points in the glass frame (origin at the floor centre, z along the glass axis)."""
    return transform_by_quat(points - center, inv_quat(quat).expand(points.shape[0], 4))


def pour(beer, center: torch.Tensor, quat: torch.Tensor) -> float:
    """Start the beer packed into the glass instead of as a column taller than it; returns the height scale.

    The regular particle sampling is sparser than the settled liquid, so a column with the right amount of beer
    stands above the rim. The particles' heights above the floor are scaled so that the column starts 8 mm
    below the rim; the MPM state starts unstressed, so the liquid then simply settles.
    """
    floor, top = 0.001, GLASS_HEIGHT - 0.008
    rel = to_glass_frame(beer.get_particles_pos()[0], center, quat)
    scale = min(1.0, (top - floor) / max(float(rel[:, 2].max()) - floor, 1e-6))
    if scale < 1.0:
        rel[:, 2] = floor + (rel[:, 2] - floor) * scale
        beer.set_particles_pos(center + transform_by_quat(rel, quat.expand(rel.shape[0], 4)))
    return scale


def guard_walls(beer, body_velocity: torch.Tensor, center: torch.Tensor, quat: torch.Tensor, funnel: bool) -> int:
    """Put particles that sank into the glass's floor or wall back inside; returns how many were moved.

    The particle-rigid coupling acts on velocities only and never corrects positions, so when the glass bounces
    in the holder, as it does at every step of a gait, the liquid lags the floor and creeps into it and into the
    wall. Particles below the floor are placed on it, particles below the rim but outside the inner wall just
    inside it, both with the body's velocity. Only beer that goes over the rim can leave the glass. While
    filling (``funnel``), the wall reaches 4 cm above the rim, so the settling column cannot overflow.
    """
    rel = to_glass_frame(beer.get_particles_pos()[0], center, quat)
    radial = rel[:, :2].norm(dim=1)
    below = (rel[:, 2] < 0.0) & (radial < CUP_INNER_RADIUS + 0.005)
    rim = GLASS_HEIGHT + (0.04 if funnel else 0.0)
    in_wall = (rel[:, 2] >= 0.0) & (rel[:, 2] < rim) & (radial > CUP_INNER_RADIUS)
    idx = (below | in_wall).nonzero().squeeze(-1)
    if idx.numel() == 0:
        return 0
    fixed = rel[idx].clone()
    fixed[:, 2] = torch.where(fixed[:, 2] < 0.0, torch.full_like(fixed[:, 2], 0.0005), fixed[:, 2])
    r = fixed[:, :2].norm(dim=1)
    fixed[:, :2] *= torch.where(
        r > CUP_INNER_RADIUS, (CUP_INNER_RADIUS - 0.0005) / r.clamp(min=1e-6), torch.ones_like(r)
    )[:, None]
    beer.set_particles_pos(center + transform_by_quat(fixed, quat.expand(idx.numel(), 4)), particles_idx_local=idx)
    beer.set_particles_vel(body_velocity.expand(idx.numel(), 3), particles_idx_local=idx)
    return int(idx.numel())


def measure(beer, center: torch.Tensor, quat: torch.Tensor, grid: int) -> dict[str, float]:
    """Spilled fraction (particles that left the glass sideways), particle volume and settled liquid level."""
    rel = to_glass_frame(beer.get_state().pos.reshape(-1, 3), center, quat)
    outside = rel[:, :2].norm(dim=1) > CUP_INNER_RADIUS + 0.01
    particle_size = 0.01 * 64.0 / grid  # the regular sampler places particles on this spacing
    # Level: 98% quantile of the particle heights, so that a few stray particles do not count.
    level = float(torch.quantile(rel[~outside, 2], 0.98)) if (~outside).any() else float("nan")
    return {
        "particles": int(rel.shape[0]),
        "volume_ml": rel.shape[0] * particle_size**3 * 1e6,
        "spilled": float(outside.float().mean()),
        "level_m": level,
    }
