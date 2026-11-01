"""Vectorised Genesis environment: Mini Pupper 2 walks on flat ground following velocity commands while it
carries something on the tray on its back.

The environment follows the ``VecEnv`` protocol of rsl_rl (``step`` / ``get_observations`` / ``reset``). The
actor only sees what the real robot can measure with its IMU and servos (see ``ObsCfg``).
"""

from __future__ import annotations

import math
from dataclasses import asdict

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, transform_by_quat
from tensordict import TensorDict

from pupper_waiter.config import CUP_X, FOOT_LINKS, JOINT_NAMES, TRAY_TOP_Z, EnvCfg, ObsCfg, RewardCfg
from pupper_waiter.payload import LIQUID_RADIUS


def _uniform(lower: float, upper: float, shape: tuple[int, ...]) -> torch.Tensor:
    """Sample a uniform tensor in ``[lower, upper)`` on the simulation device."""
    return torch.rand(shape, dtype=gs.tc_float, device=gs.device) * (upper - lower) + lower


class PupperEnv:
    """A batch of flat-ground scenes, one Mini Pupper 2 per environment.

    The action (12) is a joint-angle offset around the default pose, executed by a PD position controller that
    stands in for the position-controlled hobby servos.

    Besides the training set-up, the evaluation and demo scripts use a few hooks: ``free_cup`` puts an
    unattached cup on the tray (``{"mass": kg, "friction": float}`` plus optionally ``radius``/``height`` for a
    cylinder, or ``urdf`` for the glass model and its ``material``), ``add_entities`` adds e.g. the liquid
    before the scene is built, and ``fixed_command`` / ``fixed_payload`` pin the command and the payload.
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvCfg,
        obs_cfg: ObsCfg,
        reward_cfg: RewardCfg,
        free_cup: dict | None = None,
        add_entities=None,
        scene_kwargs: dict | None = None,
        robot_material=None,
        floor_material=None,
    ):
        self.num_envs = num_envs
        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.device = gs.device
        self.dt = env_cfg.dt
        self.num_actions = len(JOINT_NAMES)
        self.num_obs = obs_cfg.num_obs
        self.max_episode_length = math.ceil(env_cfg.episode_length_s / self.dt)

        self.free_cup_cfg = free_cup
        if env_cfg.free_cup.enabled and free_cup is None:
            # Training cup; its mass and friction are randomised at every reset.
            c = env_cfg.free_cup
            self.free_cup_cfg = {"mass": c.mass_range[1], "friction": 1.0, "radius": c.radius, "height": c.height}
        self.add_entities = add_entities
        self.scene_kwargs = scene_kwargs or {}
        self.robot_material = robot_material
        self.floor_material = floor_material

        self._build_scene()
        self._init_buffers()
        self._init_rewards()
        self.foot_rest_height = 0.0
        self.reset()
        if self.add_entities is None:
            # Zero of the gait-clearance reward: the height of the foot links once the robot has settled on the
            # floor in its default pose. Skipped when the scene holds extra entities such as the liquid, which
            # these steps would move while reset() restores only the rigid bodies.
            zero = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
            for _ in range(25):
                self.step(zero)
            self.foot_rest_height = float(self.robot.get_links_pos(self.foot_links)[0, :, 2].mean())
            self.reset()

    # ------------------------------------------------------------------ scene construction
    def _build_scene(self) -> None:
        cfg = self.cfg
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=cfg.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # Per-environment gains, masses and friction need batched dof and link info.
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            **self.scene_kwargs,
        )
        self.scene.add_entity(gs.morphs.Plane(), material=self.floor_material)
        # Fixed links are merged into their parent for speed; the payload link stays separate so that its mass
        # properties can be randomised per environment.
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.robot_urdf,
                pos=cfg.base_init_pos,
                quat=cfg.base_init_quat,
                links_to_keep=[cfg.payload.link] if cfg.payload.enabled else [],
            ),
            material=self.robot_material,
        )
        self.cup = None
        if self.free_cup_cfg is not None:
            self._add_cup(self.free_cup_cfg)
        if self.add_entities is not None:
            self.add_entities(self.scene, self)
        self.scene.build(n_envs=self.num_envs)
        c = self.free_cup_cfg
        if self.cup is not None and "urdf" in c and "mass" in c:
            # The glass model is built with the masses in its file; the requested mass is applied here.
            self.cup.set_mass(torch.full((self.num_envs,), float(c["mass"]), device=gs.device))

        self.motor_dofs = [self.robot.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
        self.base_link_idx = self.robot.get_link("base_link").idx_local
        self.foot_links = [self.robot.get_link(n).idx_local for n in FOOT_LINKS]
        self.robot.set_dofs_force_range(
            [-cfg.torque_limit] * self.num_actions, [cfg.torque_limit] * self.num_actions, self.motor_dofs
        )
        self.payload_link = self.robot.get_link(cfg.payload.link) if cfg.payload.enabled else None
        # Nominal mass and inertia of every body link (the payload link is randomised on its own), for the
        # per-episode mass scaling, which then behaves like a density change of the same shape.
        self.body_links_idx = [link.idx_local for link in self.robot.links if link is not self.payload_link]
        self.body_link_masses = [
            (link, float(link.get_mass().reshape(-1)[0])) for link in self.robot.links if link is not self.payload_link
        ]
        if not cfg.base_inertia_duplicate:
            # Drop the duplicated inertial block of the upstream URDF (kept as a near-massless link).
            self.body_link_masses = [
                (link, 1e-4 if link.name == "base_inertia" else nominal) for link, nominal in self.body_link_masses
            ]
        self.body_links_inertia = self.robot.get_links_inertia(self.body_links_idx)[0].clone()

    def _add_cup(self, c: dict) -> None:
        """Add the free cup on the tray: a rigid cylinder of the requested mass, or the glass model."""
        cfg = self.cfg
        height = c.get("height", cfg.free_cup.height)
        self.cup_offset = torch.tensor(
            (CUP_X, 0.0, TRAY_TOP_Z + height / 2 + 0.002), dtype=gs.tc_float, device=gs.device
        )
        if "urdf" in c:
            # The glass model's origin is its floor disk, so it sits on the tray top; the material (particle
            # coupling) comes from the caller.
            self.cup_offset[2] -= height / 2
            cup_pos = tuple(a + b for a, b in zip(cfg.base_init_pos, self.cup_offset.tolist(), strict=True))
            morph = gs.morphs.MJCF if c["urdf"].endswith(".xml") else gs.morphs.URDF
            self.cup = self.scene.add_entity(
                morph(file=c["urdf"], pos=cup_pos),
                material=c.get("material") or gs.materials.Rigid(friction=c["friction"]),
            )
        else:
            radius = c.get("radius", cfg.free_cup.radius)
            cup_pos = tuple(a + b for a, b in zip(cfg.base_init_pos, self.cup_offset.tolist(), strict=True))
            # Contact friction in Genesis is the larger of the two surfaces' values, so the tray gets the same
            # value at reset.
            self.cup = self.scene.add_entity(
                gs.morphs.Cylinder(radius=radius, height=height, pos=cup_pos),
                material=gs.materials.Rigid(rho=c["mass"] / (math.pi * radius**2 * height), friction=c["friction"]),
            )

    def _init_buffers(self) -> None:
        n, a = self.num_envs, self.num_actions
        f, dev = gs.tc_float, self.device
        cfg = self.cfg

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=f, device=dev)
        self.default_dof_pos = torch.tensor(cfg.default_joint_angles, dtype=f, device=dev)
        self.init_base_pos = torch.tensor(cfg.base_init_pos, dtype=f, device=dev)
        self.init_base_quat = torch.tensor(cfg.base_init_quat, dtype=f, device=dev)

        self.obs_buf = torch.zeros((n, self.num_obs), dtype=f, device=dev)
        self.privileged_buf = torch.zeros((n, self.obs_cfg.num_privileged), dtype=f, device=dev)
        self.rew_buf = torch.zeros((n,), dtype=f, device=dev)
        self.reset_buf = torch.ones((n,), dtype=torch.bool, device=dev)
        self.episode_length_buf = torch.zeros((n,), dtype=torch.int32, device=dev)
        self.actions = torch.zeros((n, a), dtype=f, device=dev)
        self.last_actions = torch.zeros((n, a), dtype=f, device=dev)
        self.filtered_actions = torch.zeros((n, a), dtype=f, device=dev)
        self.dof_pos = torch.zeros((n, a), dtype=f, device=dev)
        self.torques = torch.zeros((n, a), dtype=f, device=dev)
        self.base_quat = torch.zeros((n, 4), dtype=f, device=dev)
        self.base_lin_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.base_ang_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.projected_gravity = torch.zeros((n, 3), dtype=f, device=dev)
        self.height = torch.zeros((n,), dtype=f, device=dev)

        # Velocity command and the countdown to its next resample; ``fixed_command`` pins it (demos).
        self.commands = torch.zeros((n, 3), dtype=f, device=dev)
        self.command_timer = torch.zeros((n,), dtype=f, device=dev)
        self.fixed_command: torch.Tensor | None = None
        # Feet: contact, height above the standing height, time in the air.
        self.feet_air_time = torch.zeros((n, len(self.foot_links)), dtype=f, device=dev)
        self.last_contacts = torch.zeros((n, len(self.foot_links)), dtype=torch.bool, device=dev)
        self.feet_contact = torch.zeros((n, len(self.foot_links)), dtype=torch.bool, device=dev)
        self.feet_height = torch.zeros((n, len(self.foot_links)), dtype=f, device=dev)
        # Trot clock: phase in [0, 1); the diagonal pair lf + rb swings in the first half, rf + lb in the second.
        self.gait_phase = torch.zeros((n,), dtype=f, device=dev)
        self.swing_pattern = torch.tensor([[True, False, True, False], [False, True, False, True]], device=dev)

        # Pushes: countdown to the next push, remaining duration and the force vector.
        self.push_timer = torch.zeros((n,), dtype=f, device=dev)
        self.push_remaining = torch.zeros((n,), dtype=f, device=dev)
        self.push_force = torch.zeros((n, 1, 3), dtype=f, device=dev)

        # Payload (pinned per environment by ``fixed_payload`` in evaluations) and the training cup.
        self.payload_mass = torch.zeros((n,), dtype=f, device=dev)
        self.payload_com_z = torch.zeros((n,), dtype=f, device=dev)
        self.fixed_payload: tuple[torch.Tensor, torch.Tensor] | None = None
        self.cup_mass = torch.zeros((n,), dtype=f, device=dev)
        self.cup_friction = torch.ones((n,), dtype=f, device=dev)
        self.total_steps = 0

        # Randomised physics, kept so that a reset only overwrites the rows of the reset environments.
        self.kp = torch.zeros((n, a), dtype=f, device=dev)
        self.kv = torch.zeros((n, a), dtype=f, device=dev)
        self.friction = torch.ones((n, len(self.foot_links)), dtype=f, device=dev)
        self.mass_scale = torch.ones((n,), dtype=f, device=dev)

        self.extras: dict = {}

    def _init_rewards(self) -> None:
        self.reward_functions = {}
        self.reward_scales = {}
        self.episode_sums = {}
        for name, scale in self.reward_cfg.scales.items():
            if name.startswith("cup_") and self.cup is None:
                continue  # a cup-stage configuration evaluated without the training cup
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.reward_scales[name] = scale * self.dt
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)

    # ------------------------------------------------------------------ rsl_rl interface
    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self.obs_buf, "privileged": self.privileged_buf}, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:
        self._reset_idx(torch.ones((self.num_envs,), dtype=torch.bool, device=self.device))
        self._update_state()
        self._update_observation()
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        self.actions = torch.clip(actions, -cfg.clip_actions, cfg.clip_actions)
        delayed = self.last_actions if cfg.simulate_action_latency else self.actions
        self.filtered_actions = (
            cfg.action_filter_alpha * self.filtered_actions + (1 - cfg.action_filter_alpha) * delayed
        )
        target_dof_pos = self.filtered_actions * cfg.action_scale + self.default_dof_pos
        if cfg.max_joint_speed is not None:
            # Servo speed limit: the target may move at most max_joint_speed * dt away from the joint.
            step = cfg.max_joint_speed * self.dt
            target_dof_pos = self.dof_pos + torch.clamp(target_dof_pos - self.dof_pos, -step, step)
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        self._step_commands()
        self._step_gait_clock()
        self._step_pushes()
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()

        self.rew_buf.zero_()
        for name, func in self.reward_functions.items():
            rew = func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        timed_out = self.episode_length_buf > self.max_episode_length
        tilted = torch.norm(self.projected_gravity[:, :2], dim=1) > cfg.max_tilt_gravity_xy
        collapsed = self.height < cfg.min_height
        self.reset_buf = timed_out | tilted | collapsed | self.scene.rigid_solver.get_error_envs_mask()
        if cfg.free_cup.enabled and self.cup is not None:
            self.reset_buf |= self.cup_state()[2]
        self.total_steps += 1
        self.extras["time_outs"] = timed_out.to(gs.tc_float)

        self._reset_idx(self.reset_buf)
        if bool(self.reset_buf.any()):
            self._update_state()
        self._update_observation()

        self.last_actions.copy_(self.actions)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------ commands and disturbances
    def _step_commands(self) -> None:
        self.command_timer -= self.dt
        self._resample_commands(self.command_timer <= 0.0)

    def _resample_commands(self, mask: torch.Tensor) -> None:
        c = self.cfg.command
        n = self.num_envs
        if self.fixed_command is not None:
            new = self.fixed_command.expand(n, 3)
        else:
            new = torch.stack(
                (
                    _uniform(*c.lin_vel_x_range, (n,)),
                    _uniform(*c.lin_vel_y_range, (n,)),
                    _uniform(*c.ang_vel_range, (n,)),
                ),
                dim=1,
            )
            stand = _uniform(0.0, 1.0, (n,)) < c.stand_prob
            new = torch.where(stand[:, None], torch.zeros_like(new), new)
        self.commands = torch.where(mask[:, None], new, self.commands)
        self.command_timer = torch.where(mask, _uniform(*c.resample_interval_s, (n,)), self.command_timer)

    def _step_gait_clock(self) -> None:
        """Advance the trot phase while the command is non-zero; hold phase 0 (all feet down) otherwise."""
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (self.commands[:, 2].abs() > 0.1)
        advanced = (self.gait_phase + self.cfg.command.gait_frequency * self.dt) % 1.0
        self.gait_phase = torch.where(moving, advanced, torch.zeros_like(self.gait_phase))

    def _swing_mask(self) -> torch.Tensor:
        """Boolean (n_envs, 4): which feet the clock expects in the air right now."""
        pair = (self.gait_phase >= 0.5).long()
        return self.swing_pattern[pair] & (self.gait_phase > 0.0)[:, None]

    def _step_pushes(self) -> None:
        """Apply a horizontal force on the body for a short burst at random times."""
        p = self.cfg.push
        n = self.num_envs
        self.push_timer -= self.dt
        start = (self.push_timer <= 0.0) & (self.push_remaining <= 0.0)
        yaw = _uniform(0.0, 2 * math.pi, (n,))
        magnitude = _uniform(*p.force_range, (n,))
        new_force = torch.stack((torch.cos(yaw) * magnitude, torch.sin(yaw) * magnitude, torch.zeros_like(yaw)), 1)
        self.push_force[:, 0, :] = torch.where(start[:, None], new_force, self.push_force[:, 0, :])
        self.push_remaining = torch.where(
            start, torch.full_like(self.push_remaining, p.duration_s), self.push_remaining
        )
        self.push_timer = torch.where(start, _uniform(*p.interval_s, (n,)), self.push_timer)

        active = self.push_remaining > 0.0
        self.push_remaining -= self.dt
        self.robot.apply_links_external_wrench(
            force=self.push_force * active[:, None, None], links_idx_local=[self.base_link_idx]
        )

    # ------------------------------------------------------------------ state and observation
    def _update_state(self) -> None:
        self.base_quat = self.robot.get_quat()
        to_body = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), to_body)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), to_body)
        self.projected_gravity = transform_by_quat(self.global_gravity.expand(self.num_envs, 3), to_body)
        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.torques = self.robot.get_dofs_control_force(self.motor_dofs)
        self.height = self.robot.get_pos()[:, 2]
        # A standing foot carries about 1.7 N.
        self.feet_contact = self.robot.get_links_net_contact_force()[:, self.foot_links, 2] > 0.5
        self.feet_height = self.robot.get_links_pos(self.foot_links)[:, :, 2] - self.foot_rest_height

    def _update_observation(self) -> None:
        noise = self.cfg.obs_noise
        n = self.num_envs
        ang_vel = self.base_ang_vel + _uniform(-noise.ang_vel, noise.ang_vel, (n, 3))
        gravity = self.projected_gravity + _uniform(-noise.gravity, noise.gravity, (n, 3))
        dof_pos = self.dof_pos - self.default_dof_pos + _uniform(-noise.dof_pos, noise.dof_pos, (n, self.num_actions))
        if self.cfg.free_cup.enabled and self.cup is not None:
            slip, tilt, _ = self.cup_state()
            self.privileged_buf[:, 2:4] = slip / 0.05
            self.privileged_buf[:, 4] = tilt
        command_scale = torch.tensor(
            [self.obs_cfg.cmd_lin_vel, self.obs_cfg.cmd_lin_vel, self.obs_cfg.cmd_ang_vel], device=self.device
        )
        clock = 2 * math.pi * self.gait_phase[:, None]
        self.obs_buf = torch.cat(
            (
                ang_vel * self.obs_cfg.ang_vel,
                gravity,
                dof_pos * self.obs_cfg.dof_pos,
                self.actions,
                self.commands * command_scale,
                torch.sin(clock),
                torch.cos(clock),
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, mask: torch.Tensor) -> None:
        """Reset the environments selected by the boolean ``mask`` and resample their randomisation."""
        if not bool(mask.any()):
            return
        n = self.num_envs
        envs_idx = mask.nonzero().squeeze(-1)
        cfg = self.cfg
        dr = cfg.domain_rand

        # Robot: default pose with a small joint perturbation, dropped from its spawn height.
        dof_noise = _uniform(-dr.init_joint_noise, dr.init_joint_noise, (n, self.num_actions))
        qpos = torch.cat(
            (self.init_base_pos.expand(n, 3), self.init_base_quat.expand(n, 4), self.default_dof_pos + dof_noise),
            dim=1,
        )
        self.robot.set_qpos(qpos[envs_idx], envs_idx=envs_idx, zero_velocity=True)
        if self.cup is not None:
            cup_pos = (self.init_base_pos + self.cup_offset).expand(len(envs_idx), 3).clone()
            if cfg.free_cup.enabled:
                cup_pos[:, :2] += _uniform(-cfg.free_cup.offset_noise, cfg.free_cup.offset_noise, (len(envs_idx), 2))
            self.cup.set_pos(cup_pos, envs_idx=envs_idx, zero_velocity=True)
            self.cup.set_quat(self.init_base_quat.expand(len(envs_idx), 4), envs_idx=envs_idx, zero_velocity=True)
        # Two unused draws, left from an earlier version of this environment. They keep the random sequence, so
        # seeded evaluations reproduce the numbers of that version.
        _uniform(0.0, 1.0, (n,))
        _uniform(0.0, 1.0, (n,))

        self._resample_commands(mask)
        self.feet_air_time.masked_fill_(mask[:, None], 0.0)
        self.last_contacts.masked_fill_(mask[:, None], False)
        self.gait_phase.masked_fill_(mask, 0.0)
        self.push_remaining.masked_fill_(mask, 0.0)
        self.push_timer = torch.where(mask, _uniform(*cfg.push.interval_s, (n,)), self.push_timer)

        # Physical randomisation.
        self.kp = torch.where(mask[:, None], _uniform(*dr.kp_range, (n, 1)).expand(n, self.num_actions), self.kp)
        self.kv = torch.where(mask[:, None], _uniform(*dr.kv_range, (n, 1)).expand(n, self.num_actions), self.kv)
        self.friction = torch.where(
            mask[:, None], _uniform(*dr.friction_range, (n, len(self.foot_links))), self.friction
        )
        self.mass_scale = torch.where(mask, _uniform(*dr.mass_scale_range, (n,)), self.mass_scale)
        self.robot.set_dofs_kp(self.kp, self.motor_dofs)
        self.robot.set_dofs_kv(self.kv, self.motor_dofs)
        self.robot.set_friction_ratio(self.friction, self.foot_links)
        if self.cup is not None and not cfg.free_cup.enabled:
            tray_friction = torch.full((n, 1), self.free_cup_cfg["friction"], dtype=gs.tc_float, device=self.device)
            self.robot.set_friction_ratio(tray_friction, [self.base_link_idx])
        if cfg.free_cup.enabled and self.cup is not None:
            self._randomize_cup(mask)
        for link, nominal in self.body_link_masses:
            link.set_mass(nominal * self.mass_scale)
        self.robot.set_links_inertia(
            self.body_links_inertia[None] * self.mass_scale[:, None, None, None], self.body_links_idx
        )
        if self.payload_link is not None:
            self._randomize_payload(mask)

        self.actions.masked_fill_(mask[:, None], 0.0)
        self.last_actions.masked_fill_(mask[:, None], 0.0)
        self.filtered_actions.masked_fill_(mask[:, None], 0.0)
        self.episode_length_buf.masked_fill_(mask, 0)

        # Episode statistics for the logger: mean per-second reward of the finished episodes.
        count = mask.sum()
        self.extras["episode"] = {}
        for name, value in self.episode_sums.items():
            self.extras["episode"]["rew_" + name] = value[mask].sum() / count / cfg.episode_length_s
            value.masked_fill_(mask, 0.0)

    def _randomize_cup(self, mask: torch.Tensor) -> None:
        """Draw the training cup's mass and cup-tray friction for the reset environments."""
        n = self.num_envs
        c = self.cfg.free_cup
        easy = self.total_steps < c.curriculum_steps
        mass_range = c.easy_mass_range if easy else c.mass_range
        friction_range = c.easy_friction_range if easy else c.friction_range
        self.cup_mass = torch.where(mask, _uniform(*mass_range, (n,)), self.cup_mass)
        self.cup_friction = torch.where(mask, _uniform(*friction_range, (n,)), self.cup_friction)
        self.cup.set_mass(self.cup_mass)
        # Contact friction is the larger of the two surfaces' values, so the cup and the tray (part of the base
        # link) get the same ratio.
        self.cup.set_friction_ratio(self.cup_friction[:, None], [0])
        self.robot.set_friction_ratio(self.cup_friction[:, None], [self.base_link_idx])
        slip, tilt, _ = self.cup_state()
        self.privileged_buf = torch.cat(
            (
                torch.zeros((n, 2), dtype=gs.tc_float, device=self.device),
                slip / 0.05,
                tilt[:, None],
                self.cup_mass[:, None] / c.mass_range[1],
                self.cup_friction[:, None],
            ),
            dim=1,
        )

    def _randomize_payload(self, mask: torch.Tensor) -> None:
        """Draw a payload mass and centre-of-mass height for the reset environments (or use the pinned one)."""
        n = self.num_envs
        p = self.cfg.payload
        if self.fixed_payload is None:
            mass = torch.where(mask, _uniform(*p.mass_range, (n,)), self.payload_mass)
            com_z = torch.where(mask, _uniform(*p.com_z_range, (n,)), self.payload_com_z)
        else:
            mass, com_z = self.fixed_payload
        self.payload_mass, self.payload_com_z = mass, com_z
        com = torch.zeros((n, 3), dtype=gs.tc_float, device=self.device)
        com[:, 2] = com_z
        # The payload is modelled as a liquid cylinder of radius LIQUID_RADIUS, twice as tall as its centre of mass;
        # Genesis expects the inertia about the centre of mass.
        height = 2.0 * com_z
        inertia = torch.zeros((n, 3, 3), dtype=gs.tc_float, device=self.device)
        inertia[:, 0, 0] = mass * (3 * LIQUID_RADIUS**2 + height**2) / 12 + 1e-7
        inertia[:, 1, 1] = inertia[:, 0, 0]
        inertia[:, 2, 2] = mass * LIQUID_RADIUS**2 / 2 + 1e-7
        self.payload_link.set_mass(mass)
        self.payload_link.set_COM(com)
        self.payload_link.set_inertia(inertia)
        self.privileged_buf = torch.stack((mass / p.mass_range[1], com_z / p.com_z_range[1]), dim=1)

    # ------------------------------------------------------------------ rewards
    def _reward_level(self) -> torch.Tensor:
        """Gaussian in the squared tilt: 1 when the body is level."""
        tilt_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-tilt_sq / self.reward_cfg.level_sigma)

    def _reward_tilt(self) -> torch.Tensor:
        """Size of the tilt (norm of the gravity direction's horizontal part)."""
        return torch.norm(self.projected_gravity[:, :2], dim=1)

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_lin_vel_z(self) -> torch.Tensor:
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_height(self) -> torch.Tensor:
        return torch.square(self.height - self.reward_cfg.height_target)

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """Gaussian in the error between the commanded and the actual planar velocity (body frame)."""
        err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / self.reward_cfg.tracking_sigma)

    def _reward_feet_air_time(self) -> torch.Tensor:
        """Paid at touchdown for swings longer than the target (as in legged_gym)."""
        contact = self.feet_contact | self.last_contacts  # tolerate one dropped contact frame
        first_contact = (self.feet_air_time > 0.0) & contact
        self.feet_air_time += self.dt
        rew = torch.sum((self.feet_air_time - self.reward_cfg.feet_air_time_target) * first_contact, dim=1)
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for stepping in place when told to stand
        self.feet_air_time *= ~contact
        self.last_contacts = self.feet_contact.clone()
        return rew

    def _reward_gait_clearance(self) -> torch.Tensor:
        """Swing feet (by the clock) lifted to the target height, stance feet down."""
        target = torch.where(self._swing_mask(), self.reward_cfg.swing_height, 0.0)
        err = torch.square(self.feet_height - target)
        walking = (self.gait_phase > 0.0)[:, None]
        return torch.sum(torch.exp(-err / (0.5 * self.reward_cfg.swing_height) ** 2) * walking, dim=1)

    def _reward_gait_contact(self) -> torch.Tensor:
        """Stance feet in contact and swing feet in the air, per the clock."""
        agree = (self.feet_contact != self._swing_mask()).to(gs.tc_float) * 2.0 - 1.0
        return torch.sum(agree * (self.gait_phase > 0.0)[:, None], dim=1)

    def _reward_action_rate(self) -> torch.Tensor:
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_torques(self) -> torch.Tensor:
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_cup_slip(self) -> torch.Tensor:
        slip, _, _ = self.cup_state()
        return torch.sum(torch.square(slip), dim=1)

    def _reward_cup_tilt(self) -> torch.Tensor:
        _, tilt, _ = self.cup_state()
        return torch.square(tilt)

    # ------------------------------------------------------------------ cup
    def cup_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Free cup: (slip on the tray [m] in the tray frame, tilt [rad], fallen) per environment."""
        rel = transform_by_quat(self.cup.get_pos() - self.robot.get_pos(), inv_quat(self.base_quat)) - self.cup_offset
        up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, 3)
        tilt = torch.acos(transform_by_quat(up, self.cup.get_quat())[:, 2].clamp(-1.0, 1.0))
        c = self.cfg.free_cup
        fallen = (tilt > c.fall_tilt) | (rel[:, 2] < -c.fall_drop)
        return rel[:, :2], tilt, fallen

    def cup_rattle(self) -> torch.Tensor:
        """Free cup: horizontal contact force on the cup [N] per environment (it hits the holder)."""
        force = self.cup.get_links_net_contact_force()
        return force[:, :, :2].sum(dim=1).norm(dim=1)

    def config_dict(self) -> dict:
        """Serialisable copy of the configuration, saved next to the checkpoints."""
        return {"env": asdict(self.cfg), "obs": asdict(self.obs_cfg), "reward": asdict(self.reward_cfg)}
