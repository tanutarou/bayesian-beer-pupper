"""Configuration of the walking tasks.

All physical quantities are SI (metres, radians, seconds, newtons). Joints are ordered leg by leg
(``lf, rf, rb, lb``) and, inside each leg, ``abduction, hip, knee``, as in the URDF.

Three training stages share these dataclasses (see the ``*_config`` functions at the end):

1. ``pretrain``: walking with a stiff, idealised actuator, trained from scratch.
2. ``walk``: the same task on a servo model matched to the Mini Pupper 2 spec sheet, fine-tuned from stage 1.
3. ``cupwalk``: stage 2 with a free cup standing on the tray, fine-tuned from stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

JOINT_NAMES: tuple[str, ...] = (
    "base_lf1",
    "lf1_lf2",
    "lf2_lf3",
    "base_rf1",
    "rf1_rf2",
    "rf2_rf3",
    "base_rb1",
    "rb1_rb2",
    "rb2_rb3",
    "base_lb1",
    "lb1_lb2",
    "lb2_lb3",
)
FOOT_LINKS: tuple[str, ...] = ("lf3", "rf3", "rb3", "lb3")

# Where a cup stands on the tray, relative to the base link origin: the tray top height and the forward offset
# of the cup's centre (the holder of the URDF variants is centred there).
TRAY_TOP_Z = 0.078
CUP_X = -0.055


@dataclass
class PushCfg:
    """Invisible horizontal force bursts on the body at random times."""

    interval_s: tuple[float, float] = (2.0, 5.0)
    duration_s: float = 0.1
    force_range: tuple[float, float] = (0.5, 2.0)


@dataclass
class CommandCfg:
    """Velocity commands (body frame: forward, left, yaw rate) and the gait clock.

    A new command is drawn at random intervals; with probability ``stand_prob`` it is zero, so the policy also
    learns to stand still with the glass on its back. Commands slower than 0.08 m/s are left out because
    standing still nearly satisfies them and the policy would settle for that.
    """

    lin_vel_x_range: tuple[float, float] = (0.08, 0.3)
    lin_vel_y_range: tuple[float, float] = (-0.1, 0.1)
    ang_vel_range: tuple[float, float] = (-0.6, 0.6)
    resample_interval_s: tuple[float, float] = (3.0, 6.0)
    stand_prob: float = 0.1
    # Trot clock, observed as (sin, cos) of its phase and frozen at phase 0 while the command is zero. Without a
    # rhythm in the observation, random exploration of a position-controlled, torque-limited robot never moves
    # the body and the velocity reward gives no gradient away from standing still.
    gait_frequency: float = 2.0


@dataclass
class PayloadCfg:
    """Randomised payload on the tray (a link of the ``_tray`` URDF variant).

    The policy never observes the payload; the critic receives its mass and centre-of-mass height as
    privileged information.
    """

    enabled: bool = True
    link: str = "payload"
    mass_range: tuple[float, float] = (0.005, 0.2)
    com_z_range: tuple[float, float] = (0.02, 0.08)


@dataclass
class FreeCupCfg:
    """Free rigid cup standing on the tray during training (the ``cupwalk`` stage).

    Genesis cannot vary an entity's geometry per environment, so the cup shape is fixed; mass, cup-tray friction
    and the placement on the tray are randomised per episode. Training starts with heavy, grippy cups and
    widens the ranges after ``curriculum_steps`` environment steps.
    """

    enabled: bool = False
    radius: float = 0.02
    height: float = 0.06
    mass_range: tuple[float, float] = (0.05, 0.2)
    friction_range: tuple[float, float] = (0.3, 0.8)
    easy_mass_range: tuple[float, float] = (0.15, 0.2)
    easy_friction_range: tuple[float, float] = (0.6, 0.8)
    curriculum_steps: int = 2400  # 100 PPO iterations x 24 steps
    offset_noise: float = 0.005
    # The episode ends when the cup tips past this angle [rad] or drops this far below its resting height [m].
    fall_tilt: float = 0.785
    fall_drop: float = 0.02


@dataclass
class DomainRandCfg:
    """Per-episode physical randomisation. ``kp``/``kv`` are the servo stiffness and damping of the PD model."""

    kp_range: tuple[float, float] = (20.0, 40.0)
    kv_range: tuple[float, float] = (0.1, 0.3)
    friction_range: tuple[float, float] = (0.5, 1.5)
    mass_scale_range: tuple[float, float] = (0.85, 1.25)
    init_joint_noise: float = 0.1


@dataclass
class ObsNoiseCfg:
    """Additive uniform noise on the observations."""

    gravity: float = 0.03
    ang_vel: float = 0.2
    dof_pos: float = 0.02


@dataclass
class EnvCfg:
    """Robot, actuator model, randomisation and termination. The defaults are the spec-sheet servo model."""

    robot_urdf: str = "assets/mini_pupper_2/mini_pupper_2_tray.urdf"
    # Standing pose (abduction, hip, knee per leg), the zero of the action. Taller than the crouched start-up
    # pose of the real robot: crouched, the stance legs of a trot need more knee torque than the servos have.
    default_joint_angles: tuple[float, ...] = (0.0, 0.55, -1.0) * 4
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.14)
    base_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    dt: float = 0.02  # 50 Hz control
    substeps: int = 4
    episode_length_s: float = 15.0
    # target joint angle = default + action * action_scale
    action_scale: float = 0.25
    clip_actions: float = 4.0
    # Servo model: torque limit [N m], one control cycle of command latency, a first-order low-pass on the
    # executed action, and a speed limit [rad/s] on how fast the joint target may move (None: unlimited).
    torque_limit: float = 0.34
    simulate_action_latency: bool = True
    action_filter_alpha: float = 0.6
    max_joint_speed: float | None = 10.0
    # The upstream xacro gives base_link and base_inertia the same 0.175 kg inertial block. False drops the
    # duplicate; the pretraining stage was trained with it.
    base_inertia_duplicate: bool = False
    # Termination: tilt (norm of the gravity direction's horizontal part, ~45 degrees) and body height [m].
    max_tilt_gravity_xy: float = 0.7
    min_height: float = 0.035
    push: PushCfg = field(default_factory=PushCfg)
    command: CommandCfg = field(default_factory=CommandCfg)
    payload: PayloadCfg = field(default_factory=PayloadCfg)
    free_cup: FreeCupCfg = field(default_factory=FreeCupCfg)
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
    obs_noise: ObsNoiseCfg = field(default_factory=ObsNoiseCfg)


@dataclass
class ObsCfg:
    """Observation layout and scaling.

    Actor (35): body angular velocity (3), gravity direction in the body frame (3), joint angles relative to the
    default pose (12), previous action (12), velocity command (3), gait clock (2). Each group is scaled to be of
    order one. Critic only: payload mass and centre-of-mass height (2); the cup stage adds the cup's slip (2),
    tilt, mass and friction.
    """

    ang_vel: float = 0.25
    dof_pos: float = 1.0
    cmd_lin_vel: float = 2.0
    cmd_ang_vel: float = 0.25
    num_obs: int = 35
    num_privileged: int = 2


@dataclass
class RewardCfg:
    """Reward weights; each term is multiplied by ``dt``, so the values are per-second rates."""

    # Width of the Gaussian level reward on the squared tilt (0.04: reward 1/e at about 11 degrees).
    level_sigma: float = 0.04
    height_target: float = 0.10
    # Width of the Gaussian velocity-tracking rewards, matched to this robot's speeds: with the 0.25 of a
    # Go2-sized robot, standing still already earns most of the tracking reward and the policy never walks.
    tracking_sigma: float = 0.02
    feet_air_time_target: float = 0.2
    swing_height: float = 0.02
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Keep the tray level: a Gaussian term for precision near level, a linear term that still pulls
            # back from a large tilt, and a penalty on rocking.
            "level": 1.0,
            "tilt": -0.5,
            "ang_vel_xy": -0.05,
            "lin_vel_z": -1.0,
            "height": -20.0,
            "tracking_lin_vel": 4.0,
            "tracking_ang_vel": 1.0,
            "feet_air_time": 1.0,
            # Terms that ask the feet to follow the gait clock. Rewarding them made the policy hop in place, so
            # they are off; the clock stays in the observation as a rhythm reference.
            "gait_clearance": 0.0,
            "gait_contact": 0.0,
            "action_rate": -0.02,
            "similar_to_default": -0.02,
            "torques": -0.01,
        }
    )


@dataclass
class TrainCfg:
    """PPO hyper-parameters for rsl_rl's OnPolicyRunner."""

    num_envs: int = 4096
    max_iterations: int = 600
    num_steps_per_env: int = 24
    save_interval: int = 50
    seed: int = 1
    init_std: float = 0.8

    def to_rsl_rl(self, run_name: str) -> dict:
        """Build the nested dictionary that rsl_rl expects."""
        return {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "entropy_coef": 0.002,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [256, 128, 64],
                "activation": "elu",
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": self.init_std,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [256, 128, 64],
                "activation": "elu",
            },
            "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
            "num_steps_per_env": self.num_steps_per_env,
            "save_interval": self.save_interval,
            "run_name": run_name,
            "logger": "tensorboard",
        }


def pretrain_config() -> tuple[EnvCfg, ObsCfg, RewardCfg]:
    """Stage 1: walking with a stiff, idealised actuator (like the Genesis Go2 example).

    On the spec-sheet servo model a policy trained from scratch learns to stand but barely walks, so the
    walking stages start from a policy trained with this actuator: fixed stiff gains, no latency or filter,
    a 5 N m torque cap, no speed limit.
    """
    env_cfg, obs_cfg, reward_cfg = EnvCfg(), ObsCfg(), RewardCfg()
    env_cfg.torque_limit = 5.0
    env_cfg.simulate_action_latency = False
    env_cfg.action_filter_alpha = 0.0
    env_cfg.max_joint_speed = None
    env_cfg.base_inertia_duplicate = True
    env_cfg.domain_rand.kp_range = (20.0, 20.0)
    env_cfg.domain_rand.kv_range = (0.5, 0.5)
    return env_cfg, obs_cfg, reward_cfg


def walk_config() -> tuple[EnvCfg, ObsCfg, RewardCfg]:
    """Stage 2: walking on the servo model matched to the Mini Pupper 2 spec sheet.

    Torque cap 0.34 N m, stiffness 20-40 and damping 0.1-0.3, a no-load speed of about 10 rad/s (assumed; the
    spec sheet gives none), one control cycle of latency and the action low-pass. These are the defaults.
    """
    return EnvCfg(), ObsCfg(), RewardCfg()


def cupwalk_config() -> tuple[EnvCfg, ObsCfg, RewardCfg]:
    """Stage 3: stage 2 with a free cup on the bare tray.

    The actor's observation is unchanged; the critic gets the cup state, an episode ends when the cup falls,
    and two penalties ask for little slip and tilt.
    """
    env_cfg, obs_cfg, reward_cfg = walk_config()
    env_cfg.robot_urdf = "assets/mini_pupper_2/mini_pupper_2_tray_bare.urdf"
    env_cfg.payload.enabled = False
    env_cfg.free_cup.enabled = True
    obs_cfg.num_privileged = 7
    reward_cfg.scales.update({"cup_slip": -5.0, "cup_tilt": -2.0})
    return env_cfg, obs_cfg, reward_cfg


TASKS = {"pretrain": pretrain_config, "walk": walk_config, "cupwalk": cupwalk_config}
