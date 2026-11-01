# Bayesian Beer Pupper

**English** | [日本語](README.ja.md)

A Mini Pupper 2 that carries a glass of beer carefully: it estimates how much beer it carries, with
uncertainty, and walks slowly enough not to spill it. Everything runs in the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator; the beer is an MPM (Material Point
Method) liquid.

- **Walk**: a walking policy that follows velocity commands with a glass on its back, trained with PPO
  ([rsl_rl](https://github.com/leggedrobotics/rsl_rl)). The servo settings follow the Mini Pupper 2 spec sheet
  (0.34 N·m torque limit, one control step of latency).
- **Weigh**: standing still, the robot estimates the volume of beer from how much its servos sag and how hard
  they work, with an uncertainty interval.
- **Carry**: the walking speed is chosen from the upper 90th percentile of the estimate, so a glass that might
  be full is carried slowly.

## Setup

Requires Linux, an NVIDIA GPU with a CUDA 12.8 capable driver, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tanutarou/bayesian-beer-pupper.git
cd bayesian-beer-pupper
uv sync
```

## Usage

No trained weights are included, so start with training. Results go to `logs/<experiment>/`. Training refuses
to start if that directory already exists, so earlier results are never overwritten. Times are for an
RTX 5060 Ti.

### 1. Train

Three stages: a policy pretrained with an idealised actuator, fine-tuned with the spec-sheet servo model, and
fine-tuned again with an unattached cup on the tray.

```bash
uv run python scripts/train.py -e pretrain --task pretrain                                   # 600 iterations, ~30 min
uv run python scripts/train.py -e level-walker --task walk --resume logs/pretrain/model_599.pt --max-iterations 300
uv run python scripts/train.py -e plain-walker --task walk --resume logs/pretrain/model_599.pt --max-iterations 300 \
    --reward level=0 tilt=0 ang_vel_xy=0                                                    # no levelling reward (baseline)
uv run python scripts/train.py -e cup-walker --task cupwalk --resume logs/level-walker/model_898.pt --max-iterations 300
```

### 2. Evaluate walking with a weight or a cup

```bash
uv run python scripts/eval.py -e cup-walker --ckpt 299 --volume 100                # 105 g weight fixed to the tray
uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --seed 1          # 100 g cup on the bare tray
uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --holder --seed 1 # the cup in a 35 mm deep holder
```

The other policies are `-e level-walker --ckpt 898` and `-e plain-walker --ckpt 898`.

### 3. Carry liquid beer

62 ml of beer is poured into the glass in the holder, settles for 3 s, and the robot walks for 12 s
(15–20 min per run). It prints the spilled fraction and the slosh of the surface.

```bash
uv run python scripts/demo_beer.py -e cup-walker --ckpt 299 --volume 62 --command 0.05 0 0 --seed 1
```

### 4. Calibrate the beer volume estimator

The robot stands with the glass in the holder while the glass's mass is set to 0–90 ml of beer. The estimator
is saved to `logs/cup-walker/sim_estimator.npz`.

```bash
uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --no-obs-noise                        # calibrate (~30 min)
uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --domain-rand --eval-only --repeats 3 # score with noise
```

### 5. Estimate the beer volume, then carry

The robot weighs the glass for 6 s standing still, then walks for 30 s at the speed chosen from the estimate
(~45 min). The estimate at every step is written to `logs/cup-walker/serve_62ml_posterior.csv`.
`--no-liquid` runs the same with an empty glass of the same mass, in about a minute.

```bash
uv run python scripts/demo_serve.py -e cup-walker --ckpt 299 --volume 62 --seed 1
```

Rigid-body runs give the same numbers for the same `--seed`; runs with liquid vary slightly between runs.

## Layout

```
pupper_waiter/   config.py (the three training stages), env.py (environment), estimator.py (volume estimator),
                 liquid.py (glass and liquid), metrics.py, payload.py, runs.py (loading trained runs)
scripts/         train.py, eval.py, demo_beer.py, sim_probe.py, demo_serve.py
assets/          Mini Pupper 2 robot models (with a tray or a holder), meshes, and the glass for the liquid
```

## License

MIT (see [LICENSE](LICENSE)). The Mini Pupper 2 robot model in `assets/mini_pupper_2/` is derived from
[mini_pupper_ros](https://github.com/mangdangroboticsclub/mini_pupper_ros) by MangDang and is distributed under
the Apache License 2.0; see [assets/mini_pupper_2/NOTICE](assets/mini_pupper_2/NOTICE) for the source and the
changes made.
