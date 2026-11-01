# Bayesian Beer Pupper

[English](README.md) | **日本語**

不確かさ推定を用いたビールを大切に運ぶ犬ロボットです。犬型ロボット Mini Pupper 2 が、背中に載せたビールの量を
不確かさつきで量り、こぼさない速さで運びます。すべて [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
のシミュレーションで、ビールは MPM (Material Point Method) の液体です。

- **歩く**: グラスを載せたまま速度指令に従って歩く方策を、強化学習 (PPO、[rsl_rl](https://github.com/leggedrobotics/rsl_rl)) で学習します。サーボは Mini Pupper 2 のスペック表に合わせた設定 (トルク上限 0.34 N·m、1 制御周期の遅れ) です。
- **量る**: 立ったまま、サーボの沈みと負荷からビールの量を不確かさつきで推定します。
- **運ぶ**: 推定の上側 90% percentile から歩く速さを決めます。満杯かもしれなければゆっくり運びます。

## セットアップ

Linux、NVIDIA GPU (CUDA 12.8 対応ドライバ)、[uv](https://docs.astral.sh/uv/) が必要です。

```bash
git clone https://github.com/tanutarou/bayesian-beer-pupper.git
cd bayesian-beer-pupper
uv sync
```

## 使い方

学習済みの重みは含めていないので、学習から順に実行します。結果は `logs/<実験名>/` に入ります。
同じ名前のディレクトリがすでにあると学習は始まらないので、前の結果を上書きすることはありません。
時間は RTX 5060 Ti での目安です。

### 1. 学習

3 段階です。理想化したアクチュエータで事前学習した方策を、スペック表のサーボ模型で追加学習し、
さらに固定していないカップを載せて追加学習します。

```bash
uv run python scripts/train.py -e pretrain --task pretrain                                   # 600 反復、約 30 分
uv run python scripts/train.py -e level-walker --task walk --resume logs/pretrain/model_599.pt --max-iterations 300
uv run python scripts/train.py -e plain-walker --task walk --resume logs/pretrain/model_599.pt --max-iterations 300 \
    --reward level=0 tilt=0 ang_vel_xy=0                                                    # 水平保持なし (比較用)
uv run python scripts/train.py -e cup-walker --task cupwalk --resume logs/level-walker/model_898.pt --max-iterations 300
```

### 2. 重りやカップを載せて歩く評価

```bash
uv run python scripts/eval.py -e cup-walker --ckpt 299 --volume 100                # トレイに固定した 105 g の重り
uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --seed 1          # 裸のトレイに置いた 100 g のカップ
uv run python scripts/eval.py -e cup-walker --ckpt 299 --cup 100 --holder --seed 1 # 深さ 35 mm のホルダーに置いたカップ
```

ほかの方策は `-e level-walker --ckpt 898`、`-e plain-walker --ckpt 898` です。

### 3. 液体のビールを運ぶ

ホルダーのグラスにビールを 62 ml 注ぎ、3 s 落ち着かせてから 12 s 歩かせます (1 本 15〜20 分)。
こぼれた割合と液面の揺れを表示します。

```bash
uv run python scripts/demo_beer.py -e cup-walker --ckpt 299 --volume 62 --command 0.05 0 0 --seed 1
```

### 4. ビール量の推定器の calibration

ホルダーにグラスを置いて立たせ、グラスの質量を 0〜90 ml 相当で変えて calibration します。
推定器は `logs/cup-walker/sim_estimator.npz` に保存されます。

```bash
uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --no-obs-noise                        # calibration (約 30 分)
uv run python scripts/sim_probe.py -e cup-walker --ckpt 299 --domain-rand --eval-only --repeats 3 # ノイズを入れて評価
```

### 5. ビール量を推定してから運ぶ

6 s 立って量り、推定から決めた速さで 30 s 歩きます (約 45 分)。毎制御周期の推定は
`logs/cup-walker/serve_62ml_posterior.csv` に書き出します。`--no-liquid` を付けると、同じ質量の空のグラスで
1 分ほどで試せます。

```bash
uv run python scripts/demo_serve.py -e cup-walker --ckpt 299 --volume 62 --seed 1
```

剛体だけの実行は、同じ `--seed` なら同じ数値になります。液体を含む実行は、実行ごとに少し変わります。

## 構成

```
pupper_waiter/   config.py (学習の 3 段階の設定), env.py (学習環境), estimator.py (量の推定器),
                 liquid.py (グラスと液体), metrics.py, payload.py, runs.py (学習結果の読み込み)
scripts/         train.py, eval.py, demo_beer.py, sim_probe.py, demo_serve.py
assets/          Mini Pupper 2 のロボットモデル (トレイ・ホルダー付き)、メッシュ、液体用のグラス
```

## ライセンス

MIT ([LICENSE](LICENSE))。`assets/mini_pupper_2/` の Mini Pupper 2 のロボットモデルは、MangDang の
[mini_pupper_ros](https://github.com/mangdangroboticsclub/mini_pupper_ros) から派生したもので、Apache License 2.0
で配布しています。出典と変更点は [assets/mini_pupper_2/NOTICE](assets/mini_pupper_2/NOTICE) にあります。
