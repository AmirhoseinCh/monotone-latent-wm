# Latent World Models with Monotone Planning Costs for Image-Goal Navigation

Official code for **"Latent World Models with Monotone Planning Costs for Image-Goal Navigation"**.

A JEPA-style latent world model plans image-goal navigation with the Cross-Entropy Method (CEM). The planning cost is the cosine distance between the predicted future embedding and the goal embedding. For CEM to converge reliably, this cost has to be **monotone**: it should grow as an action sequence deviates further from the optimal plan. Two training stages enforce this:

1. **Rollout loss**: trains the predictor autoregressively on its own predictions, so training matches multi-step CEM rollouts. The rollout horizon follows a counter-curriculum from K=2 to K=8.
2. **Monotone Cost Ranking (MCR)**: a short fine-tuning stage with a pairwise margin-ranking loss over graded action perturbations. It shapes the cosine cost landscape to be monotone in trajectory deviation.

The model is a frozen DINO-family encoder (DINOv2, DINOv3 or V-JEPA 2) followed by an AdaLN ViT predictor. Planning happens entirely in latent space; no pixels are reconstructed.

<!-- TODO: add teaser figure, paper link and project page after the review period. -->

---

## Repository layout

```
train.py              # training (Base / MCR / TF / ACT / DINO-WM, selected by config)
eval_planning.py      # CEM planning evaluation (AOE / MAOE / ADE / MADE)
configs/              # base.py, mcr.py, tf.py, act.py, dinowm.py
scripts/              # train_base.sh, train_mcr.sh, train_dinowm.sh, eval.sh
deploy/               # standalone robot inference (no TensorFlow)
tools/                # planning / cost-surface visualisations used for the figures
lmwm/
  world_model.py      # LatentWorldModel + rollout, MCR and ACT losses + builders
  encoders/           # frozen DINOv2 / DINOv3 / V-JEPA 2 wrappers
  models/             # AdaLN ViT predictor, DINO-WM ViT predictor
  data/               # RLDS data pipeline (adapted from Octo)
```

## Installation

```bash
conda create -n lmwm python=3.10 -y
conda activate lmwm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Robot deployment only needs `pip install -r requirements-deploy.txt` (no TensorFlow).

**Encoders.** DINOv2 and V-JEPA 2 download automatically through `torch.hub`. DINOv3 weights are gated, so you need to:

1. Clone [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
2. Download `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`.
3. Set these environment variables:

```bash
export DINOV3_REPO=/path/to/dinov3
export DINOV3_CKPT_ROOT=/path/to/dir/containing/the/pth
```

## Data

We train and evaluate on the image portion of GNM, as released in the [CAST dataset](https://huggingface.co/datasets/catglossop/CAST-dataset) (`cast_filtered_dataset`, RLDS format). It has 14,240 training trajectories and a held-out `val` split.

```bash
hf download catglossop/CAST-dataset cast_filtered_dataset.tar.gz --repo-type=dataset --local-dir downloads
mkdir -p data/cast_filtered_dataset
tar -xf downloads/cast_filtered_dataset.tar.gz -C data/cast_filtered_dataset
# -> data/cast_filtered_dataset/cast_filtered_dataset/1.0.0/*.tfrecord
```

The default data root is `./data`. To use another location, pass `--data_dir /path/to/root` to `train.py` or `eval_planning.py`. Action normalization statistics are computed on the first run and cached next to the dataset.

## Training

All paper models use 4 GPUs (batch sizes in the configs are per GPU). Each run writes its checkpoints and a copy of its `config.py` to `checkpoints/<encoder>_<name>_<timestamp>/`.

**Stage 1: Base** (rollout counter-curriculum, 18k steps)

```bash
bash scripts/train_base.sh dino        # or: dinov3
```

**Stage 2: MCR** (+1k steps, resumed from the Base checkpoint)

```bash
bash scripts/train_mcr.sh checkpoints/dino_base_<timestamp>/final_model.pt dino
```

`configs/mcr.py` sets `num_steps = 19000` (18k Base + 1k MCR). For a Base checkpoint saved at a different step, pass `--num_steps <step + 1000>`.

**Ablations and baselines**

```bash
torchrun --nproc_per_node 4 train.py --config configs/tf.py  --encoder dino --name tf    # teacher forcing
torchrun --nproc_per_node 4 train.py --config configs/act.py --encoder dino --name act   # action-contrastive (negative result)
bash scripts/train_dinowm.sh dino                                                        # DINO-WM baseline
```

MCR can be applied to any of them with `scripts/train_mcr.sh`. Pass `--debug` to `train.py` for a quick single-GPU sanity check.

## Evaluation

CEM planning uses a K=6 horizon, 4 context frames, 128 samples × 6 iterations, 4 elites and a unicycle action prior:

```bash
bash scripts/eval.sh checkpoints/dino_mcr_<timestamp>/final_model.pt dino
```

Results go to `eval_results/<run>/eval.json`. `--num_examples N` evaluates a subset.

## Robot deployment

`deploy/infer_robot.py` plans from a single current image and a goal image:

```bash
python deploy/infer_robot.py \
    --checkpoint checkpoints/dino_mcr_<timestamp>/final_model.pt \
    --current_image current.jpg --goal_image goal.jpg \
    --encoder dino --output_plot robot_plan.png
```

See [deploy/README.md](deploy/README.md) for details. In the paper, inference runs on an NVIDIA Jetson AGX Orin and replans at every timestep.

## Results

GNM test split, K=6 horizon, C=4 context. All metrics are lower-is-better.

| Method | AOE (°) | MAOE (°) | ADE (m) | MADE (m) |
|---|---:|---:|---:|---:|
| OmniVLA (img) | 7.73 | 11.58 | 0.94 | 1.72 |
| NoMaD (GC) | 9.09 | 14.20 | **0.88** | 1.67 |
| NWM | 14.57 | 26.57 | 1.04 | 1.72 |
| DINO-WM (DINOv2) | 20.27 | 30.14 | 3.47 | 6.25 |
| DINO-WM (DINOv3) | 21.82 | 32.34 | 2.09 | 3.69 |
| Ours: Base (DINOv3) | 10.15 | 14.62 | 1.22 | 1.97 |
| Ours: MCR (DINOv3) | 8.54 | 12.22 | 1.17 | 1.91 |
| Ours: Base (DINOv2) | 9.12 | 13.29 | 1.12 | 1.83 |
| **Ours: MCR (DINOv2)** | **7.63** | **11.11** | 0.99 | **1.64** |

See the paper for the full ablation over TF / Base / ACT, each with and without MCR.

## Citation

```bibtex
@article{monotone_latent_wm,
  title  = {Latent World Models with Monotone Planning Costs for Image-Goal Navigation},
  author = {Anonymous},
  year   = {2026}
}
```

## License

This code is released under [CC BY-NC 4.0](LICENSE). It contains code adapted from
[jepa-wms](https://github.com/facebookresearch/jepa-wms) (CC BY-NC 4.0),
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) and [Octo](https://github.com/octo-models/octo) (MIT).
See [NOTICE.md](NOTICE.md).
