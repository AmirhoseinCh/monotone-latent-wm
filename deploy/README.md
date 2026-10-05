# Robot deployment

`infer_robot.py` runs CEM planning from a single current camera image and a goal image, with the trained world model as the dynamics function. It needs no TensorFlow and no dataset. Only `torch` and the packages in `requirements-deploy.txt` are required.

## Setup

```bash
pip install torch torchvision          # matching your CUDA / Jetson build
pip install -r requirements-deploy.txt
```

To use DINOv3, set `DINOV3_REPO` and `DINOV3_CKPT_ROOT` as described in the main README.

## Run

```bash
python deploy/infer_robot.py \
    --checkpoint /path/to/final_model.pt \
    --current_image current.jpg \
    --goal_image goal.jpg \
    --encoder dino \
    --output_plot robot_plan.png
```

The model architecture comes from the `config.py` saved next to the checkpoint. Use `--config` to point at a different file. Action normalization statistics are read from the checkpoint.

The script prints the planned metric trajectory and saves `robot_plan.png`. This 4-panel figure shows:
- the current image
- the goal image
- the planned trajectory over the initial CEM sample cloud
- the CEM cost over iterations

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to the `.pt` checkpoint |
| `--current_image` | *(required)* | Current camera image |
| `--goal_image` | *(required)* | Goal image |
| `--encoder` | `dino` | `dino` (DINOv2), `dinov3`, or `vjepa` |
| `--config` | auto | Override path to `config.py` (default: checkpoint directory) |
| `--plan_horizon` | `6` | Planning steps |
| `--cem_samples` | `128` | Action sequences sampled per CEM iteration |
| `--cem_iters` | `6` | CEM iterations |
| `--cem_elites` | `4` | Elites used to refit the sampling distribution |
| `--cem_var_scale` | `2.0` | Scale of the initial sampling std |
| `--cem_momentum` | `0.1` | Momentum of the mean/std update |
| `--no_unicycle` | off | Disable the unicycle kinematic prior |
| `--no_compile` | off | Disable `torch.compile` (useful on older CUDA / CPU) |
| `--output_plot` | `robot_plan.png` | Output figure |
| `--seed` | `42` | Random seed |

`cem_planning.py` is a TensorFlow-free copy of the CEM planner in `eval_planning.py`.
