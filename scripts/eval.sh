#!/usr/bin/env bash
# CEM planning evaluation with the paper's settings
# (K = 6 horizon, 4 context frames, 128 samples x 6 iterations, 4 elites, unicycle prior).
# Usage: bash scripts/eval.sh <checkpoint.pt> [dino|dinov3] [extra eval_planning.py flags...]
#
# config.py is read from the checkpoint directory. Results go to eval_results/.
set -euo pipefail
CKPT="$1"; shift
ENCODER="${1:-dino}"; shift || true
OUTDIR="${OUTDIR:-eval_results/$(basename "$(dirname "$CKPT")")}"
mkdir -p "$OUTDIR"

python eval_planning.py \
    --checkpoint "$CKPT" \
    --encoder "$ENCODER" \
    --dataset_name cast_filtered_dataset \
    --split val \
    --plan_horizon 6 \
    --seed_frames 4 \
    --ctxt_window 4 \
    --eval_window_size 12 \
    --cem_samples 128 \
    --cem_iters 6 \
    --cem_elites 4 \
    --cem_var_scale 2.0 \
    --cem_momentum 0.1 \
    --unicycle \
    --batch_size "${BATCH_SIZE:-4}" \
    --seed 42 \
    --output_json "$OUTDIR/eval.json" \
    --output_plot "$OUTDIR/last_example.png" \
    "$@"
