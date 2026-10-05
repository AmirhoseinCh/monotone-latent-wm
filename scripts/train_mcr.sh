#!/usr/bin/env bash
# Stage 2 — MCR: Monotone Cost Ranking fine-tuning (+1k steps) from a Base checkpoint.
# Usage: bash scripts/train_mcr.sh <base_checkpoint.pt> [dino|dinov3] [extra train.py flags...]
#
# configs/mcr.py sets num_steps = 19000 (an 18k-step Base checkpoint + 1k MCR steps).
# For a checkpoint at a different step, pass --num_steps <ckpt_step + 1000>.
set -euo pipefail
BASE_CKPT="$1"; shift
ENCODER="${1:-dino}"; shift || true
NGPU="${NGPU:-4}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun --nproc_per_node "$NGPU" --master_port "$MASTER_PORT" train.py \
    --config configs/mcr.py \
    --encoder "$ENCODER" \
    --name mcr \
    --resume "$BASE_CKPT" \
    --skip_optimizer_state \
    "$@"
