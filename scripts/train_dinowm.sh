#!/usr/bin/env bash
# DINO-WM baseline: ViT predictor with per-patch action concatenation,
# single-step teacher forcing, pure L2 loss.
# Usage: bash scripts/train_dinowm.sh [dino|dinov3] [extra train.py flags...]
set -euo pipefail
ENCODER="${1:-dino}"; shift || true
NGPU="${NGPU:-4}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun --nproc_per_node "$NGPU" --master_port "$MASTER_PORT" train.py \
    --config configs/dinowm.py \
    --encoder "$ENCODER" \
    --name dinowm \
    "$@"
