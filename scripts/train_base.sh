#!/usr/bin/env bash
# Stage 1 — Base: rollout counter-curriculum (K = 2 -> 8), 18k steps.
# Usage: bash scripts/train_base.sh [dino|dinov3] [extra train.py flags...]
set -euo pipefail
ENCODER="${1:-dino}"; shift || true
NGPU="${NGPU:-4}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun --nproc_per_node "$NGPU" --master_port "$MASTER_PORT" train.py \
    --config configs/base.py \
    --encoder "$ENCODER" \
    --name base \
    "$@"
