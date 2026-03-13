#!/usr/bin/env bash
set -euo pipefail

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="tmp/runs/${RUN_ID}"

echo "=== Run: ${RUN_ID} ==="
echo "=== Artifacts: ${RUN_DIR} ==="

uv run python scripts/train.py \
    --total-timesteps 5000 \
    --num-envs 2 \
    --num-steps 256 \
    --no-wandb \
    --learning-rate 3e-4 \
    --run-dir "${RUN_DIR}" \

CKPT=$(ls -t "${RUN_DIR}/checkpoints/"*.ckpt 2>/dev/null | head -1)
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found, skipping evaluation."
    exit 0
fi

echo ""
echo "=== Evaluating: ${CKPT} ==="

uv run python scripts/evaluate.py \
    --checkpoint "${CKPT}" \
    --n-rollouts 5 \
    --output-pdb "${RUN_DIR}/prediction.pdb"

echo ""
echo "=== Done. Artifacts in ${RUN_DIR} ==="
