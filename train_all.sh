#!/usr/bin/env bash
set -euo pipefail

SCRIPT="train.py"
CONFIG_DIR="configs/swa"
LOG_DIR="logs_swa"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "train_1h.yaml"
  "train_10h.yaml"
  "train_20h.yaml"
  "train_30h.yaml"
  "train_50h.yaml"
  "train_100h.yaml"
  "train_150h.yaml"
  "train_200h.yaml"
  "train_250h.yaml"
  "train_300h.yaml"
  "train_full.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  name="${cfg%.yaml}"
  echo "🚀 Starting ${name} ..."
  # tee logs so you can tail while it runs
  uv run python "${SCRIPT}" --config "${CONFIG_DIR}/${cfg}" 2>&1 | tee "${LOG_DIR}/${name}.log"
  # Clear VRAM cache (usually unnecessary after process exit, but harmless)
  python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY
  # small pause between runs
  sleep 10
  echo "✅ Finished ${name}"
done


