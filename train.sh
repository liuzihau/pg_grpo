#!/usr/bin/env bash
set -Eeuo pipefail

# Fail fast if the key isn't set in the shell
: "${WANDB_API_KEY:?Set WANDB_API_KEY first:  export WANDB_API_KEY=xxxxxxxx}"

# (Optional) run from repo root regardless of where the script is called
cd "$(dirname "$0")"

python -m scripts.train_kd \
  --config configs/kd_train.yaml \
  --wandb --wandb_api_key "$WANDB_API_KEY"

python -m scripts.eval_specdec \
  --config configs/eval.yaml \
  --wandb --wandb_api_key "$WANDB_API_KEY"

