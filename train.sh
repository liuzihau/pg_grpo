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
  --config configs/defaults.yaml \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --draft_name Qwen/Qwen3-0.6B \
  --teacher_name Qwen/Qwen3-8B \
  --lora_path outputs/kd_lora \
  --split validation \
  --num_samples 64 \
  --K 8 \
  --max_new 64 \
  --temperature 0.0

