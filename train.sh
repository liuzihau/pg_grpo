#!/usr/bin/env bash
set -Eeuo pipefail

# Fail fast if the key isn't set in the shell
: "${WANDB_API_KEY:?Set WANDB_API_KEY first:  export WANDB_API_KEY=xxxxxxxx}"

# (Optional) run from repo root regardless of where the script is called
cd "$(dirname "$0")"

python -m scripts.train_kd --config configs/defaults.yaml --wandb_api_key "$WANDB_API_KEY"

python -m scripts.eval_specdec \
  --config configs/defaults.yaml \
  --lora_path outputs/kd_lora \
  --split validation \
  --num_samples 256 \
  --K 8 \
  --max_new 128 \
  --temperature 0.0 \
  --wandb \
  --wandb_api_key "$WANDB_API_KEY" \
  --wandb_project specdec-two-stage \
  --wandb_run_name qwen06_lora_eval_K8 \
  --out_dir outputs/eval_spec/qwen06_lora_K8
