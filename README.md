# Prefix-Growth GRPO Prototype (LoRA Draft vs Target)

This is a minimal, hackable prototype to align a **draft** model (LoRA-adapted)
to a **frozen target** model using **GRPO** with **prefix-growth** and **KV reuse**.
It implements **Expected Accepted Length (EAL)** so early tokens matter more.

> Draft proposes S chunks of K tokens; Target scores them; we compute an
> EAL reward; GRPO updates LoRA; then we *grow the real prefix* by accepting
> tokens stochastically using vanilla SD acceptance `a_t = min(1, q_t/p_t)`.
> We reuse KV for both models to avoid re-encoding the prompt.

## Requirements

- Python 3.10+
- PyTorch 2.2+ with CUDA (recommended)
- `transformers`, `accelerate`, `peft` (for LoRA)
- Optionally `wandb` or `tensorboard` for logging

Install (example):
```bash
pip install torch transformers accelerate peft
pip install wandb  # optional
```

## Quick Start

1. Put prompts (one JSON per line) in `data/prompts.jsonl`:
```json
{"prompt": "<|im_start|>user\nTell me a story.\n<|im_end|>\n<|im_start|>assistant\n"}
```
2. Edit `configs/train.yaml` if needed.
3. Run:
```bash
python train.py --config configs/train.yaml --prompts data/prompts.jsonl
```

This demonstrates the full control flow and is a good starting point to scale up.


## Instrumentation Added
- **Weights & Biases** logging (toggle via `logging.use_wandb`).
- **tqdm** progress bars for epochs and per-prompt steps.
- **Evaluation phase** after each epoch with real speculative decoding:
  - Logs real accepted prefix len per prompt.
  - Logs per-position alpha up to `eval.K_eval_max`.
  - Aggregates: mean/median/std acceptance length; per-position alpha means; macro/micro alpha means.
