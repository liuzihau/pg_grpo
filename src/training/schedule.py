# src/training/schedule.py
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def make_warmup_cosine(optim: Optimizer, total_steps: int, warmup_ratio: float = 0.05, min_lr: float = 0.0):
    warm = max(1, int(total_steps * warmup_ratio))
    base_lrs = [g["lr"] for g in optim.param_groups]

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        prog = (s - warm) / max(1, total_steps - warm)
        # cosine to min_lr factor
        return (min_lr / base_lrs[0]) + (1.0 - (min_lr / base_lrs[0])) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optim, lr_lambda)
