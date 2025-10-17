# src/training/optim.py
import torch

def build_adamw(model, lr: float, weight_decay: float = 0.0, betas=(0.9, 0.95), eps=1e-8):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
