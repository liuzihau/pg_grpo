# src/model/lora.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
