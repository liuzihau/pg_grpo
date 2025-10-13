
from __future__ import annotations
import json
from typing import List, Dict

class PromptOnlyDataset:
    """Minimal prompt-only dataset: loads JSONL with {"prompt": "..."}."""
    def __init__(self, path: str):
        self.items: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj = json.loads(line)
                self.items.append({"prompt": obj["prompt"]})

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
