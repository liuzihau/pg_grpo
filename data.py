# data.py
from __future__ import annotations
import json
from typing import List, Dict, Optional, Any

class PromptOnlyDataset:
    def __init__(self, path: str):
        self.items: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "prompt" in obj and isinstance(obj["prompt"], str):
                    self.items.append({"prompt": obj["prompt"]})

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, i: int) -> Dict[str, str]: return self.items[i]

def _to_user_terminal_history(msgs: List[Dict[str, str]], keep_history: bool = True) -> Optional[List[Dict[str, str]]]:
    # keep up to the last 'user' turn
    last_user = None
    for i in reversed(range(len(msgs))):
        if str(msgs[i].get("role", "")).lower() == "user":
            last_user = i
            break
    if last_user is None:
        return None
    kept = msgs[: last_user + 1]
    if not keep_history:
        kept = [kept[-1]]
    return kept

def _render_prompt(tokenizer, msgs_u: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(msgs_u, tokenize=False, add_generation_prompt=True)
    except Exception:
        # very safe fallback
        parts = []
        for m in msgs_u:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

class HFChatPromptsDataset:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        messages_field: str = "messages",
        sample_max: Optional[int] = None,
        keep_history: bool = True,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from datasets import load_dataset
        lw = dict(load_kwargs or {})
        # give users a way to pass token/cache/streaming via cfg
        ds = load_dataset(dataset_name, split=split, **lw)

        # accept multiple common field names
        candidate_fields = [messages_field, "messages", "conversations", "conversation", "turns"]
        self.items: List[Dict[str, str]] = []
        for rec in ds:
            msgs = None
            for field in candidate_fields:
                if field in rec and isinstance(rec[field], list):
                    msgs = rec[field]
                    break
            if not isinstance(msgs, list):
                continue
            # normalize to {role, content}
            norm = []
            for m in msgs:
                if isinstance(m, dict):
                    role = str(m.get("role", m.get("from", ""))).lower()
                    # some datasets call text 'value' or 'content'
                    content = m.get("content", m.get("value", ""))
                    if not content:
                        continue
                    norm.append({"role": role, "content": content})
            msgs_u = _to_user_terminal_history(norm, keep_history=keep_history)
            if not msgs_u:
                continue
            prompt_txt = _render_prompt(tokenizer, msgs_u)
            self.items.append({"prompt": prompt_txt})
            if sample_max is not None and len(self.items) >= int(sample_max):
                break

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, i: int) -> Dict[str, str]: return self.items[i]
