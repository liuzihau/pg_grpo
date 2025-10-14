# data.py
from __future__ import annotations
import json
from typing import List, Dict, Optional, Any, Iterable

# ---------- Existing JSONL dataset ----------
class PromptOnlyDataset:
    """Loads JSONL with {"prompt": "..."} rows and returns dicts: {"prompt": str}."""
    def __init__(self, path: str):
        self.items: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append({"prompt": obj["prompt"]})

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, i: int) -> Dict[str, str]: return self.items[i]


# ---------- Helpers for HF chat datasets ----------
def _normalize_messages(rec: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Try to extract a list of {'role': 'user'|'assistant'|'system', 'content': str}
    from common schema variants found in HF datasets.
    """
    # Common field names
    for key in ("messages", "conversations", "chat"):
        if key in rec and isinstance(rec[key], list):
            return rec[key]

    # Alpaca/ShareGPT-like fields
    if "prompt" in rec:
        return [{"role": "user", "content": rec["prompt"]}]
    if "instruction" in rec and "output" in rec:
        return [
            {"role": "user", "content": rec["instruction"]},
            {"role": "assistant", "content": rec["output"]},
        ]

    return None


def _to_user_terminal_history(msgs: List[Dict[str, str]], keep_history: bool = True) -> Optional[List[Dict[str, str]]]:
    """
    Ensure the conversation we keep ends on a USER turn.
    - If the last message is 'assistant', drop trailing assistant turns until the last 'user'.
    - If there is no user at all, return None (skip sample).
    - If keep_history=False, only keep the last user message (drop previous turns).
    """
    # Find the last user index
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None  # no user; skip

    # Truncate everything after the last user turn
    msgs = msgs[: last_user_idx + 1]

    # Optionally keep only the last user turn
    if not keep_history:
        msgs = [msgs[-1]]

    return msgs


def _render_prompt(tokenizer, msgs_user_terminal: List[Dict[str, str]]) -> str:
    """
    Render a prompt string that:
      - Includes history up to and including the final USER turn.
      - Ends with an assistant generation start (no assistant content).
    Prefers tokenizer.apply_chat_template(..., add_generation_prompt=True).
    Falls back to a simple Qwen-like template if no chat template exists.
    """
    try:
        # Many modern tokenizers (Qwen, Llama, etc.) define a chat template.
        return tokenizer.apply_chat_template(
            msgs_user_terminal,
            tokenize=False,
            add_generation_prompt=True,  # <-- makes it "start from assistant"
        )
    except Exception:
        # Fallback: Qwen-style tags
        parts: List[str] = []
        for m in msgs_user_terminal:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        parts.append("<|im_start|>assistant\n")  # generation starts here
        return "\n".join(parts)


# ---------- New: HF chat prompts dataset ----------
class HFChatPromptsDataset:
    """
    HuggingFace chat dataset loader that always returns prompts which:
      * end on a USER message, and
      * start generation from the ASSISTANT.

    Each item is a dict: {"prompt": <rendered prompt str>}

    Args:
        dataset_name: HF dataset repo id, e.g. "allenai/tulu-3-sft-mixture".
        split: dataset split, e.g. "train".
        tokenizer: the same tokenizer used by your models; must support chat template ideally.
        messages_field: if your dataset stores messages under a custom key (default "messages").
        sample_max: limit the number of examples (None = use all).
        keep_history: if True, keep full history up to last user; if False, keep only the last user turn.
        load_kwargs: extra args passed to datasets.load_dataset (e.g., cache_dir, streaming, etc.).
    """
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
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("Please `pip install datasets` to use HFChatPromptsDataset") from e

        self.items: List[Dict[str, str]] = []
        self.tokenizer = tokenizer

        load_kwargs = load_kwargs or {}
        ds = load_dataset(dataset_name, split=split, **load_kwargs)

        n = len(ds) if sample_max is None else min(sample_max, len(ds))
        for i in range(n):
            rec = ds[i]
            # 1) normalize schema to a list of {'role','content'}
            msgs = rec.get(messages_field) if messages_field in rec else _normalize_messages(rec)
            if msgs is None or not isinstance(msgs, list):
                continue

            # 2) ensure we end on a USER turn (drop trailing assistant turns)
            msgs_u = _to_user_terminal_history(msgs, keep_history=keep_history)
            if not msgs_u:
                continue

            # 3) render a prompt that starts generation from the assistant
            prompt_txt = _render_prompt(self.tokenizer, msgs_u)
            self.items.append({"prompt": prompt_txt})

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, i: int) -> Dict[str, str]: return self.items[i]
