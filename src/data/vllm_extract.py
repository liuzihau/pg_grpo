# src/data/vllm_extract.py
from __future__ import annotations
from typing import List, Tuple, Any

def pairs_from_logprob_step(cand: Any, tokenizer) -> List[Tuple[int, float]]:
    """
    Normalize one step's candidate logprobs into [(token_id:int, logprob:float)].

    vLLM 0.11 shapes:
      - dict: { token(str|int) -> float }
      - list: [TokenLogprob(token_id=int, logprob=float), ...]
    """
    pairs: List[Tuple[int, float]] = []
    if isinstance(cand, dict):
        for tk, lp in cand.items():
            if isinstance(tk, int):
                tid = tk
            else:
                try:
                    tid = tokenizer.convert_tokens_to_ids(tk)
                except Exception:
                    continue
            if tid is None or tid < 0:
                continue
            try:
                pairs.append((int(tid), float(lp)))
            except Exception:
                # Some builds wrap lp in a small object; best-effort cast
                try:
                    pairs.append((int(tid), float(getattr(lp, "logprob", lp))))
                except Exception:
                    continue
    else:
        # iterable of objects with .token_id and .logprob
        for obj in cand:
            tid = getattr(obj, "token_id", None)
            lp  = getattr(obj, "logprob", None)
            if tid is None or lp is None:
                continue
            try:
                pairs.append((int(tid), float(lp)))
            except Exception:
                continue
    return pairs
