# src/specdec/metrics.py
from __future__ import annotations
from typing import Dict, Any, List
from statistics import mean
from src.common.io import save_json
import csv
from pathlib import Path


def _safe_mean(vals: List[float]) -> float:
    return float(mean(vals)) if len(vals) > 0 else 0.0


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "alpha_accept": 0.0,
            "goodput_tokens_per_teacher_call": 0.0,
            "avg_accepted_span": 0.0,
            "reject_rate": 0.0,
            "mean_accepted_tokens": 0.0,
            "mean_teacher_calls": 0.0,
        }
    return {
        "alpha_accept": _safe_mean([r["alpha_accept"] for r in rows]),
        "goodput_tokens_per_teacher_call": _safe_mean([r["goodput_tokens_per_teacher_call"] for r in rows]),
        "avg_accepted_span": _safe_mean([r["avg_accepted_span"] for r in rows]),
        "reject_rate": _safe_mean([r["reject_rate"] for r in rows]),
        "mean_accepted_tokens": _safe_mean([r["accepted"] for r in rows]),
        "mean_teacher_calls": _safe_mean([r["teacher_calls"] for r in rows]),
    }


def save_metrics_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write just a header
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt", "accepted", "compared", "teacher_calls",
                        "alpha_accept", "goodput_tokens_per_teacher_call",
                        "avg_accepted_span", "reject_rate"])
        return
    keys = ["prompt", "accepted", "compared", "teacher_calls",
            "alpha_accept", "goodput_tokens_per_teacher_call",
            "avg_accepted_span", "reject_rate"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
