"""Scoring and ranking. Pure functions over metrics dicts - deterministic, no randomness.

score(fold) = average_r_multiple * sqrt(trades) - dd_penalty * max_drawdown_pct
aggregate   = median over OOS folds (robustness over peak); min is reported alongside.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ScoringConfig:
    dd_penalty: float = 0.05

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ScoringConfig":
        s = (config.get("improvement", {}) or {}).get("scoring", {}) or {}
        return cls(float(s.get("dd_penalty", 0.05)))


def fold_score(metrics: Dict[str, Any], cfg: ScoringConfig) -> float:
    n = int(metrics.get("trades", 0))
    if n == 0:
        return 0.0
    return float(metrics["average_r_multiple"]) * math.sqrt(n) - cfg.dd_penalty * float(metrics["max_drawdown_pct"])


def aggregate(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"median": 0.0, "min": 0.0, "mean": 0.0}
    return {"median": float(statistics.median(scores)), "min": float(min(scores)),
            "mean": float(sum(scores) / len(scores))}


def rank_key(row: Dict[str, Any]):
    """Passed first -> higher median OOS score -> fewer params changed -> smaller relative change -> id."""
    return (0 if row["passed"] else 1, -row["oos_score_median"], row["params_changed"], row["rel_change_pct"], row["candidate_id"])


def rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = sorted(rows, key=rank_key)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out
