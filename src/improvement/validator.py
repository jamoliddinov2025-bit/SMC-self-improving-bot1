"""Walk-forward constraints. A candidate is recommendable ONLY if every check passes.

Checks (all against the OOS folds unless stated):
  C1 total OOS trades            >= min_trades_before_change
  C2 every OOS fold trades       >= min_trades_per_fold
  C3 every OOS fold net_profit   > 0   (or >= baseline's on that fold when require_positive_all_folds=false)
  C4 robustness                  median(score) > baseline median AND min(score) >= baseline min
  C5 overfit guard               OOS expectancy / IS expectancy >= oos_is_ratio_min (per fold, when IS expectancy > 0)
  C6 drawdown                    every fold (IS and OOS) max_drawdown_pct <= limit (default risk.max_drawdown_pct)
  C7 minimum improvement         median score improvement over baseline >= min_improvement_pct
  C8 neighbourhood stability     not BOTH adjacent grid values worse than baseline (single-parameter candidates)
  H  holdout (top-N only)        net_profit > 0 and expectancy >= 50% of median WF-OOS expectancy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConstraintConfig:
    min_trades_before_change: int = 100
    min_trades_per_fold: int = 8
    require_positive_all_folds: bool = True
    oos_is_ratio_min: float = 0.5
    min_improvement_pct: float = 5.0
    max_drawdown_pct_limit: float = 10.0
    holdout_expectancy_ratio_min: float = 0.5

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConstraintConfig":
        imp = config.get("improvement", {}) or {}
        c = imp.get("constraints", {}) or {}
        d = imp.get("data", {}) or {}
        dd = c.get("max_drawdown_pct_limit")
        return cls(int(imp.get("min_trades_before_change", 100)), int(d.get("min_trades_per_fold", 8)),
                   bool(c.get("require_positive_all_folds", True)), float(c.get("oos_is_ratio_min", 0.5)),
                   float(c.get("min_improvement_pct", 5.0)),
                   float(config["risk"]["max_drawdown_pct"] if dd is None else dd),
                   float(c.get("holdout_expectancy_ratio_min", 0.5)))


@dataclass
class ValidationResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FoldMetrics:
    """Per candidate: lists aligned by fold k."""
    is_metrics: List[Dict[str, Any]]
    oos_metrics: List[Dict[str, Any]]
    oos_scores: List[float]
    agg: Dict[str, float]


def validate(cand_fm: FoldMetrics, base_fm: FoldMetrics, cfg: ConstraintConfig,
             neighbour_medians: Optional[List[float]] = None) -> ValidationResult:
    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    total = sum(int(m["trades"]) for m in cand_fm.oos_metrics)
    checks["C1_min_total_oos_trades"] = total >= cfg.min_trades_before_change
    if not checks["C1_min_total_oos_trades"]:
        reasons.append(f"C1 total OOS trades {total} < {cfg.min_trades_before_change}")

    per_fold = [int(m["trades"]) for m in cand_fm.oos_metrics]
    checks["C2_min_trades_per_fold"] = all(n >= cfg.min_trades_per_fold for n in per_fold)
    if not checks["C2_min_trades_per_fold"]:
        reasons.append(f"C2 OOS trades per fold {per_fold} < {cfg.min_trades_per_fold}")

    ok = True
    for k, (m, b) in enumerate(zip(cand_fm.oos_metrics, base_fm.oos_metrics), 1):
        if cfg.require_positive_all_folds:
            good = float(m["net_profit"]) > 0
        else:
            good = float(m["net_profit"]) > 0 or float(m["net_profit"]) >= float(b["net_profit"])
        if not good:
            ok = False
            reasons.append(f"C3 OOS fold {k} net_profit {float(m['net_profit']):.2f} not positive")
    checks["C3_positive_folds"] = ok

    checks["C4_robustness"] = (cand_fm.agg["median"] > base_fm.agg["median"] and cand_fm.agg["min"] >= base_fm.agg["min"])
    if not checks["C4_robustness"]:
        reasons.append(f"C4 median {cand_fm.agg['median']:.3f}/min {cand_fm.agg['min']:.3f} vs baseline "
                       f"{base_fm.agg['median']:.3f}/{base_fm.agg['min']:.3f}")

    ok = True
    for k, (i_m, o_m) in enumerate(zip(cand_fm.is_metrics, cand_fm.oos_metrics), 1):
        ie, oe = float(i_m["average_r_multiple"]), float(o_m["average_r_multiple"])
        if ie > 0 and oe / ie < cfg.oos_is_ratio_min:
            ok = False
            reasons.append(f"C5 fold {k} OOS/IS expectancy ratio {oe / ie:.2f} < {cfg.oos_is_ratio_min}")
    checks["C5_oos_is_ratio"] = ok

    dds = [float(m["max_drawdown_pct"]) for m in cand_fm.is_metrics + cand_fm.oos_metrics]
    checks["C6_drawdown"] = all(d <= cfg.max_drawdown_pct_limit for d in dds)
    if not checks["C6_drawdown"]:
        reasons.append(f"C6 max drawdown {max(dds):.2f}% > {cfg.max_drawdown_pct_limit}%")

    checks["C7_min_improvement"] = improvement_pct(cand_fm.agg["median"], base_fm.agg["median"]) >= cfg.min_improvement_pct
    if not checks["C7_min_improvement"]:
        reasons.append(f"C7 improvement {improvement_pct(cand_fm.agg['median'], base_fm.agg['median']):.1f}% "
                       f"< {cfg.min_improvement_pct}%")

    if neighbour_medians is None or len(neighbour_medians) == 0:
        checks["C8_neighbourhood"] = True
    else:
        worse = [n < base_fm.agg["median"] for n in neighbour_medians]
        checks["C8_neighbourhood"] = not (len(worse) >= 2 and all(worse))
        if not checks["C8_neighbourhood"]:
            reasons.append("C8 both neighbouring grid values are worse than baseline (isolated spike)")

    return ValidationResult(all(checks.values()), reasons, checks)


def validate_holdout(holdout: Dict[str, Any], cand_fm: FoldMetrics, cfg: ConstraintConfig) -> ValidationResult:
    import statistics
    checks, reasons = {}, []
    checks["H1_positive"] = float(holdout["net_profit"]) > 0
    if not checks["H1_positive"]:
        reasons.append(f"H1 holdout net_profit {float(holdout['net_profit']):.2f} <= 0")
    wf = statistics.median([float(m["average_r_multiple"]) for m in cand_fm.oos_metrics]) if cand_fm.oos_metrics else 0.0
    he = float(holdout["average_r_multiple"])
    checks["H2_expectancy"] = (he >= cfg.holdout_expectancy_ratio_min * wf) if wf > 0 else he > 0
    if not checks["H2_expectancy"]:
        reasons.append(f"H2 holdout expectancy {he:.3f}R < {cfg.holdout_expectancy_ratio_min:.0%} of WF-OOS {wf:.3f}R")
    return ValidationResult(all(checks.values()), reasons, checks)


def improvement_pct(cand: float, base: float) -> float:
    if base == 0:
        return float("inf") if cand > 0 else (0.0 if cand == 0 else -float("inf"))
    return (cand - base) / abs(base) * 100.0
