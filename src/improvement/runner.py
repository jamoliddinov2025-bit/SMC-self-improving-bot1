"""Orchestration of one improvement run. Offline analysis only.

    ImprovementRunner(config, data_root).run(df, aux_frames) -> ImprovementResult

Order of operations
  1. refuse if improvement.enabled is false
  2. build ParameterSpace, SplitPlan (sealed holdout + anchored WF folds)
  3. evaluate the BASELINE on every fold; abort clearly if any OOS fold has too few baseline trades
  4. generate candidates (single-parameter coordinate descent), evaluate each on every IS/OOS fold
  5. validate (C1-C8), score, rank  - the holdout is NOT touched in this step
  6. evaluate ONLY the top-N survivors on the sealed holdout once (H1/H2)
  7. write report files under <results_directory>/<run_id>/

The runner never imports PaperTrader, never writes to config/ or src/.
"""

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.execution.state_store import config_hash
from src.improvement.candidates import Candidate, CandidateGenerator
from src.improvement.evaluator import Evaluator
from src.improvement.report import ReportWriter
from src.improvement.scoring import ScoringConfig, aggregate, fold_score, rank
from src.improvement.space import ParameterSpace
from src.improvement.splitter import DataSplitter, SplitConfig, SplitPlan
from src.improvement.validator import ConstraintConfig, FoldMetrics, improvement_pct, validate, validate_holdout

METRIC_COLS = ["trades", "winning_trades", "losing_trades", "win_rate_pct", "net_profit", "total_return_pct",
               "expectancy", "average_r_multiple", "profit_factor", "max_drawdown_pct", "total_fees",
               "warmup_trades_discarded"]


class ImprovementDisabled(RuntimeError):
    pass


class InsufficientData(RuntimeError):
    pass


@dataclass
class ImprovementResult:
    run_id: str
    out_dir: Optional[Path]
    aborted: bool
    abort_reason: Optional[str] = None
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, Path] = field(default_factory=dict)
    plan: Optional[SplitPlan] = None


class ImprovementRunner:
    def __init__(self, config: Dict[str, Any], data_root: Optional[Path] = None, run_id: Optional[str] = None,
                 max_candidates: Optional[int] = None, dry_run: bool = False, data_label: str = "",
                 synthetic: Optional[bool] = None, now: Optional[dt.datetime] = None):
        self.config = config
        self.data_root = Path(data_root) if data_root is not None else Path.cwd()
        imp = config.get("improvement", {}) or {}
        self.enabled = bool(imp.get("enabled", False))
        self.results_dir = self.data_root / imp.get("results_directory", "data/improvement/")
        self.top_n = int(imp.get("top_n_for_holdout", 5))
        self.max_candidates = max_candidates if max_candidates is not None else (imp.get("search", {}) or {}).get("max_candidates")
        self.dry_run = dry_run
        self.data_label = data_label
        self.synthetic = synthetic
        stamp = (now or dt.datetime.utcnow()).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"run_{stamp}"

    # ----------------------------------------------------------------- API
    def run(self, df: pd.DataFrame, aux_frames: Optional[Dict[str, pd.DataFrame]] = None) -> ImprovementResult:
        if not self.enabled:
            raise ImprovementDisabled("improvement.enabled is false - set it to true in config.yaml to run the analysis")
        space = ParameterSpace(self.config)
        split_cfg = SplitConfig.from_config(self.config)
        constraints = ConstraintConfig.from_config(self.config)
        scoring = ScoringConfig.from_config(self.config)
        out_dir = self.results_dir / self.run_id
        synthetic = self.synthetic if self.synthetic is not None else ("sample" in str(self.data_label).lower())
        data_info = self._data_info(df)
        base_summary = {
            "run_id": self.run_id, "created_utc": dt.datetime.utcnow().isoformat(), "aborted": False,
            "synthetic_data": synthetic, "data": data_info, "baseline_config_hash": config_hash(self.config),
            "config": {k: self.config.get(k) for k in ("market", "strategy", "usdtd", "auxiliary", "indicators", "risk",
                                                      "execution", "backtesting", "improvement")},
            "parameter_space": space.describe(), "skipped_parameters": space.skipped_parameters(),
            "constraints": vars(constraints), "scoring": vars(scoring), "split_config": vars(split_cfg),
        }

        # --- split
        try:
            plan = DataSplitter(split_cfg).plan(len(df))
        except (ValueError, AssertionError) as exc:
            return self._abort(out_dir, base_summary, f"cannot build walk-forward split: {exc}", synthetic)

        evaluator = Evaluator(self.config, df, aux_frames=aux_frames, data_root=self.data_root)
        base_summary["warmup_bars"] = evaluator.warmup
        base_summary["split"] = plan.to_dict()

        # --- baseline first; the data guard
        baseline = Candidate({})
        base_fm = self._fold_metrics(evaluator, baseline, plan, scoring)
        per_fold = [m["trades"] for m in base_fm.oos_metrics]
        total = sum(per_fold)
        if total < constraints.min_trades_before_change or any(n < constraints.min_trades_per_fold for n in per_fold):
            reason = (f"insufficient data: baseline produced {total} OOS trades over {len(per_fold)} folds {per_fold}; "
                      f"required >= {constraints.min_trades_before_change} total and >= {constraints.min_trades_per_fold} per fold "
                      f"(dataset {len(df)} bars). Provide more history instead of lowering the thresholds.")
            return self._abort(out_dir, base_summary, reason, synthetic, plan)

        # --- candidates (holdout untouched here)
        candidates = CandidateGenerator(space, max_candidates=self.max_candidates).generate()
        fms: Dict[str, FoldMetrics] = {baseline.id: base_fm}
        by_id: Dict[str, Candidate] = {baseline.id: baseline}
        for c in candidates[1:]:
            by_id[c.id] = c
            fms[c.id] = self._fold_metrics(evaluator, c, plan, scoring)

        rows: List[Dict[str, Any]] = []
        for c in candidates:
            fm = fms[c.id]
            neighbours = self._neighbour_medians(c, space, fms) if not c.is_baseline else None
            vr = validate(fm, base_fm, constraints, neighbours) if not c.is_baseline else None
            changed = list(c.overlay)
            rel = max([space.change_pct(k, v) or 0.0 for k, v in c.overlay.items()] or [0.0])
            rows.append({
                "candidate_id": c.id, "label": c.label(space), "stage": c.stage, "overlay": _overlay_str(c.overlay),
                "params_changed": len(changed), "rel_change_pct": rel,
                "oos_trades_total": sum(m["trades"] for m in fm.oos_metrics),
                "oos_score_median": fm.agg["median"], "oos_score_min": fm.agg["min"], "oos_score_mean": fm.agg["mean"],
                "oos_net_profit_total": sum(float(m["net_profit"]) for m in fm.oos_metrics),
                "improvement_pct": 0.0 if c.is_baseline else improvement_pct(fm.agg["median"], base_fm.agg["median"]),
                "passed": False if c.is_baseline else vr.passed,
                "reasons": "" if c.is_baseline else "; ".join(vr.reasons),
                **({} if c.is_baseline else {f"check_{k}": v for k, v in vr.checks.items()}),
                "holdout_verdict": "-", "verdict": "baseline" if c.is_baseline else ("survivor" if vr.passed else "rejected"),
            })
        ranking = rank(rows)

        # --- sealed holdout: top-N survivors only, once
        proposals: List[Dict[str, Any]] = []
        survivors = [r for r in ranking if r["passed"]][: self.top_n]
        for n, row in enumerate(survivors, 1):
            c = by_id[row["candidate_id"]]
            hres = evaluator.evaluate(c, plan.holdout)
            hv = validate_holdout(hres.metrics, fms[c.id], constraints)
            row["holdout_verdict"] = "pass" if hv.passed else "FAIL"
            row["holdout_net_profit"] = hres.metrics["net_profit"]
            row["holdout_trades"] = hres.metrics["trades"]
            row["holdout_avg_r"] = hres.metrics["average_r_multiple"]
            row["verdict"] = "recommended" if hv.passed else "not recommended - failed holdout"
            proposals.append(self._proposal(n, c, row, fms[c.id], base_fm, hres.metrics, hv, space, plan))
        # baseline holdout for reference only (does not influence anything above)
        base_hold = evaluator.evaluate(baseline, plan.holdout).metrics

        fold_rows = self._fold_rows(evaluator, candidates, plan, scoring)
        summary = {
            **base_summary,
            "search": {"method": "coordinate_single_parameter", "candidates": len(candidates) - 1,
                       "backtests_run": evaluator.backtests_run, "max_candidates": self.max_candidates},
            "baseline": {"oos_score_median": base_fm.agg["median"], "oos_score_min": base_fm.agg["min"],
                         "oos_trades_total": total, "holdout": {k: base_hold.get(k) for k in METRIC_COLS}},
            "survivors": len([r for r in ranking if r["passed"]]),
            "recommended": [p["proposal_id"] for p in proposals if p["status"] == "recommended_pending_human_review"],
            "proposals": [p["proposal_id"] for p in proposals],
            "top": ranking[:10],
        }
        result = ImprovementResult(self.run_id, out_dir, False, None, ranking, proposals, summary, {}, plan)
        if not self.dry_run:
            result.files = ReportWriter(out_dir).write({
                "run_id": self.run_id, "synthetic": synthetic, "ranking": ranking, "fold_rows": fold_rows,
                "proposals": proposals, "summary": summary})
        return result

    # ------------------------------------------------------------ internals
    def _abort(self, out_dir: Path, summary: Dict[str, Any], reason: str, synthetic: bool,
               plan: Optional[SplitPlan] = None) -> ImprovementResult:
        summary = {**summary, "aborted": True, "abort_reason": reason}
        res = ImprovementResult(self.run_id, out_dir, True, reason, summary=summary, plan=plan)
        if not self.dry_run:
            res.files = ReportWriter(out_dir).write_aborted({"run_id": self.run_id, "synthetic": synthetic,
                                                              "abort_reason": reason, "summary": summary})
        return res

    def _fold_metrics(self, ev: Evaluator, c: Candidate, plan: SplitPlan, scoring: ScoringConfig) -> FoldMetrics:
        is_m = [ev.evaluate(c, f.train).metrics for f in plan.folds]
        oos_m = [ev.evaluate(c, f.test).metrics for f in plan.folds]
        scores = [fold_score(m, scoring) for m in oos_m]
        return FoldMetrics(is_m, oos_m, scores, aggregate(scores))

    def _neighbour_medians(self, c: Candidate, space: ParameterSpace, fms: Dict[str, FoldMetrics]) -> Optional[List[float]]:
        if len(c.overlay) != 1:
            return None
        (path, value), = c.overlay.items()
        out = []
        for nv in space.neighbours(path, value):
            nid = "baseline" if nv == space.baseline[path] else Candidate({path: nv}).id
            if nid in fms:
                out.append(fms[nid].agg["median"])
        return out

    def _fold_rows(self, ev: Evaluator, candidates: List[Candidate], plan: SplitPlan, scoring: ScoringConfig):
        rows = []
        for c in candidates:
            for f in plan.folds:
                for role, s in (("is", f.train), ("oos", f.test)):
                    r = ev.evaluate(c, s)
                    rows.append({"candidate_id": c.id, "label": c.label(), "fold": f.k, "role": role, "slice": s.name,
                                 "start_bar": s.start, "end_bar": s.end, "bars": s.bars,
                                 **{k: r.metrics.get(k) for k in METRIC_COLS}, "score": fold_score(r.metrics, scoring)})
            if (c.id, plan.holdout.name) in ev._cache:
                r = ev.evaluate(c, plan.holdout)
                rows.append({"candidate_id": c.id, "label": c.label(), "fold": 0, "role": "holdout", "slice": "holdout",
                             "start_bar": plan.holdout.start, "end_bar": plan.holdout.end, "bars": plan.holdout.bars,
                             **{k: r.metrics.get(k) for k in METRIC_COLS}, "score": fold_score(r.metrics, scoring)})
        return rows

    def _proposal(self, n: int, c: Candidate, row: Dict[str, Any], fm: FoldMetrics, base_fm: FoldMetrics,
                  hold: Dict[str, Any], hv, space: ParameterSpace, plan: SplitPlan) -> Dict[str, Any]:
        status = "recommended_pending_human_review" if hv.passed else "not_recommended_failed_holdout"
        return {
            "proposal_id": f"P-{n}",
            "status": status,
            "run_id": self.run_id,
            "created_utc": dt.datetime.utcnow().isoformat(),
            "baseline_config_hash": config_hash(self.config),
            "baseline_values": {k: space.baseline[k] for k in c.overlay},
            "parameter_overlay": dict(c.overlay),
            "candidate_id": c.id,
            "rank": row["rank"],
            "reason": (f"median OOS score {fm.agg['median']:.3f} vs baseline {base_fm.agg['median']:.3f} "
                       f"({row['improvement_pct']:.1f}% better), min {fm.agg['min']:.3f} >= baseline min {base_fm.agg['min']:.3f}; "
                       f"passed all {len(plan.folds)} walk-forward folds (C1-C8)"
                       + ("; passed sealed holdout" if hv.passed else "; FAILED sealed holdout: " + "; ".join(hv.reasons))),
            "evidence": {
                "oos_trades_total": row["oos_trades_total"], "oos_score_median": fm.agg["median"],
                "oos_score_min": fm.agg["min"], "improvement_pct": row["improvement_pct"],
                "baseline_oos_score_median": base_fm.agg["median"], "baseline_oos_score_min": base_fm.agg["min"],
                "constraint_checks": {k: bool(v) for k, v in row.items() if k.startswith("check_")},
            },
            "fold_metrics": [{"fold": k + 1, "is": {m: fm.is_metrics[k].get(m) for m in METRIC_COLS},
                              "oos": {m: fm.oos_metrics[k].get(m) for m in METRIC_COLS}, "oos_score": fm.oos_scores[k]}
                             for k in range(len(fm.oos_metrics))],
            "holdout_result": {"verdict": "pass" if hv.passed else "fail", "reasons": hv.reasons,
                               "bars": plan.holdout.bars, "metrics": {m: hold.get(m) for m in METRIC_COLS}},
            "how_to_apply": "python src/main.py proposal apply <id> --confirm <id>  (writes config/config.proposed.<id>.yaml only)",
        }

    def _data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
        return {"file": self.data_label, "symbol": self.config["market"]["symbol"], "timeframe": self.config["market"]["timeframe"],
                "bars": int(len(df)), "first_timestamp": str(df["timestamp"].iloc[0]) if len(df) else None,
                "last_timestamp": str(df["timestamp"].iloc[-1]) if len(df) else None, "sha256": h}


def _overlay_str(o: Dict[str, Any]) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(o.items()))
