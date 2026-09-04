"""Step 11 - REAL-HISTORICAL BASELINE benchmark for the CURRENT strategy. READ-ONLY.

Answers "what does the current SMCStrategy actually do on this frozen dataset?" and nothing else:
no tuning, no proposals, no config changes. It reuses, unchanged:

    DatasetMarketData (Step 10)  -> hash-verified frozen dataset, refuses tampered data
    validate_frame    (Step 10)  -> data-quality gate (critical problems ABORT the benchmark)
    BacktestEngine.from_config   -> the one execution model (fills, stops, targets, slippage, fees)
    SMCStrategy.from_config      -> the current strategy, as configured
    compute_metrics              -> base metrics; extra metrics here are derived from the journal only

Outputs go to <benchmark.results_directory>/<benchmark_id>/ (manifest.json, metrics.json,
trades.csv, rejections.csv, equity_curve.csv, validation.json, report.md).
"""

import copy
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.backtesting import EXIT_END, EXIT_SIGNAL, EXIT_STOP, EXIT_TARGET, BacktestEngine
from src.backtesting.aux_data import aux_filename, aux_specs_from_config
from src.data.dataset import DatasetError, DatasetMarketData, load_manifest, read_series_csv, verify_dataset
from src.data.timeframes import iso
from src.data.validate import ValidationConfig, alignment_preview, validate_frame
from src.execution.state_store import config_hash
from src.strategy.smc_strategy import SMCStrategy

BENCHMARK_SCHEMA = 1
LABEL_REAL = "REAL-HISTORICAL BASELINE"
LABEL_SYNTHETIC = "SYNTHETIC / FIXTURE BASELINE (NOT real market data)"
DISCLAIMER = ("This is a historical backtest of the CURRENT strategy on a frozen dataset. It is not a recommendation "
              "and does not prove profitability. Historical backtest performance does not guarantee future performance. "
              "No parameter was tuned and no change is proposed or applied by this report.")

# validation codes that make a performance number misleading -> the benchmark refuses to run
CRITICAL_CODES = ("V1", "V2", "V3", "V4", "V6", "V8", "V10")

# config sections that define the benchmark (superset of the paper-trader config hash sections)
_SNAPSHOT_SECTIONS = ("market", "data", "strategy", "usdtd", "auxiliary", "indicators", "risk", "execution", "backtesting")


class BenchmarkError(Exception):
    pass


@dataclass
class BenchmarkConfig:
    results_directory: str = "data/benchmarks/"
    dataset_directory: Optional[str] = None      # default: data.directory (must hold a manifest.json)
    min_trades_for_statistics: int = 30          # below this, ratio metrics are reported as unavailable
    fail_on_warnings: bool = False
    start_date: Optional[str] = None             # optional window inside the dataset
    end_date: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BenchmarkConfig":
        b = config.get("benchmark", {}) or {}
        return cls(str(b.get("results_directory", "data/benchmarks/")), b.get("dataset_directory"),
                   int(b.get("min_trades_for_statistics", 30)), bool(b.get("fail_on_warnings", False)),
                   b.get("start_date"), b.get("end_date"))


@dataclass
class BenchmarkResult:
    benchmark_id: str
    aborted: bool = False
    abort_reason: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, Path] = field(default_factory=dict)
    backtest: Any = None


# ------------------------------------------------------------------ helpers
def repo_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy({k: config.get(k) for k in _SNAPSHOT_SECTIONS})


def snapshot_hash(snapshot: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _clean(v):
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


# ------------------------------------------------------------------ runner
class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any], root: Union[str, Path], benchmark_id: Optional[str] = None,
                 dataset_directory: Optional[str] = None, dry_run: bool = False,
                 now: Optional[dt.datetime] = None, commit: Optional[str] = None):
        self.config = copy.deepcopy(config)          # the live dict is never touched
        self.root = Path(root)
        self.bcfg = BenchmarkConfig.from_config(config)
        rel = dataset_directory or self.bcfg.dataset_directory or config["data"]["directory"]
        self.dataset_rel = str(rel)
        self.dataset_dir = (self.root / rel) if not Path(rel).is_absolute() else Path(rel)
        self.dry_run = dry_run
        self.now = now or dt.datetime.now(dt.timezone.utc)
        self.commit = commit if commit is not None else repo_commit(self.root)
        self.benchmark_id = benchmark_id or f"bench_{self.now.strftime('%Y%m%dT%H%M%SZ')}"
        self.out_dir = self.root / self.bcfg.results_directory / self.benchmark_id

    # -------------------------------------------------------------- steps
    def run(self) -> BenchmarkResult:
        res = BenchmarkResult(self.benchmark_id)
        try:
            manifest = self._load_manifest()
            validation = self._validate(manifest)
            res.validation = validation
            if validation["status"] == "failed":
                raise BenchmarkError("data validation failed: " + "; ".join(validation["critical_problems"]))
            if self.dry_run:
                res.manifest = self._manifest(manifest, validation, None)
                return res
            provider = DatasetMarketData(self.dataset_dir)          # re-verifies hashes
            cfg = self._engine_config()
            strategy = SMCStrategy.from_config(cfg)
            engine = BacktestEngine.from_config(cfg, strategy, run_id=self.benchmark_id, data_root=self.root)
            df = provider.get_ohlcv(cfg["market"]["symbol"], cfg["market"]["timeframe"])
            bt = engine.run(df)
            res.backtest = bt
            res.metrics = self._metrics(bt, strategy, df)
            res.manifest = self._manifest(manifest, validation, res.metrics)
            res.files = self._write(res, bt)
        except (BenchmarkError, DatasetError, FileNotFoundError, ValueError) as exc:
            res.aborted, res.abort_reason = True, str(exc)
            if not self.dry_run and res.validation:      # only once a real dataset was examined
                res.files = self._write_aborted(res)
        return res

    def _load_manifest(self) -> Dict[str, Any]:
        m = load_manifest(self.dataset_dir)
        if m is None:
            raise BenchmarkError(f"{self.dataset_dir} is not a frozen dataset (no manifest.json). Build one with "
                                 f"`python src/main.py data export --dataset <id>` and point benchmark.dataset_directory "
                                 f"or data.directory at it. The synthetic sample is not a valid benchmark input.")
        problems = verify_dataset(self.dataset_dir)
        if problems:
            raise BenchmarkError("dataset hash verification failed: " + "; ".join(problems))
        sym, tf = self.config["market"]["symbol"], self.config["market"]["timeframe"]
        if m["primary"]["symbol"] != sym or m["primary"]["timeframe"] != tf:
            raise BenchmarkError(f"dataset primary is {m['primary']['symbol']} {m['primary']['timeframe']}, "
                                 f"config expects {sym} {tf}")
        return m

    def _engine_config(self) -> Dict[str, Any]:
        """Same trading config; only the data pointer (and optional benchmark window) is set for the engine."""
        cfg = copy.deepcopy(self.config)
        cfg["data"]["directory"] = self.dataset_rel
        if self.bcfg.start_date or self.bcfg.end_date:
            cfg.setdefault("backtesting", {})
            cfg["backtesting"]["start_date"] = self.bcfg.start_date
            cfg["backtesting"]["end_date"] = self.bcfg.end_date
        return cfg

    # -------------------------------------------------------------- validation gate
    def _validate(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        vcfg = ValidationConfig.from_config(self.config)
        vcfg.fail_on_warnings = self.bcfg.fail_on_warnings or vcfg.fail_on_warnings
        now_ts = pd.Timestamp(self.now)
        series: Dict[str, Any] = {}
        critical: List[str] = []
        warnings: List[str] = []
        prim = manifest["primary"]
        raw = read_series_csv(self.dataset_dir / prim["file"], prim["kind"])
        rep = validate_frame(raw, Path(prim["file"]).name.split(".")[0], prim["timeframe"], prim["kind"], vcfg, now=now_ts)
        series["primary"] = rep.to_dict()
        prim_ts = pd.to_datetime(raw["timestamp"], utc=True) if "timestamp" in raw else None
        for i in rep.issues:
            (critical if i.severity == "error" else warnings).append(f"primary {i.code}: {i.message}")

        # auxiliary feeds required by the CURRENT config must be present, valid and aligned
        aux_reports = {}
        enabled_specs = [s for s in aux_specs_from_config(self.config) if s.enabled]
        by_file = {a["file"].split(".")[0]: a for a in manifest.get("auxiliary", [])}
        for spec in enabled_specs:
            sid = aux_filename(spec.symbol, spec.timeframe)[:-4]
            entry = by_file.get(sid)
            if entry is None:
                critical.append(f"auxiliary feed '{spec.name}' ({sid}) is enabled in config but missing from the dataset")
                continue
            araw = read_series_csv(self.dataset_dir / entry["file"], entry["kind"])
            arep = validate_frame(araw, sid, entry["timeframe"], entry["kind"], vcfg, now=now_ts)
            d = arep.to_dict()
            if arep.ok and prim_ts is not None:
                d["alignment"] = alignment_preview(prim_ts, prim["timeframe"], pd.to_datetime(araw["timestamp"], utc=True),
                                                   entry["timeframe"])
                if d["alignment"]["aux_ends_before_primary"]:
                    warnings.append(f"{spec.name}: auxiliary series ends before the primary "
                                    f"({d['alignment']['aux_last_close_time']} < {d['alignment']['primary_last_close_time']})")
            aux_reports[spec.name] = d
            for i in arep.issues:
                (critical if i.severity == "error" else warnings).append(f"{spec.name} {i.code}: {i.message}")
        series["auxiliary"] = aux_reports

        status = "failed" if critical or (vcfg.fail_on_warnings and warnings) else ("warnings" if warnings else "ok")
        return {"status": status, "critical_problems": critical, "warnings": warnings, "series": series,
                "dataset_hash_verified": True, "fail_on_warnings": vcfg.fail_on_warnings,
                "checks": "V1-V10 (schema, tz/grid, order, duplicates, gaps, OHLC, outliers, closed last candle, hashes)"}

    # -------------------------------------------------------------- metrics
    def _metrics(self, bt, strategy, df: pd.DataFrame) -> Dict[str, Any]:
        m = dict(bt.metrics)
        trades = bt.journal.trades
        n = len(trades)
        enough = n >= self.bcfg.min_trades_for_statistics
        rs = [t.r_multiple for t in trades]
        pnls = [t.net_pnl for t in trades]

        def if_enough(v):
            return v if enough else None

        # consecutive losses (from the journal, in closing order)
        worst = cur = 0
        for p in pnls:
            cur = cur + 1 if p < 0 else 0
            worst = max(worst, cur)
        bars_held = [t.bars_held for t in trades]
        durations_h = None
        if n:
            durations_h = [(pd.Timestamp(t.exit_timestamp) - pd.Timestamp(t.entry_timestamp)).total_seconds() / 3600 for t in trades]

        diag = getattr(strategy, "diag", None)
        buy_signals = int(diag.buy_signals) if diag else None
        rejections = bt.journal.rejections          # every executed BUY ends up in the journal when it closes
        slippage_pct = float(self.config["execution"].get("slippage_pct", 0.0))
        fee_pct = float(self.config["execution"].get("paper_fee_pct", 0.0))
        # slippage cost estimate: entries filled at open*(1+s) -> cost = qty*open*s ; stops at ref*(1-s)
        # estimated slippage cost, reversing the engine's fill formulas (entry ref*(1+s); stop/signal/target ref*(1-s))
        # under the SAME flags the engine used - a reporting figure, not part of any P&L calculation
        bcfg = self.config.get("backtesting", {}) or {}
        on_entries, on_stops, on_targets = (bool(bcfg.get("slippage_on_entries", True)), bool(bcfg.get("slippage_on_stops", True)),
                                            bool(bcfg.get("slippage_on_targets", False)))
        slip_cost = 0.0
        if slippage_pct:
            s = slippage_pct / 100.0
            for t in trades:
                if on_entries:
                    slip_cost += t.quantity * (t.entry_price - t.entry_price / (1 + s))
                slipped = (t.exit_reason in (EXIT_STOP, EXIT_SIGNAL) and on_stops) or (t.exit_reason == EXIT_TARGET and on_targets)
                if slipped and s < 1:
                    slip_cost += t.quantity * (t.exit_price / (1 - s) - t.exit_price)

        funnel = {
            "buy_signals": buy_signals,
            "risk_rejected_buys": int(len(rejections)),
            "risk_approved_buys": (buy_signals - len(rejections)) if buy_signals is not None else None,
            "executed_buys": int(n),
            "closed_trades": int(sum(1 for t in trades if t.exit_reason != EXIT_END)),
            "force_closed_end_of_data": int(bt.metrics["exit_reasons"].get(EXIT_END, 0)),
            "note": "signal -> risk-approved -> executed (filled at next open) -> closed. Signals that were still "
                    "pending at the last bar are neither approved nor executed.",
        }
        out = {
            "label": None,  # filled by manifest
            "starting_equity": m["starting_equity"], "ending_equity": m["ending_equity"],
            "net_pnl": m["net_profit"], "return_pct": m["total_return_pct"],
            "total_trades": n, "winning_trades": m["winning_trades"], "losing_trades": m["losing_trades"],
            "breakeven_trades": m["breakeven_trades"],
            "win_rate_pct": if_enough(m["win_rate_pct"]),
            "expectancy": if_enough(m["expectancy"]),
            "average_r": if_enough(m["average_r_multiple"]),
            "median_r": if_enough(statistics.median(rs)) if n else None,
            "profit_factor": if_enough(m["profit_factor"]),
            "max_drawdown_pct": m["max_drawdown_pct"], "max_drawdown_bars": m["max_drawdown_bars"],
            "max_consecutive_losses": worst,
            "average_trade_duration_bars": (sum(bars_held) / n) if n else None,
            "average_trade_duration_hours": (sum(durations_h) / n) if n else None,
            "median_trade_duration_bars": statistics.median(bars_held) if n else None,
            "fees": {"fee_rate_pct": fee_pct, "total_fees": m["total_fees"], "broker_total_fees": m.get("broker_total_fees")},
            "slippage": {"slippage_pct": slippage_pct, "on_entries": on_entries, "on_stops": on_stops, "on_targets": on_targets,
                         "estimated_cost": slip_cost},
            "risk_rejections": m["risk_rejections"], "exit_reasons": m["exit_reasons"],
            "signal_funnel": funnel,
            "strategy_diagnostics": {"setups_armed": diag.setups_armed, "buy_signals": diag.buy_signals,
                                     "exit_signals": diag.exit_signals, "gate_failures": dict(diag.gate_failures),
                                     "riskoff_skips": diag.riskoff_skips} if diag else None,
            "bars": m["bars"], "risk_per_trade_pct": m["risk_per_trade_pct"],
            "gross_profit": m["gross_profit"], "gross_loss": m["gross_loss"],
            "statistics_available": enough,
            "min_trades_for_statistics": self.bcfg.min_trades_for_statistics,
            "unavailable_reason": None if enough else
                f"only {n} closed trades (< {self.bcfg.min_trades_for_statistics}); ratio metrics are not statistically meaningful",
        }
        return _clean(out)

    # -------------------------------------------------------------- manifest / artifacts
    def _manifest(self, dataset: Dict[str, Any], validation: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        snap = config_snapshot(self.config)
        synthetic = bool(dataset.get("synthetic", False))
        aux = [{"name": a.get("name"), "symbol": a["symbol"], "timeframe": a["timeframe"], "file": a["file"],
                "sha256": a["sha256"], "rows": a["rows"], "first_open": a["first_open"], "last_open": a["last_open"]}
               for a in dataset.get("auxiliary", [])]
        rng = {"start": self.bcfg.start_date or dataset["primary"]["first_open"],
               "end": self.bcfg.end_date or dataset["primary"]["last_open"]}
        return {
            "schema_version": BENCHMARK_SCHEMA, "benchmark_id": self.benchmark_id,
            "label": LABEL_SYNTHETIC if synthetic else LABEL_REAL, "synthetic": synthetic, "disclaimer": DISCLAIMER,
            "generated_utc": self.now.strftime("%Y-%m-%dT%H:%M:%SZ"), "repository_commit": self.commit,
            "dataset": {"directory": self.dataset_rel, "dataset_id": dataset["dataset_id"],
                        "dataset_sha256": dataset["dataset_sha256"], "created_utc": dataset.get("created_utc"),
                        "primary": {k: dataset["primary"].get(k) for k in ("symbol", "timeframe", "file", "sha256",
                                                                            "rows", "first_open", "last_open", "source")},
                        "auxiliary": aux},
            "strategy": {"name": "smc", "class": "SMCStrategy", "config_hash_trading": config_hash(self.config)},
            "configuration": {"snapshot_sha256": snapshot_hash(snap), "snapshot": snap},
            "symbol": self.config["market"]["symbol"], "timeframe": self.config["market"]["timeframe"],
            "starting_equity": float(self.config["risk"]["starting_balance"]),
            "fee_rate_pct": float(self.config["execution"].get("paper_fee_pct", 0.0)),
            "slippage_pct": float(self.config["execution"].get("slippage_pct", 0.0)),
            "benchmark_date_range": rng,
            "validation_status": validation["status"], "dry_run": self.dry_run,
            "metrics_available": metrics is not None,
            "safety": {"config_yaml_modified": False, "strategy_modified": False, "risk_modified": False,
                       "improvement_enabled": bool((self.config.get("improvement", {}) or {}).get("enabled", False)),
                       "proposals_applied": 0, "network_used": False},
        }

    def _write(self, res: BenchmarkResult, bt) -> Dict[str, Path]:
        from src.benchmark.report import render_markdown
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        (out / "manifest.json").write_text(json.dumps(res.manifest, indent=2, default=str), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(res.metrics, indent=2, default=str), encoding="utf-8")
        (out / "validation.json").write_text(json.dumps(res.validation, indent=2, default=str), encoding="utf-8")
        bt.journal.to_frame().to_csv(out / "trades.csv", index=False)
        bt.journal.rejections_frame().to_csv(out / "rejections.csv", index=False)
        bt.equity_curve.to_csv(out / "equity_curve.csv", index=False)
        (out / "report.md").write_text(render_markdown(res), encoding="utf-8")
        for name in ("manifest.json", "metrics.json", "validation.json", "trades.csv", "rejections.csv", "equity_curve.csv", "report.md"):
            files[name] = out / name
        return files

    def _write_aborted(self, res: BenchmarkResult) -> Dict[str, Path]:
        from src.benchmark.report import render_aborted
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(render_aborted(res), encoding="utf-8")
        (out / "validation.json").write_text(json.dumps(res.validation, indent=2, default=str), encoding="utf-8")
        return {"report.md": out / "report.md", "validation.json": out / "validation.json"}
