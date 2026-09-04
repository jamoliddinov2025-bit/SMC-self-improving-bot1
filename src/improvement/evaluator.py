"""Evaluate candidates on data slices with the EXISTING BacktestEngine / SMCStrategy / compute_metrics.

For a slice [start, end) the engine is fed `warmup + slice` bars (warm-up = indicator warm-up +
setup_max_age_bars + a structure margin) so indicators and SMC structure are formed by the time
the slice begins. Only what happens INSIDE the slice counts:

  * trades whose entry timestamp is before the slice start are discarded,
  * the equity curve is re-based to the slice start (starting equity = equity at the bar before
    the slice, so a warm-up trade cannot inflate/deflate the slice's return),
  * metrics are then produced by `compute_metrics` - no second metrics engine.

A trade that is still open at the slice end is force-closed at the last close of the slice by the
engine itself (`close_open_position_at_end` is forced True for evaluation) so every slice is
self-contained. Every (candidate, slice) result is cached in memory and keyed deterministically.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.journal import TradeJournal
from src.backtesting.metrics import compute_metrics
from src.improvement.candidates import Candidate
from src.improvement.space import apply_overlay
from src.improvement.splitter import Slice
from src.indicators.engine import IndicatorConfig
from src.strategy.smc_strategy import SMCStrategy

STRUCTURE_MARGIN_BARS = 50


@dataclass
class SliceResult:
    candidate_id: str
    slice_name: str
    role: str
    metrics: Dict[str, Any]
    trades: pd.DataFrame
    warmup_bars: int
    warmup_trades_discarded: int
    diag: Dict[str, Any] = field(default_factory=dict)


def warmup_bars_for(config: Dict[str, Any]) -> int:
    ind = IndicatorConfig.from_config(config).warmup_bars
    age = int(config["strategy"]["entry"].get("setup_max_age_bars", 60))
    return ind + age + STRUCTURE_MARGIN_BARS


class Evaluator:
    def __init__(self, config: Dict[str, Any], df: pd.DataFrame, aux_frames: Optional[Dict[str, pd.DataFrame]] = None,
                 data_root=None):
        self.config = config
        self.df = df.reset_index(drop=True)
        self.aux_frames = aux_frames
        self.data_root = data_root
        self.warmup = warmup_bars_for(config)
        self._cache: Dict[Tuple[str, str], SliceResult] = {}
        self.backtests_run = 0

    # ---------------------------------------------------------------- API
    def evaluate(self, cand: Candidate, s: Slice) -> SliceResult:
        key = (cand.id, s.name)
        if key not in self._cache:
            self._cache[key] = self._run(cand, s)
        return self._cache[key]

    # ------------------------------------------------------------ internal
    def _config_for(self, cand: Candidate) -> Dict[str, Any]:
        cfg = apply_overlay(self.config, cand.overlay)
        b = cfg.setdefault("backtesting", {})
        b["close_open_position_at_end"] = True   # slices are self-contained
        b["start_date"] = None
        b["end_date"] = None
        b["save_results"] = False
        return cfg

    def _run(self, cand: Candidate, s: Slice) -> SliceResult:
        cfg = self._config_for(cand)
        feed_start = max(0, s.start - self.warmup)
        frame = self.df.iloc[feed_start:s.end].reset_index(drop=True)
        slice_start_ts = pd.Timestamp(self.df["timestamp"].iloc[s.start])
        strategy = SMCStrategy.from_config(cfg)
        engine = BacktestEngine.from_config(cfg, strategy, run_id=f"imp_{cand.id}_{s.name}",
                                            data_root=self.data_root, aux_frames=self.aux_frames)
        res = engine.run(frame)
        self.backtests_run += 1

        # keep only what happened inside the slice
        all_trades = res.journal.trades
        kept = [t for t in all_trades if pd.Timestamp(t.entry_timestamp) >= slice_start_ts]
        discarded = len(all_trades) - len(kept)
        journal = TradeJournal(trades=list(kept),
                               rejections=[r for r in res.journal.rejections if pd.Timestamp(r.timestamp) >= slice_start_ts])
        offset = s.start - feed_start                  # rows of warm-up fed
        curve = res.equity_curve.iloc[offset:].reset_index(drop=True)
        # Re-base the slice on the equity held just before it starts and rebuild the path from the kept
        # trades only (mark-to-market while open, realised PnL at exit). With no warm-up trade this is
        # identical to the engine's own curve (cash + position*close); with one, the warm-up trade's PnL
        # is excluded from the slice exactly as its trade record is.
        start_equity = float(res.equity_curve["equity"].iloc[offset - 1]) if offset > 0 else float(cfg["risk"]["starting_balance"])
        curve = _rebuild_curve(curve, kept, start_equity)
        metrics = compute_metrics(journal, curve, start_equity, float(cfg["risk"]["risk_per_trade_pct"]))
        metrics["warmup_trades_discarded"] = discarded
        diag = {}
        if hasattr(strategy, "diag"):
            d = strategy.diag
            diag = {"setups": d.setups_armed, "buys": d.buy_signals, "exits": d.exit_signals,
                    "gate_failures": dict(d.gate_failures)}
        return SliceResult(cand.id, s.name, s.role, metrics, journal.to_frame(), offset, discarded, diag)


def _rebuild_curve(curve: pd.DataFrame, kept_trades: List[Any], start_equity: float) -> pd.DataFrame:
    """Equity path driven only by `kept_trades`: mark-to-market while open, realised PnL from the exit bar on."""
    out = curve[["timestamp", "close"]].copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    equity = pd.Series(start_equity, index=out.index, dtype=float)
    for t in kept_trades:
        entry, exit_ = pd.Timestamp(t.entry_timestamp), pd.Timestamp(t.exit_timestamp)
        open_mask = (ts >= entry) & (ts < exit_)
        equity[open_mask] += (out.loc[open_mask, "close"] - t.entry_price) * t.quantity - t.entry_fee
        equity[ts >= exit_] += t.net_pnl
    out["cash"] = equity
    out["position"] = 0.0
    out["equity"] = equity
    return out
