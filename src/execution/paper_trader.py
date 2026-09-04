"""Paper-trading loop (Step 8). PAPER ONLY - no exchange connection, no API keys, no live orders.

`PaperTrader.process_candle(candle)` executes exactly the BacktestEngine's per-bar steps
0-7 for ONE closed primary candle, then persists its state atomically. Running a whole
replay through it therefore produces the same trades/equity as `BacktestEngine.run`
(with `close_open_position_at_end=False`) - the consistency tests assert this.

Per closed candle (same order as the backtester):
  0. advance every auxiliary feed (USDT.D, ...) to the candle's CLOSE time - never beyond
  1. fill the pending entry (signal from the previous close) at this open -> TradeValidator -> PaperBroker
  2. stop / target / requested-exit against this candle (stop-first on conflict, gap fill at open)
  3. SMCEngine.update(candle)
  4. indicator row for this candle (recomputed on the stored history; all indicators are prefix-invariant)
  5. strategy.on_candle(BacktestContext)
  6. BUY while flat -> pending entry for the next open; EXIT in a trade -> exit at the next open
  7. RiskState mark-to-market + day roll, journal / candle log rows, atomic state write

Persistence & restart: `data/paper/state.json` (account, risk, position, pending entry,
cursor, strategy snapshot, aux cursors, config hash) + `history.csv` (every accepted
candle). On restart the SMC engine and indicators are rebuilt by replaying `history.csv`
(last `paper.warmup_bars` rows, 0 = all) with the strategy disconnected, then the
strategy snapshot is re-attached to the rebuilt SMC result. Nothing is re-executed:
no duplicate entries, no lost risk state.

Safety: malformed candles are rejected before anything runs; any exception inside the
loop halts the trader without writing state and drops the pending entry, so a restart
resumes from the last good candle. Risk validation is never bypassed - the only way into
the broker is `BacktestEngine.try_enter`.
"""

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.backtesting.aux_data import REGIME_FEED, AuxFeed, AuxReplayer, build_aux_feeds, timeframe_to_timedelta
from src.backtesting.engine import BacktestConfig, BacktestEngine, _day_of, _OpenPosition
from src.backtesting.journal import RejectedProposal, TradeJournal, TradeRecord
from src.backtesting.strategy import BUY, EXIT, BacktestContext, Signal, Strategy
from src.data.base import OHLCV_COLUMNS, Candle, MarketDataProvider
from src.execution.candle_log import CANDLE_COLUMNS, CsvAppender
from src.execution.paper_broker import PaperBroker
from src.execution.state_store import StateStore, config_hash
from src.indicators.engine import IndicatorEngine
from src.risk.state import RiskState
from src.risk.validator import TradeValidator
from src.strategy.smc_engine import SMCEngine
from src.strategy.smc_types import BEARISH, BULLISH

log = logging.getLogger("paper_trader")

STATUS_OK = "processed"
STATUS_MALFORMED = "rejected_malformed"
STATUS_DUPLICATE = "skipped_duplicate"
STATUS_OUT_OF_ORDER = "skipped_out_of_order"
STATUS_HALTED = "halted"
STATUS_ERROR = "error"


class ConfigMismatch(RuntimeError):
    """The saved state was produced by a different trading configuration."""


@dataclass(frozen=True)
class PaperConfig:
    state_directory: str = "data/paper/"
    warmup_bars: int = 0            # history rows replayed on restart. 0 = ALL (exactly reproduces an
                                    # uninterrupted run). A cap bounds restart time/memory but recursive
                                    # indicators (EMA, Wilder ATR) and SMC structure then see a shorter
                                    # history, so results may differ slightly from an uninterrupted run.
    allow_config_change: bool = False
    strategy: str = "smc"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PaperConfig":
        p = config.get("paper", {}) or {}
        return cls(str(p.get("state_directory", "data/paper/")), int(p.get("warmup_bars", 0)),
                   bool(p.get("allow_config_change", False)), str(p.get("strategy", "smc")))


@dataclass
class CandleReport:
    """What happened on one candle (also written as a row to candles.csv)."""
    bar_index: int
    timestamp: Any
    status: str
    signal: Optional[Signal] = None
    risk_decision: Optional[str] = None
    risk_reason: Optional[str] = None
    fill_side: Optional[str] = None
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    exit_reason: Optional[str] = None
    equity: Optional[float] = None
    has_position: bool = False
    regime: Optional[str] = None
    note: str = ""
    row: Dict[str, Any] = field(default_factory=dict)


class PaperTrader:
    def __init__(self, config: Dict[str, Any], strategy: Strategy, state_dir: Optional[Union[str, Path]] = None,
                 aux_feeds: Optional[List[AuxFeed]] = None, data_root: Optional[Union[str, Path]] = None,
                 reset: bool = False):
        self.config = config
        self.pcfg = PaperConfig.from_config(config)
        root = Path(data_root) if data_root is not None else Path.cwd()
        self.state_dir = Path(state_dir) if state_dir is not None else root / self.pcfg.state_directory
        self.strategy = strategy
        self.strategy_name = self.pcfg.strategy
        self.cfg = BacktestConfig.from_config(config)
        self.symbol, self.timeframe = self.cfg.symbol, self.cfg.timeframe
        self.tf_delta = timeframe_to_timedelta(self.timeframe)
        self.indicator_engine = IndicatorEngine.from_config(config)
        self.validator = TradeValidator.from_config(config)
        self.smc_config = config
        self.aux_feeds: List[AuxFeed] = (aux_feeds if aux_feeds is not None
                                         else build_aux_feeds(config, data_root=root))
        # the fill model / risk path is literally the backtester's own code
        self.engine = BacktestEngine(self.cfg, strategy, self.indicator_engine,
                                     lambda: SMCEngine.from_config(config), self.validator, run_id="paper",
                                     aux_feeds=self.aux_feeds)
        self.config_hash = config_hash(config)
        # additive metadata only (Step 10): which frozen dataset (if any) config.data.directory points at.
        # Never read back, never affects resume/validation/execution.
        self.dataset_info = _dataset_identity(root / config.get("data", {}).get("directory", "data"))

        self.store = StateStore(self.state_dir / "state.json")
        self.history_path = self.state_dir / "history.csv"
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.journal = TradeJournal()          # this process's session
        self.curve: List[Dict[str, Any]] = []  # this process's session
        self.warnings: List[str] = []

        if reset:
            self._wipe()
        state = self.store.load()
        if state is not None:
            self._restore(state)
        else:
            self._fresh()
        self._open_logs()
        self._save_state()

    # ------------------------------------------------------------ lifecycle
    def _fresh(self) -> None:
        self.broker = PaperBroker(self.cfg.starting_equity, self.cfg.fee_pct, self.symbol)
        self.risk = RiskState(self.cfg.starting_equity)
        self.smc = SMCEngine.from_config(self.smc_config)
        self.history: List[Candle] = []
        self.replayers: List[AuxReplayer] = [f.replayer() for f in self.aux_feeds]
        self.pos: Optional[_OpenPosition] = None
        self.pending_entry: Optional[Signal] = None
        self.pending_entry_index = -1
        self.pending_entry_ts: Any = None
        self.bar_index = -1                  # index of the last processed candle (backtester's i)
        self.last_timestamp: Optional[pd.Timestamp] = None
        self.next_trade_id = 1
        self.index_shift = 0                 # bars of history dropped by a capped warm-up
        self.total_trades = 0
        self.history_rows = 0

    def _restore(self, state: Dict[str, Any]) -> None:
        if state.get("config_hash") != self.config_hash:
            if not self.pcfg.allow_config_change:
                raise ConfigMismatch("state.json was written with a different trading configuration; "
                                     "run with --reset or set paper.allow_config_change: true")
            self.warnings.append("config changed since last run (paper.allow_config_change=true)")
        if state.get("symbol") != self.symbol or state.get("timeframe") != self.timeframe:
            raise ConfigMismatch("state.json belongs to a different symbol/timeframe")
        if state.get("strategy_name") != self.strategy_name:
            raise ConfigMismatch(f"state.json was written by strategy {state.get('strategy_name')!r}, "
                                 f"not {self.strategy_name!r}; run with --reset")

        if state.get("halted"):
            self.warnings.append(f"previous run halted: {state.get('halt_reason')}; resuming from last good candle")
        self.broker = PaperBroker.from_snapshot(state["broker"])
        self.risk = RiskState.from_snapshot(state["risk"])
        cur = state["cursor"]
        self.bar_index = int(cur["bar_index"])
        self.last_timestamp = pd.Timestamp(cur["last_timestamp"]) if cur.get("last_timestamp") else None
        self.next_trade_id = int(cur["next_trade_id"])
        self.total_trades = int(cur.get("total_trades", 0))
        self.history_rows = int(cur.get("history_rows", 0))

        # rebuild point-in-time engines from the stored candle history (strategy disconnected)
        hist = self._load_history()
        if len(hist) < self.history_rows:
            raise ValueError(f"history.csv has {len(hist)} rows but state.json expects {self.history_rows}")
        if len(hist) > self.history_rows:   # crash after the history append but before the state write
            self.warnings.append(f"dropping {len(hist) - self.history_rows} unconfirmed history row(s)")
            hist = hist.iloc[:self.history_rows].reset_index(drop=True)
            hist.assign(timestamp=hist["timestamp"].map(_ts)).to_csv(self.history_path, index=False)
        n_keep = len(hist) if self.pcfg.warmup_bars <= 0 else min(len(hist), self.pcfg.warmup_bars)
        self.index_shift = len(hist) - n_keep
        keep = hist.iloc[len(hist) - n_keep:].reset_index(drop=True)
        self.history = [Candle(r.timestamp, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
                        for r in keep.itertuples(index=False)]
        self.smc = SMCEngine.from_config(self.smc_config)
        self.replayers = [f.replayer() for f in self.aux_feeds]
        for c in self.history:
            close_time = pd.Timestamp(c.timestamp) + self.tf_delta
            for r in self.replayers:
                r.advance(close_time)
            self.smc.update(c.timestamp, c.open, c.high, c.low, c.close)

        p = state.get("position")
        self.pos = None
        if p:
            self.pos = _OpenPosition(**p)
            self.pos.signal_index -= self.index_shift
            self.pos.entry_index -= self.index_shift
        pe = state.get("pending_entry")
        if pe:
            self.pending_entry = Signal(**pe["signal"])
            self.pending_entry_index = int(pe["index"]) - self.index_shift
            self.pending_entry_ts = pe["timestamp"]
        else:
            self.pending_entry, self.pending_entry_index, self.pending_entry_ts = None, -1, None

        # consistency checks between account and position state
        if (self.pos is None) != (self.broker.position == 0.0):
            raise ValueError("corrupt state: broker position and open-position record disagree")
        if self.pos is not None and self.risk.open_positions != 1:
            raise ValueError("corrupt state: risk open_positions does not match the open position")

        ss = state.get("strategy")
        if ss and hasattr(self.strategy, "restore_snapshot"):
            self.warnings.extend(self.strategy.restore_snapshot(ss, self.smc.result, index_shift=-self.index_shift))
        log.info("restored paper state at bar %s (%s), %d history bars, position=%s, pending=%s",
                 self.bar_index, self.last_timestamp, len(self.history), self.pos is not None,
                 self.pending_entry is not None)

    def _wipe(self) -> None:
        for name in ("state.json", "history.csv", "candles.csv", "trades.csv", "rejections.csv"):
            p = self.state_dir / name
            if p.exists():
                p.unlink()

    def _open_logs(self) -> None:
        self.candle_log = CsvAppender(self.state_dir / "candles.csv", CANDLE_COLUMNS)
        self.trade_log = CsvAppender(self.state_dir / "trades.csv", list(TradeRecord.__dataclass_fields__))
        self.reject_log = CsvAppender(self.state_dir / "rejections.csv", list(RejectedProposal.__dataclass_fields__))
        self.history_log = CsvAppender(self.history_path, OHLCV_COLUMNS)

    def close(self) -> None:
        for lg in (self.candle_log, self.trade_log, self.reject_log, self.history_log):
            lg.close()

    def _load_history(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_csv(self.history_path)
        if len(df) == 0:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for c in OHLCV_COLUMNS[1:]:
            df[c] = df[c].astype(float)
        return df

    # --------------------------------------------------------------- state
    def _state_dict(self) -> Dict[str, Any]:
        strat = self.strategy.to_snapshot() if hasattr(self.strategy, "to_snapshot") else None
        # indices inside the strategy snapshot are in the current (possibly shifted) frame; store absolute
        return {
            "symbol": self.symbol, "timeframe": self.timeframe, "config_hash": self.config_hash,
            "strategy_name": self.strategy_name,
            "broker": self.broker.to_snapshot(),
            "risk": self.risk.to_snapshot(),
            "position": _pos_dict(self.pos, self.index_shift),
            "pending_entry": None if self.pending_entry is None else {
                "signal": asdict(self.pending_entry),
                "index": self.pending_entry_index + self.index_shift,
                "timestamp": _ts(self.pending_entry_ts),
            },
            "cursor": {
                "bar_index": self.bar_index, "last_timestamp": _ts(self.last_timestamp),
                "next_trade_id": self.next_trade_id, "total_trades": self.total_trades,
                "history_rows": self.history_rows,
                "aux": {r.name: r.fed_index for r in self.replayers},
            },
            "strategy": _shift_strategy_snapshot(strat, self.index_shift) if strat else None,
            "halted": self.halted, "halt_reason": self.halt_reason,
            "dataset": self.dataset_info,
        }

    def _save_state(self) -> None:
        self._last_good_state = self._state_dict()
        self.store.save(self._last_good_state)

    def _save_halted(self) -> None:
        """Persist the LAST GOOD state (pre-candle) with the pending order dropped and the halt flag set.
        The in-memory objects may be half-mutated, so they are deliberately not serialised here."""
        state = dict(getattr(self, "_last_good_state", None) or {})
        if not state:
            return
        state.update({"pending_entry": None, "halted": True, "halt_reason": self.halt_reason})
        try:
            self.store.save(state)
        except Exception:  # noqa: BLE001
            log.exception("could not persist halted state")

    # ------------------------------------------------------------- main loop
    def run_replay(self, provider: MarketDataProvider, limit: Optional[int] = None) -> List[CandleReport]:
        """Feed every candle of the provider once, in order (skips already-processed timestamps)."""
        reports = []
        for n, c in enumerate(provider.iter_candles(self.symbol, self.timeframe)):
            if limit is not None and n >= limit:
                break
            reports.append(self.process_candle(c))
            if self.halted:
                break
        return reports

    def process_candle(self, candle: Candle) -> CandleReport:
        if self.halted:
            return CandleReport(self.bar_index, candle.timestamp, STATUS_HALTED, note=self.halt_reason or "")
        problem = _validate_candle(candle)
        if problem:
            log.warning("malformed candle %s rejected: %s", candle.timestamp, problem)
            rep = CandleReport(self.bar_index, candle.timestamp, STATUS_MALFORMED, note=problem)
            self._log_candle(rep, candle)
            return rep
        ts = pd.Timestamp(candle.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if self.last_timestamp is not None:
            if ts == self.last_timestamp:
                return CandleReport(self.bar_index, ts, STATUS_DUPLICATE, note="timestamp already processed")
            if ts < self.last_timestamp:
                return CandleReport(self.bar_index, ts, STATUS_OUT_OF_ORDER, note="older than last processed")
        candle = Candle(ts, float(candle.open), float(candle.high), float(candle.low), float(candle.close),
                        float(candle.volume))
        try:
            rep = self._step(candle)
        except Exception as exc:  # noqa: BLE001 - fail safe: halt, drop pending order, keep last good state on disk
            self.halted, self.halt_reason = True, f"{type(exc).__name__}: {exc}"
            self.pending_entry = None
            log.exception("paper trader halted on candle %s", ts)
            self._save_halted()
            return CandleReport(self.bar_index, ts, STATUS_ERROR, note=self.halt_reason)
        self._save_state()
        return rep

    def _step(self, c: Candle) -> CandleReport:
        eng, broker, risk = self.engine, self.broker, self.risk
        i = len(self.history)                    # index within the in-memory history (== backtester's i
        rep = CandleReport(i + self.index_shift, c.timestamp, STATUS_OK)  # when warmup is uncapped)
        gate_before = dict(getattr(getattr(self.strategy, "diag", None), "gate_failures", {}) or {})
        n_bos, n_sw, n_fvg, n_ob = (len(self.smc.result.bos_events), len(self.smc.result.liquidity_sweeps),
                                    len(self.smc.result.fair_value_gaps), len(self.smc.result.order_blocks))

        risk.new_day(_day_of(c.timestamp))

        # 0. auxiliary feeds up to this bar's close
        close_time = pd.Timestamp(c.timestamp) + self.tf_delta
        aux_state = {r.name: r.advance(close_time) for r in self.replayers}
        regime_state = aux_state.get(REGIME_FEED)

        # 1. pending entry at this open (validator -> broker; the backtester's own code path)
        journal_before = (len(self.journal.trades), len(self.journal.rejections))
        if self.pending_entry is not None and self.pos is None:
            self.pos, self.next_trade_id = eng.try_enter(self.pending_entry, self.pending_entry_index,
                                                         self.pending_entry_ts, i, c.open, c.timestamp,
                                                         broker, risk, self.journal, self.next_trade_id)
            if self.pos is not None:
                rep.fill_side, rep.fill_price, rep.fill_qty = "buy", self.pos.entry_price, self.pos.quantity
                rep.risk_decision, rep.risk_reason = "APPROVED", "entry filled"
                log.info("BUY filled #%d %.6f @ %.2f stop %.2f tp %s", self.pos.trade_id, self.pos.quantity,
                         self.pos.entry_price, self.pos.stop_loss, self.pos.take_profit)
            else:
                rj = self.journal.rejections[-1]
                rep.risk_decision, rep.risk_reason = rj.decision, rj.reason
                log.info("entry rejected by risk engine: %s %s", rj.decision, rj.reason)
        self.pending_entry = None

        # 2. exits
        if self.pos is not None:
            hit = eng.check_exit(self.pos, c)
            if hit is not None:
                price, reason = hit
                eng.close_position(self.pos, i, c.timestamp, price, reason, broker, risk, self.journal)
                tr = self.journal.trades[-1]
                rep.fill_side, rep.fill_price, rep.fill_qty, rep.exit_reason = "sell", tr.exit_price, tr.quantity, reason
                log.info("EXIT #%d %s @ %.2f pnl %+.2f", tr.trade_id, reason, tr.exit_price, tr.net_pnl)
                self.pos = None
                self.total_trades += 1

        # 3-4. SMC + indicators on the stored history (prefix-invariant -> identical to a full-frame run)
        self.smc.update(c.timestamp, c.open, c.high, c.low, c.close)
        self.history.append(c)
        ind_row = self.indicator_engine.compute(_frame(self.history))[self.indicator_engine.columns].iloc[-1]

        # 5. strategy sees bar i only
        equity_now = broker.equity(c.close)
        ctx = BacktestContext(i, c, ind_row, self.smc.result, self.pos is not None, equity_now, risk.snapshot(),
                              regime=regime_state, aux=aux_state)
        signal = self.strategy.on_candle(ctx)

        # 6. route
        if signal is not None:
            rep.signal = signal
            if signal.side == BUY and self.pos is None:
                self.pending_entry, self.pending_entry_index, self.pending_entry_ts = signal, i, c.timestamp
                log.info("BUY signal at %s: stop %.2f tp %s (%s) -> next open", c.timestamp, signal.stop_loss,
                         signal.take_profit, signal.reason)
            elif signal.side == EXIT and self.pos is not None:
                self.pos.exit_requested = True
                log.info("EXIT signal at %s (%s) -> next open", c.timestamp, signal.reason)

        # 7. bookkeeping
        equity_now = broker.equity(c.close)
        risk.update_equity(equity_now)
        self.bar_index = i + self.index_shift
        self.last_timestamp = c.timestamp
        self.curve.append({"timestamp": c.timestamp, "close": c.close, "cash": broker.cash,
                           "position": broker.position, "equity": equity_now})
        rep.equity, rep.has_position = equity_now, self.pos is not None
        rep.regime = regime_state.regime.value if regime_state is not None and hasattr(regime_state, "regime") else None

        # journals (append-only CSVs) - written before the state file, so a crash between the two at
        # worst duplicates a log row on restart, never loses account state
        sh = self.index_shift
        for tr in self.journal.trades[journal_before[0]:]:
            d = tr.to_dict()
            d.update(signal_index=tr.signal_index + sh, entry_index=tr.entry_index + sh, exit_index=tr.exit_index + sh)
            self.trade_log.append(d)
        for rj in self.journal.rejections[journal_before[1]:]:
            self.reject_log.append({**rj.to_dict(), "index": rj.index + sh})
        self.history_log.append({"timestamp": _ts(c.timestamp), "open": c.open, "high": c.high, "low": c.low,
                                 "close": c.close, "volume": c.volume})
        self.history_rows += 1

        r = self.smc.result
        events = []
        for b in r.bos_events[n_bos:]:
            events.append(("CHoCH" if b.is_choch else "BOS") + ("+" if b.direction == BULLISH else "-"))
        events += ["SWEEP" + ("+" if s.direction == BULLISH else "-") for s in r.liquidity_sweeps[n_sw:]]
        events += ["FVG" + ("+" if g.direction == BULLISH else "-") for g in r.fair_value_gaps[n_fvg:]]
        events += ["OB" + ("+" if o.direction == BULLISH else "-") for o in r.order_blocks[n_ob:]]
        gate_after = dict(getattr(getattr(self.strategy, "diag", None), "gate_failures", {}) or {})
        gate_delta = {k: v - gate_before.get(k, 0) for k, v in gate_after.items() if v != gate_before.get(k, 0)}
        poi = getattr(self.strategy, "last_poi", None)
        setup = getattr(self.strategy, "setup", None)
        setup_idx = getattr(setup, "trigger_index", None)
        cols = self.indicator_engine.config
        rep.row = {
            "ema_fast": _f(ind_row.get(f"ema_{cols.ema_periods[0]}")),
            "ema_mid": _f(ind_row.get(f"ema_{cols.ema_periods[min(1, len(cols.ema_periods) - 1)]}")),
            "ema_slow": _f(ind_row.get(f"ema_{cols.ema_periods[-1]}")),
            "atr": _f(ind_row.get(f"atr_{cols.atr_period}")),
            "volume_ratio": _f(ind_row.get(f"volume_ratio_{cols.volume_ma_period}")),
            "regime": rep.regime,
            "regime_raw": regime_state.raw_regime.value if regime_state is not None and hasattr(regime_state, "raw_regime") else None,
            "aux": {k: _aux_str(v) for k, v in aux_state.items() if k != REGIME_FEED},
            "smc_structure": r.structure, "smc_events": events,
            "strategy_state": getattr(self.strategy, "state", None),
            "setup_trigger_index": None if setup_idx is None else setup_idx + self.index_shift,
            "poi_kind": None if poi is None else poi.kind,
            "poi_low": None if poi is None else poi.low, "poi_high": None if poi is None else poi.high,
            "gate_failures": gate_delta,
            "position_qty": broker.position,
            "entry_price": None if self.pos is None else self.pos.entry_price,
            "stop_loss": None if self.pos is None else self.pos.stop_loss,
            "take_profit": None if self.pos is None else self.pos.take_profit,
            "cash": broker.cash, "equity": equity_now, "unrealized_pnl": broker.unrealized_pnl(c.close),
            "realized_pnl_total": broker.realized_pnl, "daily_pnl": risk.daily_pnl,
            "drawdown_pct": risk.drawdown_pct, "consecutive_losses": risk.consecutive_losses,
        }
        self._log_candle(rep, c)
        return rep

    def _log_candle(self, rep: CandleReport, c: Candle) -> None:
        row = {"bar_index": rep.bar_index if rep.status == STATUS_OK else "", "timestamp": _ts(c.timestamp),
               "status": rep.status, "open": c.open, "high": c.high, "low": c.low, "close": c.close,
               "volume": c.volume, "note": rep.note}
        if rep.signal is not None:
            row.update({"signal_side": rep.signal.side, "signal_stop": rep.signal.stop_loss,
                        "signal_target": rep.signal.take_profit, "signal_reason": rep.signal.reason})
        row.update({"risk_decision": rep.risk_decision, "risk_reason": rep.risk_reason, "fill_side": rep.fill_side,
                    "fill_price": rep.fill_price, "fill_qty": rep.fill_qty, "exit_reason": rep.exit_reason})
        if rep.fill_side == "buy" and self.pos is not None:
            row["fill_fee"] = self.pos.entry_fee
        elif rep.fill_side == "sell" and self.journal.trades:
            row["fill_fee"] = self.journal.trades[-1].exit_fee
        row.update(rep.row)
        self.candle_log.append(row)

    # ------------------------------------------------------------ reporting
    def equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.curve, columns=["timestamp", "close", "cash", "position", "equity"])

    def status(self) -> Dict[str, Any]:
        price = self.history[-1].close if self.history else 0.0
        return {
            "bar_index": self.bar_index, "last_timestamp": _ts(self.last_timestamp), "halted": self.halted,
            "halt_reason": self.halt_reason, "position": None if self.pos is None else asdict(self.pos),
            "pending_entry": None if self.pending_entry is None else asdict(self.pending_entry),
            "portfolio": self.broker.portfolio(price), "risk": self.risk.snapshot(),
            "strategy_state": getattr(self.strategy, "state", None), "total_trades": self.total_trades,
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------- helpers
def _validate_candle(c: Candle) -> Optional[str]:
    try:
        vals = [float(c.open), float(c.high), float(c.low), float(c.close), float(c.volume)]
    except (TypeError, ValueError):
        return "non-numeric OHLCV"
    if any(not math.isfinite(v) for v in vals):
        return "non-finite OHLCV"
    o, h, l, cl, v = vals
    if min(o, h, l, cl) <= 0:
        return "non-positive price"
    if h < max(o, cl) or l > min(o, cl) or h < l:
        return "OHLC ordering violated"
    if v < 0:
        return "negative volume"
    try:
        pd.Timestamp(c.timestamp)
    except (TypeError, ValueError):
        return "bad timestamp"
    if c.timestamp is None or pd.isna(pd.Timestamp(c.timestamp)):
        return "bad timestamp"
    return None


def _ts(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).isoformat()
    except (TypeError, ValueError):
        return str(ts)


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _aux_str(v: Any) -> str:
    if v is None:
        return "none"
    if hasattr(v, "regime"):
        return v.regime.value
    if hasattr(v, "close"):
        return f"{v.close}"
    return str(v)


def _pos_dict(pos: Optional[_OpenPosition], shift: int) -> Optional[Dict[str, Any]]:
    if pos is None:
        return None
    d = asdict(pos)
    d["signal_timestamp"], d["entry_timestamp"] = _ts(d["signal_timestamp"]), _ts(d["entry_timestamp"])
    d["signal_index"] += shift
    d["entry_index"] += shift
    return d


def _frame(history: List[Candle]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [c.timestamp for c in history], "open": [c.open for c in history],
        "high": [c.high for c in history], "low": [c.low for c in history],
        "close": [c.close for c in history], "volume": [c.volume for c in history],
    })


def _dataset_identity(directory) -> Optional[Dict[str, Any]]:
    """{'dataset_id','dataset_sha256','synthetic'} when `directory` holds a manifest.json, else None."""
    try:
        from src.data.dataset import dataset_identity
        return dataset_identity(directory)
    except Exception:  # noqa: BLE001 - metadata must never break the trader
        return None


def _shift_strategy_snapshot(snap: Dict[str, Any], shift: int) -> Dict[str, Any]:
    """Strategy indices are relative to the (possibly truncated) in-memory history; persist them absolute."""
    if not shift:
        return snap
    out = dict(snap)
    out["last_exit_index"] = snap["last_exit_index"] + shift
    out["bos_vol"] = {str(int(k) + shift): v for k, v in (snap.get("bos_vol") or {}).items()}
    for key in ("trade", "pending_entry"):
        if snap.get(key):
            out[key] = {**snap[key], "entry_index": snap[key]["entry_index"] + shift}
    if snap.get("setup"):
        sd = dict(snap["setup"])
        sd["trigger_index"] += shift
        sd["impulse_start_index"] += shift
        sd["pois"] = [{**p, "created_index": p["created_index"] + shift} for p in sd["pois"]]
        out["setup"] = sd
    return out
