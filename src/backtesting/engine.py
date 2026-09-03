"""Point-in-time backtest engine.

Per candle i (strictly in this order):
  1. Fill a pending entry order at open[i] (signal was produced at close[i-1]).
  2. Check the open position's stop / target against candle i.
  3. smc.update(candle i)             -> SMC events with detected_index <= i only
  4. Build BacktestContext (bar i indicator row, live SMC state, account snapshot)
  5. signal = strategy.on_candle(ctx)
  6. BUY signal while flat  -> queue for the next open (entry_fill=next_open)
                              or fill now at close[i] (entry_fill=same_close)
     EXIT signal in a trade -> queue exit for the next open
  7. Update RiskState (mark-to-market, day roll) and append the equity curve row.
After the loop an open position may be force-closed at the last close ('end_of_data').

Fill model (all configurable in config.yaml -> backtesting):
  entry           : fill_price = ref * (1 + slippage)         [slippage_on_entries]
  stop-loss       : if open <= stop and gap_fill_at_open -> ref = open, else ref = stop
                    fill_price = ref * (1 - slippage)         [slippage_on_stops]
  take-profit     : fill_price = target (limit-like)          [slippage_on_targets -> * (1 - slippage)]
  conflict        : stop and target both touched in one candle -> stop first if
                    stop_first_on_conflict, else target first.
Indicators are computed once on the full frame; this is safe because every
indicator is prefix-invariant (row i depends only on rows <= i), which the
indicator tests assert and the anti-lookahead tests here re-check.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.backtesting.journal import (
    EXIT_END,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    RejectedProposal,
    TradeJournal,
    TradeRecord,
)
from src.backtesting.metrics import compute_metrics
from src.backtesting.result import BacktestResult
from src.backtesting.strategy import BUY, EXIT, BacktestContext, Signal, Strategy
from src.data.base import OHLCV_COLUMNS, Candle, MarketDataProvider
from src.execution.paper_broker import PaperBroker
from src.indicators.engine import IndicatorEngine
from src.risk.state import RiskState
from src.risk.validator import TradeValidator
from src.strategy.smc_engine import SMCEngine

ENTRY_NEXT_OPEN = "next_open"
ENTRY_SAME_CLOSE = "same_close"


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    timeframe: str
    starting_equity: float
    fee_pct: float
    slippage_pct: float
    risk_per_trade_pct: float
    entry_fill: str = ENTRY_NEXT_OPEN
    stop_first_on_conflict: bool = True
    gap_fill_at_open: bool = True
    slippage_on_entries: bool = True
    slippage_on_stops: bool = True
    slippage_on_targets: bool = False
    close_open_position_at_end: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __post_init__(self):
        if self.entry_fill not in (ENTRY_NEXT_OPEN, ENTRY_SAME_CLOSE):
            raise ValueError(f"entry_fill must be '{ENTRY_NEXT_OPEN}' or '{ENTRY_SAME_CLOSE}'")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be >= 0")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BacktestConfig":
        b = config.get("backtesting", {})
        return cls(
            symbol=config["market"]["symbol"],
            timeframe=config["market"]["timeframe"],
            starting_equity=float(config["risk"]["starting_balance"]),
            fee_pct=float(config["execution"]["paper_fee_pct"]),
            slippage_pct=float(config["execution"].get("slippage_pct", 0.0)),
            risk_per_trade_pct=float(config["risk"]["risk_per_trade_pct"]),
            entry_fill=b.get("entry_fill", ENTRY_NEXT_OPEN),
            stop_first_on_conflict=bool(b.get("stop_first_on_conflict", True)),
            gap_fill_at_open=bool(b.get("gap_fill_at_open", True)),
            slippage_on_entries=bool(b.get("slippage_on_entries", True)),
            slippage_on_stops=bool(b.get("slippage_on_stops", True)),
            slippage_on_targets=bool(b.get("slippage_on_targets", False)),
            close_open_position_at_end=bool(b.get("close_open_position_at_end", True)),
            start_date=b.get("start_date"),
            end_date=b.get("end_date"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _OpenPosition:
    trade_id: int
    signal_index: int
    signal_timestamp: Any
    entry_index: int
    entry_timestamp: Any
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: Optional[float]
    entry_fee: float
    risk_amount: float
    entry_reason: str
    exit_requested: bool = False


class BacktestEngine:
    def __init__(self, config: BacktestConfig, strategy: Strategy, indicator_engine: IndicatorEngine,
                 smc_factory, validator: TradeValidator, run_id: str = "backtest"):
        """`smc_factory` is a zero-arg callable returning a fresh SMCEngine (engines are stateful)."""
        self.cfg = config
        self.strategy = strategy
        self.indicator_engine = indicator_engine
        self.smc_factory = smc_factory
        self.validator = validator
        self.run_id = run_id

    @classmethod
    def from_config(cls, config: Dict[str, Any], strategy: Strategy, run_id: str = "backtest") -> "BacktestEngine":
        return cls(
            BacktestConfig.from_config(config),
            strategy,
            IndicatorEngine.from_config(config),
            lambda: SMCEngine.from_config(config),
            TradeValidator.from_config(config),
            run_id,
        )

    # ------------------------------------------------------------------ API
    def run_provider(self, provider: MarketDataProvider) -> BacktestResult:
        df = provider.get_ohlcv(self.cfg.symbol, self.cfg.timeframe)
        return self.run(df)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        df = self._prepare(df)
        n = len(df)
        if n:
            ind_rows = self.indicator_engine.compute(df)[self.indicator_engine.columns]
        else:
            ind_rows = pd.DataFrame(columns=self.indicator_engine.columns, dtype=float)

        broker = PaperBroker(self.cfg.starting_equity, self.cfg.fee_pct, self.cfg.symbol)
        risk = RiskState(self.cfg.starting_equity)
        smc = self.smc_factory()
        journal = TradeJournal()
        curve = []

        pending_entry: Optional[Signal] = None
        pending_entry_index = -1
        pending_entry_ts: Any = None
        pos: Optional[_OpenPosition] = None
        next_trade_id = 1

        candles = [Candle(r.timestamp, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
                   for r in df.itertuples(index=False)]

        for i, c in enumerate(candles):
            risk.new_day(_day_of(c.timestamp))

            # 1. pending entry fills at this open
            if pending_entry is not None and pos is None:
                pos, next_trade_id = self._try_enter(pending_entry, pending_entry_index, pending_entry_ts, i, c.open,
                                                     c.timestamp, broker, risk, journal, next_trade_id)
            pending_entry = None

            # 2. exits against this candle
            if pos is not None:
                exit_hit = self._check_exit(pos, c)
                if exit_hit is not None:
                    price, reason = exit_hit
                    self._close(pos, i, c.timestamp, price, reason, broker, risk, journal)
                    pos = None

            # 3. SMC update (only candles <= i are inside the engine)
            smc.update(c.timestamp, c.open, c.high, c.low, c.close)

            # 4-5. strategy sees bar i only
            equity_now = broker.equity(c.close)
            ctx = BacktestContext(i, c, ind_rows.iloc[i], smc.result, pos is not None, equity_now, risk.snapshot())
            signal = self.strategy.on_candle(ctx)

            # 6. route signal
            if signal is not None:
                if signal.side == BUY and pos is None:
                    if self.cfg.entry_fill == ENTRY_SAME_CLOSE:
                        pos, next_trade_id = self._try_enter(signal, i, c.timestamp, i, c.close, c.timestamp,
                                                             broker, risk, journal, next_trade_id)
                    else:
                        pending_entry, pending_entry_index, pending_entry_ts = signal, i, c.timestamp
                elif signal.side == EXIT and pos is not None:
                    pos.exit_requested = True

            # 7. bookkeeping
            equity_now = broker.equity(c.close)
            risk.update_equity(equity_now)
            curve.append({"timestamp": c.timestamp, "close": c.close, "cash": broker.cash,
                          "position": broker.position, "equity": equity_now})

        # end of data
        if pos is not None and self.cfg.close_open_position_at_end and candles:
            last = candles[-1]
            self._close(pos, n - 1, last.timestamp, last.close, EXIT_END, broker, risk, journal)
            pos = None
            curve[-1].update({"cash": broker.cash, "position": broker.position,
                              "equity": broker.equity(last.close)})

        curve_df = pd.DataFrame(curve, columns=["timestamp", "close", "cash", "position", "equity"])
        if len(curve_df):
            curve_df["peak"] = curve_df["equity"].cummax()
            curve_df["drawdown_pct"] = (curve_df["peak"] - curve_df["equity"]) / curve_df["peak"] * 100.0
        else:
            curve_df["peak"] = curve_df["drawdown_pct"] = pd.Series(dtype=float)

        metrics = compute_metrics(journal, curve_df, self.cfg.starting_equity, self.cfg.risk_per_trade_pct)
        metrics["broker_total_fees"] = broker.total_fees()
        return BacktestResult(self.run_id, self.cfg.symbol, self.cfg.timeframe, journal, curve_df, metrics,
                              self.cfg.to_dict())

    # -------------------------------------------------------------- internals
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV frame missing columns: {missing}")
        out = df.copy()
        if self.cfg.start_date or self.cfg.end_date:
            ts = pd.to_datetime(out["timestamp"], utc=True)
            if self.cfg.start_date:
                out = out[ts >= pd.Timestamp(self.cfg.start_date, tz="UTC")]
                ts = ts[ts >= pd.Timestamp(self.cfg.start_date, tz="UTC")]
            if self.cfg.end_date:
                out = out[ts <= pd.Timestamp(self.cfg.end_date, tz="UTC")]
        return out.reset_index(drop=True)

    def _slip(self, price: float, adverse_up: bool, enabled: bool) -> float:
        if not enabled or self.cfg.slippage_pct == 0:
            return price
        f = self.cfg.slippage_pct / 100.0
        return price * (1 + f) if adverse_up else price * (1 - f)

    def _try_enter(self, signal: Signal, signal_index: int, signal_ts: Any, i: int, ref_price: float, ts: Any,
                   broker: PaperBroker, risk: RiskState, journal: TradeJournal, trade_id: int):
        fill = self._slip(ref_price, adverse_up=True, enabled=self.cfg.slippage_on_entries)
        assessment = self.validator.validate_with_state(risk, fill, signal.stop_loss, account_equity=broker.equity(ref_price))
        if not assessment.approved:
            journal.add_rejection(RejectedProposal(i, ts, assessment.decision.value, assessment.reason, fill, signal.stop_loss))
            return None, trade_id

        sizing = assessment.sizing
        max_affordable = broker.cash / (fill * (1 + broker.fee_rate))
        qty = min(sizing.position_size, max_affordable)
        if qty <= 0:
            journal.add_rejection(RejectedProposal(i, ts, "REJECTED_INVALID_INPUT", "no affordable quantity", fill, signal.stop_loss))
            return None, trade_id

        trade = broker.buy(fill, qty, timestamp=ts)
        risk.open_position()
        pos = _OpenPosition(trade_id, signal_index, signal_ts, i, ts, trade.price, trade.quantity,
                            signal.stop_loss, signal.take_profit, trade.fee, qty * (fill - signal.stop_loss), signal.reason)
        return pos, trade_id + 1

    def _check_exit(self, pos: _OpenPosition, c: Candle):
        """Return (fill_price, reason) or None. Entries filled on this same bar's open are
        still exposed to this bar's range, which is realistic."""
        if pos.exit_requested:
            return self._slip(c.open, adverse_up=False, enabled=self.cfg.slippage_on_stops), EXIT_SIGNAL

        stop_hit = c.low <= pos.stop_loss
        target_hit = pos.take_profit is not None and c.high >= pos.take_profit

        def stop_fill():
            ref = c.open if (self.cfg.gap_fill_at_open and c.open <= pos.stop_loss) else pos.stop_loss
            return self._slip(ref, adverse_up=False, enabled=self.cfg.slippage_on_stops), EXIT_STOP

        def target_fill():
            return self._slip(pos.take_profit, adverse_up=False, enabled=self.cfg.slippage_on_targets), EXIT_TARGET

        if stop_hit and target_hit:
            return stop_fill() if self.cfg.stop_first_on_conflict else target_fill()
        if stop_hit:
            return stop_fill()
        if target_hit:
            return target_fill()
        return None

    def _close(self, pos: _OpenPosition, i: int, ts: Any, price: float, reason: str,
               broker: PaperBroker, risk: RiskState, journal: TradeJournal) -> None:
        trade = broker.sell(price, pos.quantity, timestamp=ts)
        risk.record_trade(trade.realized_pnl, day=_day_of(ts))
        risk.close_position()
        gross = (trade.price - pos.entry_price) * pos.quantity
        journal.add_trade(TradeRecord(
            trade_id=pos.trade_id, signal_index=pos.signal_index, signal_timestamp=pos.signal_timestamp,
            entry_index=pos.entry_index, entry_timestamp=pos.entry_timestamp, entry_price=pos.entry_price,
            quantity=pos.quantity, stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            exit_index=i, exit_timestamp=ts, exit_price=trade.price, exit_reason=reason,
            entry_fee=pos.entry_fee, exit_fee=trade.fee, gross_pnl=gross, net_pnl=trade.realized_pnl,
            r_multiple=trade.realized_pnl / pos.risk_amount if pos.risk_amount else 0.0,
            risk_amount=pos.risk_amount, bars_held=i - pos.entry_index, entry_reason=pos.entry_reason,
        ))


def _day_of(ts: Any):
    if isinstance(ts, (pd.Timestamp, dt.datetime)):
        return ts.date()
    if isinstance(ts, dt.date):
        return ts
    try:
        return pd.Timestamp(ts).date()
    except Exception:  # non-datetime keys (tests use "t0", "t1"): treat as one day
        return None
