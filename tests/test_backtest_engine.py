"""Backtest engine tests: fills, risk integration, journal, anti-lookahead, determinism, config overrides."""

import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting import (  # noqa: E402
    BUY,
    EXIT,
    EXIT_END,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    BacktestConfig,
    BacktestContext,
    BacktestEngine,
    FixedIntervalTestStrategy,
    Signal,
)
from src.data import CSVMarketData  # noqa: E402
from src.indicators import IndicatorConfig, IndicatorEngine  # noqa: E402
from src.main import load_config  # noqa: E402
from src.risk import RiskConfig, RiskDecision, TradeValidator  # noqa: E402
from src.strategy import SMCConfig, SMCEngine  # noqa: E402


# ------------------------------------------------------------------ helpers
def frame(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close[, volume])."""
    n = len(rows)
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="15min", tz="UTC"),
        "open": [float(r[0]) for r in rows], "high": [float(r[1]) for r in rows],
        "low": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows],
        "volume": [float(r[4]) if len(r) > 4 else 100.0 for r in rows],
    })


def flat(p, n):
    return [(p, p + 1, p - 1, p)] * n


def make_engine(strategy, fee=0.0, slippage=0.0, risk_pct=1.0, equity=10_000.0, **overrides):
    cfg = BacktestConfig("BTC/USDT", "15m", equity, fee, slippage, risk_pct, **overrides)
    ind = IndicatorEngine(IndicatorConfig(ema_periods=[3], atr_period=3, volume_ma_period=3))
    validator = TradeValidator(RiskConfig(risk_per_trade_pct=risk_pct, max_daily_loss_pct=3.0,
                                          max_drawdown_pct=10.0, max_consecutive_losses=5, max_open_positions=1))
    return BacktestEngine(cfg, strategy, ind, lambda: SMCEngine(SMCConfig(pivot_strength=2)), validator)


class ScriptedStrategy:
    """Emit a fixed signal on given bar indices."""

    def __init__(self, signals):
        self.signals = dict(signals)

    def on_candle(self, ctx: BacktestContext) -> Optional[Signal]:
        return self.signals.get(ctx.index)


class NoTradeStrategy:
    def on_candle(self, ctx):
        return None


def sample_df():
    cfg = load_config()
    return CSVMarketData(ROOT / cfg["data"]["directory"]).get_ohlcv(cfg["market"]["symbol"], cfg["market"]["timeframe"])


# ------------------------------------------------------------- fill model
def test_entry_fills_next_open_and_stop_fills_at_stop():
    # signal at bar 2 close (100); bar 3 open 101 -> entry; bar 5 low touches 95 -> stop
    rows = flat(100, 3) + [(101, 102, 100, 101), (101, 102, 100, 101), (101, 101, 94, 96), (96, 97, 95, 96)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=120.0)})
    res = make_engine(strat).run(frame(rows))
    assert len(res.journal.trades) == 1
    t = res.journal.trades[0]
    assert t.signal_index == 2 and t.entry_index == 3 and t.entry_price == 101.0
    assert t.exit_index == 5 and t.exit_price == 95.0 and t.exit_reason == EXIT_STOP
    assert t.bars_held == 2 and t.net_pnl == pytest.approx((95 - 101) * t.quantity)


def test_take_profit_fills_at_target_without_slippage():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 112, 99, 105), (105, 106, 104, 105)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=110.0)})
    res = make_engine(strat, slippage=0.5).run(frame(rows))
    t = res.journal.trades[0]
    assert t.entry_price == pytest.approx(100 * 1.005)   # entry slippage applied
    assert t.exit_price == 110.0 and t.exit_reason == EXIT_TARGET  # no target slippage


def test_same_candle_conflict_defaults_to_stop_first():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 115, 94, 100)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=110.0)})
    res = make_engine(strat).run(frame(rows))
    assert res.journal.trades[0].exit_reason == EXIT_STOP


def test_gap_through_stop_fills_at_open_by_default():
    rows = flat(100, 3) + [(100, 101, 99, 100), (90, 91, 89, 90)]  # opens at 90, stop 95
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0)})
    res = make_engine(strat).run(frame(rows))
    t = res.journal.trades[0]
    assert t.exit_price == 90.0 and t.exit_reason == EXIT_STOP


def test_stop_slippage_applied_adversely():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 94, 96)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0)})
    res = make_engine(strat, slippage=1.0).run(frame(rows))
    assert res.journal.trades[0].exit_price == pytest.approx(95 * 0.99)


def test_exit_signal_fills_next_open():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100), (103, 104, 102, 103), (103, 104, 102, 103)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0), 4: Signal(EXIT)})
    res = make_engine(strat).run(frame(rows))
    t = res.journal.trades[0]
    assert t.exit_index == 5 and t.exit_price == 103.0 and t.exit_reason == EXIT_SIGNAL


def test_open_position_closed_at_end_of_data_and_flagged():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 102)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0)})
    res = make_engine(strat).run(frame(rows))
    t = res.journal.trades[0]
    assert t.exit_reason == EXIT_END and t.exit_price == 102.0
    assert res.equity_curve["position"].iloc[-1] == 0.0
    assert res.metrics["ending_equity"] == pytest.approx(res.equity_curve["equity"].iloc[-1])


def test_entry_never_before_signal_bar_plus_one():
    df = sample_df()
    res = make_engine(FixedIntervalTestStrategy(interval_bars=10, atr_column="atr_3")).run(df)
    assert res.journal.trades
    for t in res.journal.trades:
        assert t.entry_index == t.signal_index + 1
        assert t.exit_index >= t.entry_index
        assert t.entry_price == pytest.approx(df["open"].iloc[t.entry_index])


# ---------------------------------------------------------- config overrides
def test_override_entry_fill_same_close():
    rows = flat(100, 3) + [(101, 102, 100, 101), (101, 102, 100, 101)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0)})
    res = make_engine(strat, entry_fill="same_close").run(frame(rows))
    t = res.journal.trades[0]
    assert t.entry_index == 2 and t.entry_price == 100.0 and t.signal_index == 2


def test_override_target_first_on_conflict():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 115, 94, 100)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=110.0)})
    res = make_engine(strat, stop_first_on_conflict=False).run(frame(rows))
    assert res.journal.trades[0].exit_reason == EXIT_TARGET and res.journal.trades[0].exit_price == 110.0


def test_override_gap_fill_at_stop_price():
    rows = flat(100, 3) + [(100, 101, 99, 100), (90, 91, 89, 90)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0)})
    res = make_engine(strat, gap_fill_at_open=False).run(frame(rows))
    assert res.journal.trades[0].exit_price == 95.0


def test_override_slippage_toggles():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 112, 99, 105)]
    sig = {2: Signal(BUY, stop_loss=95.0, take_profit=110.0)}
    a = make_engine(ScriptedStrategy(sig), slippage=1.0, slippage_on_entries=False, slippage_on_targets=True).run(frame(rows))
    assert a.journal.trades[0].entry_price == 100.0
    assert a.journal.trades[0].exit_price == pytest.approx(110 * 0.99)
    rows2 = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 94, 96)]
    b = make_engine(ScriptedStrategy({2: Signal(BUY, stop_loss=95.0)}), slippage=1.0, slippage_on_stops=False).run(frame(rows2))
    assert b.journal.trades[0].exit_price == 95.0


def test_override_no_close_at_end_leaves_position_open():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 102)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0)})
    res = make_engine(strat, close_open_position_at_end=False).run(frame(rows))
    assert res.journal.trades == []
    assert res.equity_curve["position"].iloc[-1] > 0
    assert res.metrics["ending_equity"] == pytest.approx(res.equity_curve["equity"].iloc[-1])


def test_date_range_filter():
    df = sample_df()
    start, end = str(df["timestamp"].iloc[50]), str(df["timestamp"].iloc[99])
    res = make_engine(NoTradeStrategy(), start_date=start, end_date=end).run(df)
    assert res.metrics["bars"] == 50
    assert res.equity_curve["timestamp"].iloc[0] == df["timestamp"].iloc[50]


def test_config_from_yaml_defaults():
    cfg = load_config()
    bc = BacktestConfig.from_config(cfg)
    assert bc.entry_fill == "next_open" and bc.stop_first_on_conflict and bc.gap_fill_at_open
    assert bc.slippage_on_entries and bc.slippage_on_stops and not bc.slippage_on_targets
    assert bc.close_open_position_at_end
    assert bc.fee_pct == cfg["execution"]["paper_fee_pct"] and bc.slippage_pct == cfg["execution"]["slippage_pct"]
    assert bc.starting_equity == cfg["risk"]["starting_balance"]
    with pytest.raises(ValueError):
        BacktestConfig("X", "1h", 1, 0, 0, 1, entry_fill="magic")


def test_yaml_override_changes_engine_behaviour():
    """Changing config.yaml values (not constructor args) must change the run."""
    base = load_config()
    df = sample_df()
    strat = lambda: FixedIntervalTestStrategy.from_config(base)  # noqa: E731
    default = BacktestEngine.from_config(base, strat()).run(df)
    changed = copy.deepcopy(base)
    changed["backtesting"]["entry_fill"] = "same_close"
    changed["backtesting"]["slippage_on_entries"] = False
    other = BacktestEngine.from_config(changed, strat()).run(df)
    assert default.journal.trades and other.journal.trades
    assert default.journal.trades[0].entry_index == default.journal.trades[0].signal_index + 1
    assert other.journal.trades[0].entry_index == other.journal.trades[0].signal_index
    assert other.journal.trades[0].entry_price == pytest.approx(df["close"].iloc[other.journal.trades[0].signal_index])


# --------------------------------------------------------- risk integration
def make_engine_limits(strategy, **risk_kw):
    """Engine whose RiskConfig limits can be set per test."""
    eng = make_engine(strategy)
    base = dict(risk_per_trade_pct=1.0, max_daily_loss_pct=3.0, max_drawdown_pct=10.0,
                max_consecutive_losses=5, max_open_positions=1)
    base.update(risk_kw)
    eng.validator = TradeValidator(RiskConfig(**base))
    return eng


def test_buy_signal_while_in_position_is_ignored_without_second_trade():
    rows = flat(100, 3) + [(100, 101, 99, 100)] * 4
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0), 4: Signal(BUY, stop_loss=90.0)})
    res = make_engine(strat).run(frame(rows))
    assert len(res.journal.trades) == 1 and res.journal.rejections == []


def test_consecutive_loss_lock_blocks_entries_and_is_journaled():
    # each cycle: entry at 100, stop at 95 hit -> loss. Daily limit lifted so only the streak lock applies.
    rows2 = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 94, 96), (96, 97, 95, 96)] * 9
    losses = ScriptedStrategy({i: Signal(BUY, stop_loss=95.0) for i in range(2, 29, 3)})
    res2 = make_engine_limits(losses, max_daily_loss_pct=50.0, max_drawdown_pct=50.0).run(frame(rows2))
    assert res2.metrics["losing_trades"] == 5 and res2.metrics["winning_trades"] == 0
    counts = res2.journal.rejection_counts()
    assert counts == {RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT.value: 4}  # 9 signals: 5 filled, 4 locked
    assert all(r.decision in RiskDecision.__members__ for r in res2.journal.rejections)
    assert all(r.index > res2.journal.trades[-1].exit_index for r in res2.journal.rejections)


def test_drawdown_lock_blocks_entries():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 84, 85), (85, 86, 84, 85)] * 3
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=85.0), 5: Signal(BUY, stop_loss=80.0), 8: Signal(BUY, stop_loss=80.0)})
    # 15% risk on one trade -> 15% drawdown after the stop, above the 10% limit
    res = make_engine_limits(strat, risk_per_trade_pct=15.0, max_daily_loss_pct=50.0).run(frame(rows))
    assert len(res.journal.trades) == 1
    assert res.journal.rejection_counts() == {RiskDecision.REJECTED_DRAWDOWN_LIMIT.value: 2}


def test_daily_loss_lock_blocks_further_entries_same_day():
    # 3% risk per trade, limit 3% daily loss -> after one full stop-out no more trades that day
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 89, 90), (90, 91, 89, 90)] * 3
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0), 5: Signal(BUY, stop_loss=80.0), 8: Signal(BUY, stop_loss=80.0)})
    res = make_engine(strat, risk_pct=3.0).run(frame(rows))
    assert len(res.journal.trades) == 1
    assert res.journal.rejection_counts() == {RiskDecision.REJECTED_DAILY_LOSS_LIMIT.value: 2}


def test_sizing_matches_validator_and_risk_amount():
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=98.0)})
    res = make_engine(strat).run(frame(rows))
    t = res.journal.trades[0]
    assert t.quantity == pytest.approx(10_000 * 0.01 / 2.0)   # 50 units
    assert t.risk_amount == pytest.approx(100.0)


def test_affordability_guard_never_raises_insufficient_funds():
    # tight stop -> sizing wants more than cash; qty must be capped by cash incl. fee
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=99.95)})
    res = make_engine(strat, fee=0.1).run(frame(rows))
    t = res.journal.trades[0]
    assert t.quantity * 100 * 1.001 <= 10_000 + 1e-6
    assert res.equity_curve["cash"].iloc[3] >= -1e-9


# ------------------------------------------------------- broker consistency
def test_fees_and_equity_consistent_with_paper_broker():
    df = sample_df()
    res = make_engine(FixedIntervalTestStrategy(interval_bars=10, atr_column="atr_3"), fee=0.1, slippage=0.02).run(df)
    j = res.journal
    assert j.trades
    assert res.metrics["total_fees"] == pytest.approx(res.metrics["broker_total_fees"])
    assert sum(t.net_pnl for t in j.trades) == pytest.approx(res.metrics["ending_equity"] - 10_000)
    for t in j.trades:
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.entry_fee - t.exit_fee)
        assert t.entry_fee == pytest.approx(t.entry_price * t.quantity * 0.001)


# ---------------------------------------------------------------- edge cases
def test_no_signal_strategy_zero_trades_flat_curve():
    res = make_engine(NoTradeStrategy()).run(sample_df())
    m = res.metrics
    assert m["trades"] == 0 and m["win_rate_pct"] == 0.0 and m["profit_factor"] == 0.0
    assert (res.equity_curve["equity"] == 10_000).all() and m["max_drawdown_pct"] == 0.0


def test_empty_and_tiny_frames():
    res = make_engine(NoTradeStrategy()).run(frame([]))
    assert res.metrics["trades"] == 0 and res.metrics["bars"] == 0 and res.metrics["ending_equity"] == 10_000
    res = make_engine(ScriptedStrategy({0: Signal(BUY, stop_loss=90.0)})).run(frame(flat(100, 1)))
    assert res.journal.trades == []  # signal on the last bar can never fill


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        make_engine(NoTradeStrategy()).run(pd.DataFrame({"close": [1.0]}))


def test_signal_validation():
    with pytest.raises(ValueError):
        Signal("short", stop_loss=1.0)
    with pytest.raises(ValueError):
        Signal(BUY)


def test_result_save_and_summary(tmp_path):
    res = make_engine(FixedIntervalTestStrategy(interval_bars=10, atr_column="atr_3")).run(sample_df())
    out = res.save(tmp_path)
    assert {p.name for p in out.iterdir()} == {"trades.csv", "rejections.csv", "equity_curve.csv", "summary.json"}
    assert "profit factor" in res.format_summary()


# --------------------------------------------------------------- anti-lookahead
class SpyStrategy:
    """Asserts at every bar that the context contains only point-in-time information."""

    def __init__(self, df, ind_engine):
        self.df = df
        self.ind_engine = ind_engine
        self.calls = 0

    def on_candle(self, ctx: BacktestContext):
        i = ctx.index
        self.calls += 1
        # SMC: nothing from the future
        for coll in (ctx.smc.swing_highs, ctx.smc.swing_lows, ctx.smc.bos_events, ctx.smc.liquidity_sweeps,
                     ctx.smc.fair_value_gaps, ctx.smc.order_blocks):
            assert all(e.detected_index <= i for e in coll), f"future SMC event visible at bar {i}"
        # indicators: row i identical to computing on the prefix only
        prefix = self.ind_engine.compute(self.df.iloc[: i + 1])
        row = prefix[self.ind_engine.columns].iloc[i]
        pd.testing.assert_series_equal(ctx.indicators, row, check_names=False)
        # candle is bar i
        assert ctx.candle.close == self.df["close"].iloc[i]
        assert not hasattr(ctx, "df")
        return None


def test_context_is_point_in_time():
    df = sample_df().iloc[:80]
    ind = IndicatorEngine(IndicatorConfig(ema_periods=[3], atr_period=3, volume_ma_period=3))
    spy = SpyStrategy(df, ind)
    eng = make_engine(spy)
    eng.run(df)
    assert spy.calls == 80


def test_prefix_run_equals_full_run_up_to_cut():
    df = sample_df()
    strat = lambda: FixedIntervalTestStrategy(interval_bars=10, atr_column="atr_3")  # noqa: E731
    full = make_engine(strat(), fee=0.1, slippage=0.02, close_open_position_at_end=False).run(df)
    for cut in (60, 123, 177):
        part = make_engine(strat(), fee=0.1, slippage=0.02, close_open_position_at_end=False).run(df.iloc[:cut])
        pd.testing.assert_frame_equal(part.equity_curve, full.equity_curve.iloc[:cut].reset_index(drop=True))
        full_closed = [t for t in full.journal.trades if t.exit_index < cut]
        assert [t.to_dict() for t in part.journal.trades] == [t.to_dict() for t in full_closed]


def test_future_shock_does_not_change_past():
    df = sample_df()
    strat = lambda: FixedIntervalTestStrategy(interval_bars=10, atr_column="atr_3")  # noqa: E731
    base = make_engine(strat(), close_open_position_at_end=False).run(df)
    shocked = df.copy()
    k = 120
    shocked.loc[k:, ["open", "high", "low", "close"]] *= 3.0
    shocked.loc[k:, "volume"] *= 10
    other = make_engine(strat(), close_open_position_at_end=False).run(shocked)
    pd.testing.assert_frame_equal(base.equity_curve.iloc[:k], other.equity_curve.iloc[:k])
    b = [t.to_dict() for t in base.journal.trades if t.exit_index < k]
    o = [t.to_dict() for t in other.journal.trades if t.exit_index < k]
    assert b == o
    assert not base.equity_curve.iloc[k:].equals(other.equity_curve.iloc[k:])


# ---------------------------------------------------------------- determinism
def test_two_identical_runs_are_identical():
    df = sample_df()
    cfg = load_config()
    a = BacktestEngine.from_config(cfg, FixedIntervalTestStrategy.from_config(cfg)).run(df)
    b = BacktestEngine.from_config(cfg, FixedIntervalTestStrategy.from_config(cfg)).run(df)
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)
    assert a.journal.to_frame().equals(b.journal.to_frame())
    assert a.summary_dict() == b.summary_dict()


def test_sample_data_smoke_run():
    """Synthetic data + fixture strategy: verifies the pipeline only; numbers mean nothing."""
    cfg = load_config()
    res = BacktestEngine.from_config(cfg, FixedIntervalTestStrategy.from_config(cfg)).run_provider(
        CSVMarketData(ROOT / cfg["data"]["directory"]))
    m = res.metrics
    assert m["bars"] == 200 and m["trades"] >= 1
    assert m["trades"] == m["winning_trades"] + m["losing_trades"] + m["breakeven_trades"]
    assert m["risk_per_trade_pct"] == cfg["risk"]["risk_per_trade_pct"]
    assert 0 <= m["max_drawdown_pct"] <= 100
