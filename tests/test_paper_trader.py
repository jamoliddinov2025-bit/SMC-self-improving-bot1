"""PaperTrader: one-candle-at-a-time loop, risk gating, SL/TP, persistence & restart, aux timing,
malformed data, and byte-for-byte consistency with BacktestEngine."""

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting import (  # noqa: E402
    BUY,
    EXIT,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    REGIME_FEED,
    AuxFeed,
    AuxPoint,
    BacktestContext,
    BacktestEngine,
    FixedIntervalTestStrategy,
    Signal,
)
from src.data import CSVMarketData, Candle  # noqa: E402
from src.data.base import MarketDataProvider  # noqa: E402
from src.execution.paper_trader import (  # noqa: E402
    STATUS_DUPLICATE,
    STATUS_ERROR,
    STATUS_HALTED,
    STATUS_MALFORMED,
    STATUS_OK,
    STATUS_OUT_OF_ORDER,
    ConfigMismatch,
    PaperTrader,
)
from src.main import load_config  # noqa: E402
from src.risk import RiskDecision  # noqa: E402
from src.strategy.regime import Regime  # noqa: E402
from src.strategy.smc_strategy import SMCStrategy  # noqa: E402


# ------------------------------------------------------------------ helpers
def frame(rows, start="2024-01-01"):
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(rows), freq="15min", tz="UTC"),
        "open": [float(r[0]) for r in rows], "high": [float(r[1]) for r in rows],
        "low": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows],
        "volume": [float(r[4]) if len(r) > 4 else 100.0 for r in rows],
    })


def flat(p, n):
    return [(p, p + 1, p - 1, p)] * n


def candles(df):
    return [Candle(r.timestamp, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
            for r in df.itertuples(index=False)]


class FrameProvider(MarketDataProvider):
    def __init__(self, df):
        self.df = df

    def get_ohlcv(self, symbol, timeframe, limit=None):
        return self.df if limit is None else self.df.tail(limit).reset_index(drop=True)


class ScriptedStrategy:
    def __init__(self, signals):
        self.signals = dict(signals)
        self.seen = []

    def on_candle(self, ctx: BacktestContext) -> Optional[Signal]:
        self.seen.append(ctx)
        return self.signals.get(ctx.index)


def cfg_small(**over):
    """Small indicator periods, pivot 2, no fees/slippage unless overridden, USDT.D off, no end-close."""
    cfg = copy.deepcopy(load_config())
    cfg["indicators"] = {"ema": {"periods": [3, 5, 8], "fast": 5, "slow": 8}, "atr": {"period": 3},
                         "volume": {"ma_period": 3, "spike_multiplier": 1.5}}
    cfg["strategy"]["structure"]["pivot_strength"] = 2
    cfg["execution"]["paper_fee_pct"] = 0.0
    cfg["execution"]["slippage_pct"] = 0.0
    cfg["usdtd"]["enabled"] = False
    cfg["backtesting"]["close_open_position_at_end"] = False
    cfg["risk"].update({"max_daily_loss_pct": 3.0, "max_drawdown_pct": 10.0, "max_consecutive_losses": 5})
    for k, v in over.items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


def trader(cfg, strategy, tmp_path, name="s", **kw):
    return PaperTrader(cfg, strategy, state_dir=tmp_path / name, data_root=ROOT, aux_feeds=kw.pop("aux_feeds", []), **kw)


def sample_cfg(**over):
    cfg = copy.deepcopy(load_config())
    cfg["backtesting"]["close_open_position_at_end"] = False
    for k, v in over.items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return cfg


def sample_provider():
    return CSVMarketData(ROOT / "data/sample")


def assert_same_as_backtest(bt, pt):
    pd.testing.assert_frame_equal(bt.equity_curve[["timestamp", "close", "cash", "position", "equity"]],
                                  pt.equity_curve())
    assert bt.journal.to_frame().equals(pt.journal.to_frame())
    assert bt.journal.rejections_frame().equals(pt.journal.rejections_frame())


# ------------------------------------------------------- one candle at a time
def test_processes_one_closed_candle_at_a_time_and_strategy_sees_only_bar_i(tmp_path):
    rows = flat(100, 8)
    strat = ScriptedStrategy({})
    t = trader(cfg_small(), strat, tmp_path)
    for i, c in enumerate(candles(frame(rows))):
        rep = t.process_candle(c)
        assert rep.status == STATUS_OK and rep.bar_index == i
        assert len(strat.seen) == i + 1
        ctx = strat.seen[-1]
        assert ctx.index == i and ctx.candle == c and t.smc.n == i + 1
        assert all(b.detected_index <= i for b in ctx.smc.bos_events)
    t.close()


def test_buy_fills_next_open_and_no_duplicate_buy_while_in_position(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100)] * 5
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0), 3: Signal(BUY, stop_loss=90.0),
                              4: Signal(BUY, stop_loss=90.0)})
    t = trader(cfg_small(), strat, tmp_path)
    reps = [t.process_candle(c) for c in candles(frame(rows))]
    assert reps[2].signal.side == BUY and reps[2].fill_side is None      # queued, not filled on the signal bar
    assert reps[3].fill_side == "buy" and reps[3].fill_price == 100.0    # filled at next open
    assert t.broker.position > 0 and t.pos is not None
    assert [r.fill_side for r in reps].count("buy") == 1                # BUYs at 3 and 4 ignored
    assert len(t.broker.trades) == 1 and t.journal.rejections == []
    t.close()


def test_exit_signal_fills_next_open_once_and_repeats_are_ignored(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100), (102, 103, 101, 102),
                           (104, 105, 103, 104), (104, 105, 103, 104)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=90.0), 4: Signal(EXIT), 5: Signal(EXIT), 6: Signal(EXIT)})
    t = trader(cfg_small(), strat, tmp_path)
    reps = [t.process_candle(c) for c in candles(frame(rows))]
    assert reps[5].fill_side == "sell" and reps[5].exit_reason == EXIT_SIGNAL and reps[5].fill_price == 102.0
    assert len(t.journal.trades) == 1 and t.pos is None and t.broker.position == 0.0
    assert [r.fill_side for r in reps].count("sell") == 1
    t.close()


def test_stop_and_target_are_executed_mechanically(tmp_path):
    # stop: entry 100 stop 95 -> low 94 hits -> fill 95 ; target: entry 100 tp 110 -> high 111 -> fill 110
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 94, 96), (96, 97, 95, 96),
                           (96, 97, 95, 96), (96, 97, 95, 96), (96, 111, 95, 108)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=110.0),
                              6: Signal(BUY, stop_loss=90.0, take_profit=110.0)})
    t = trader(cfg_small(**{"risk.max_daily_loss_pct": 50.0}), strat, tmp_path)
    reps = [t.process_candle(c) for c in candles(frame(rows))]
    assert reps[4].exit_reason == EXIT_STOP and reps[4].fill_price == 95.0
    assert reps[8].exit_reason == EXIT_TARGET and reps[8].fill_price == 110.0
    assert [tr.exit_reason for tr in t.journal.trades] == [EXIT_STOP, EXIT_TARGET]
    assert t.risk.consecutive_losses == 0
    t.close()


def test_stop_first_on_conflict_and_gap_fill_at_open(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (93, 112, 92, 100)]
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=95.0, take_profit=110.0)})
    t = trader(cfg_small(), strat, tmp_path)
    reps = [t.process_candle(c) for c in candles(frame(rows))]
    assert reps[4].exit_reason == EXIT_STOP and reps[4].fill_price == 93.0   # gap open below stop -> open
    t.close()


# ----------------------------------------------------------- risk gating
def test_risk_rejection_blocks_broker_and_is_logged(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 84, 85), (85, 86, 84, 85)] * 3
    strat = ScriptedStrategy({2: Signal(BUY, stop_loss=85.0), 5: Signal(BUY, stop_loss=80.0),
                              8: Signal(BUY, stop_loss=80.0)})
    cfg = cfg_small(**{"risk.risk_per_trade_pct": 15.0, "risk.max_daily_loss_pct": 50.0})
    t = trader(cfg, strat, tmp_path)
    reps = [t.process_candle(c) for c in candles(frame(rows))]
    assert len(t.broker.trades) == 2   # one buy + one sell only
    assert t.journal.rejection_counts() == {RiskDecision.REJECTED_DRAWDOWN_LIMIT.value: 2}
    assert reps[6].risk_decision == RiskDecision.REJECTED_DRAWDOWN_LIMIT.value and reps[6].fill_side is None
    rej = pd.read_csv(tmp_path / "s" / "rejections.csv")
    assert list(rej["decision"]) == [RiskDecision.REJECTED_DRAWDOWN_LIMIT.value] * 2
    t.close()


def test_daily_loss_lock_blocks_same_day_and_persists_across_restart(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 89, 90), (90, 91, 89, 90)] * 3
    sigs = {2: Signal(BUY, stop_loss=90.0), 5: Signal(BUY, stop_loss=80.0), 8: Signal(BUY, stop_loss=80.0)}
    cfg = cfg_small(**{"risk.risk_per_trade_pct": 3.0})
    cs = candles(frame(rows))
    t = trader(cfg, ScriptedStrategy(sigs), tmp_path)
    for c in cs[:7]:
        t.process_candle(c)
    assert len(t.journal.trades) == 1 and t.risk.daily_loss_pct >= 3.0
    t.close()
    # restart: the daily loss must still lock the next entry
    t2 = trader(cfg, ScriptedStrategy(sigs), tmp_path)
    assert t2.risk.daily_loss_pct == pytest.approx(t.risk.daily_loss_pct)
    reps = [t2.process_candle(c) for c in cs[7:]]
    assert t2.journal.rejection_counts() == {RiskDecision.REJECTED_DAILY_LOSS_LIMIT.value: 1}
    assert reps[2].risk_decision == RiskDecision.REJECTED_DAILY_LOSS_LIMIT.value
    t2.close()


def test_consecutive_loss_lock_survives_restart(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 94, 96), (96, 97, 95, 96)] * 9
    sigs = {i: Signal(BUY, stop_loss=95.0) for i in range(2, 29, 3)}
    cfg = cfg_small(**{"risk.max_daily_loss_pct": 50.0, "risk.max_drawdown_pct": 50.0})
    cs = candles(frame(rows))
    t = trader(cfg, ScriptedStrategy(sigs), tmp_path)
    for c in cs[:17]:
        t.process_candle(c)
    assert t.risk.consecutive_losses == 5 and t.journal.rejections == []
    t.close()
    t2 = trader(cfg, ScriptedStrategy(sigs), tmp_path)
    assert t2.risk.consecutive_losses == 5
    for c in cs[17:]:
        t2.process_candle(c)
    assert t2.journal.trades == [] and t2.journal.rejection_counts() == {
        RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT.value: 4}
    t2.close()


# -------------------------------------------------------- restart / recovery
def test_restart_with_open_position_no_duplicate_entry_and_stop_still_works(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 94, 96)]
    sigs = {2: Signal(BUY, stop_loss=95.0)}
    cs = candles(frame(rows))
    t = trader(cfg_small(), ScriptedStrategy(sigs), tmp_path)
    for c in cs[:5]:
        t.process_candle(c)
    assert t.pos is not None
    qty, cash = t.broker.position, t.broker.cash
    t.close()
    t2 = trader(cfg_small(), ScriptedStrategy(sigs), tmp_path)
    assert t2.pos is not None and t2.pos.trade_id == 1 and t2.pos.stop_loss == 95.0
    assert t2.broker.position == qty and t2.broker.cash == cash and t2.risk.open_positions == 1
    assert t2.bar_index == 4 and t2.smc.n == 5
    rep = t2.process_candle(cs[5])
    assert rep.exit_reason == EXIT_STOP and t2.pos is None and t2.broker.position == 0.0
    assert len(t2.broker.trades) == 1   # only the sell in this session -> no re-buy happened
    trades = pd.read_csv(tmp_path / "s" / "trades.csv")
    assert len(trades) == 1 and trades["trade_id"].iloc[0] == 1
    t2.close()


def test_restart_with_pending_entry_fills_exactly_once(tmp_path):
    rows = flat(100, 3) + [(100, 101, 99, 100), (100, 101, 99, 100)]
    sigs = {2: Signal(BUY, stop_loss=90.0)}
    cs = candles(frame(rows))
    t = trader(cfg_small(), ScriptedStrategy(sigs), tmp_path)
    for c in cs[:3]:
        t.process_candle(c)
    assert t.pending_entry is not None and t.pos is None
    t.close()
    t2 = trader(cfg_small(), ScriptedStrategy(sigs), tmp_path)
    assert t2.pending_entry is not None
    rep = t2.process_candle(cs[3])
    assert rep.fill_side == "buy" and t2.pos.signal_index == 2 and t2.pos.entry_index == 3
    t2.process_candle(cs[4])
    assert len(t2.broker.trades) == 1
    t2.close()


def test_replay_skips_duplicates_and_out_of_order(tmp_path):
    cs = candles(frame(flat(100, 6)))
    t = trader(cfg_small(), ScriptedStrategy({}), tmp_path)
    for c in cs[:4]:
        t.process_candle(c)
    assert t.process_candle(cs[3]).status == STATUS_DUPLICATE
    assert t.process_candle(cs[1]).status == STATUS_OUT_OF_ORDER
    assert t.bar_index == 3 and t.smc.n == 4
    t.close()
    # a re-run of the whole provider only processes the unseen tail
    t2 = trader(cfg_small(), ScriptedStrategy({}), tmp_path)
    reps = t2.run_replay(FrameProvider(frame(flat(100, 6))))
    assert [r.status for r in reps] == [STATUS_OUT_OF_ORDER] * 3 + [STATUS_DUPLICATE] + [STATUS_OK] * 2
    assert t2.bar_index == 5
    t2.close()


def test_config_change_refused_unless_allowed(tmp_path):
    cs = candles(frame(flat(100, 3)))
    t = trader(cfg_small(), ScriptedStrategy({}), tmp_path)
    for c in cs:
        t.process_candle(c)
    t.close()
    changed = cfg_small(**{"risk.risk_per_trade_pct": 2.0})
    with pytest.raises(ConfigMismatch):
        trader(changed, ScriptedStrategy({}), tmp_path)
    changed["paper"] = {"allow_config_change": True}
    t2 = trader(changed, ScriptedStrategy({}), tmp_path)
    assert t2.bar_index == 2 and t2.warnings
    t2.close()
    t3 = trader(changed, ScriptedStrategy({}), tmp_path, reset=True)
    assert t3.bar_index == -1
    t3.close()


def test_sample_smc_restart_at_every_bar_matches_uninterrupted_run(tmp_path):
    """The strongest recovery test: restart after EVERY candle (armed setups, in-trade, cooldown...)
    and the concatenated session must equal one uninterrupted BacktestEngine run."""
    cfg = sample_cfg(**{"strategy.filters.ema_trend.enabled": False})
    prov = sample_provider()
    bt = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT).run_provider(prov)
    assert bt.metrics["trades"] >= 2
    cs = list(prov.iter_candles(cfg["market"]["symbol"], cfg["market"]["timeframe"]))
    curves, trades, states = [], [], []
    for c in cs:
        t = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=tmp_path / "p", data_root=ROOT)
        assert not t.warnings
        rep = t.process_candle(c)
        assert rep.status == STATUS_OK
        curves.append(t.equity_curve())
        trades += t.journal.trades
        states.append(t.strategy.state)
        t.close()
    curve = pd.concat(curves, ignore_index=True)
    pd.testing.assert_frame_equal(bt.equity_curve[["timestamp", "close", "cash", "position", "equity"]], curve)
    assert [(x.entry_index, x.exit_index, x.exit_reason, x.net_pnl) for x in trades] == \
           [(x.entry_index, x.exit_index, x.exit_reason, x.net_pnl) for x in bt.journal.trades]
    assert "IN_TRADE" in states and "ARMED" in states
    # persisted CSVs are complete
    assert len(pd.read_csv(tmp_path / "p" / "trades.csv")) == len(bt.journal.trades)
    assert len(pd.read_csv(tmp_path / "p" / "candles.csv")) == len(cs)
    assert len(pd.read_csv(tmp_path / "p" / "history.csv")) == len(cs)


# ------------------------------------------------------------- USDT.D timing
def test_usdtd_regime_only_visible_after_4h_candle_close(tmp_path):
    aux = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC"),
                        "close": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5]})
    feed = AuxFeed("btcd", "BTC.D", "4h", aux)       # plain feed: state is the last CLOSED row
    strat = ScriptedStrategy({})
    t = trader(cfg_small(), strat, tmp_path, aux_feeds=[feed])
    cs = candles(frame(flat(100, 40)))               # 40 x 15m = 10h
    for c in cs:
        t.process_candle(c)
    for ctx in strat.seen:
        bar_close = ctx.candle.timestamp + pd.Timedelta(minutes=15)
        st = ctx.aux["btcd"]
        if bar_close < pd.Timestamp("2024-01-01 04:00", tz="UTC"):
            assert st is None
        else:
            assert isinstance(st, AuxPoint)
            assert st.timestamp + pd.Timedelta(hours=4) <= bar_close                 # already closed
            nxt = st.index + 1
            if nxt < len(aux):
                assert aux["timestamp"][nxt] + pd.Timedelta(hours=4) > bar_close      # next not yet closed
    # first visible exactly on the bar closing at 04:00 (open 03:45)
    first = next(ctx for ctx in strat.seen if ctx.aux["btcd"] is not None)
    assert first.candle.timestamp == pd.Timestamp("2024-01-01 03:45", tz="UTC")
    t.close()


def test_usdtd_regime_state_matches_backtester_and_survives_restart(tmp_path):
    cfg = sample_cfg()
    prov = sample_provider()
    seen_bt, seen_pt = [], []

    class Spy(SMCStrategy):
        def __init__(self, sink, *a, **k):
            super().__init__(*a, **k)
            self.sink = sink

        def on_candle(self, ctx):
            self.sink.append((ctx.index, None if ctx.regime is None else (ctx.regime.regime, ctx.regime.source_index)))
            return super().on_candle(ctx)

    BacktestEngine.from_config(cfg, Spy(seen_bt, SMCStrategy.from_config(cfg).cfg), data_root=ROOT).run_provider(prov)
    cs = list(prov.iter_candles(cfg["market"]["symbol"], cfg["market"]["timeframe"]))
    for cut in (0, 97, 150):
        t = PaperTrader(cfg, Spy(seen_pt, SMCStrategy.from_config(cfg).cfg), state_dir=tmp_path / "u", data_root=ROOT)
        end = {0: 97, 97: 150, 150: len(cs)}[cut]
        for c in cs[cut:end]:
            t.process_candle(c)
        t.close()
    assert seen_pt == seen_bt
    assert any(r and r[0] is Regime.RISING for _, r in seen_bt)   # the sample actually exercises the regime


def test_multiple_named_aux_feeds_exposed_and_only_usdtd_drives_regime(tmp_path):
    cfg = sample_cfg()
    aux_usdtd = pd.read_csv(ROOT / "data/sample/USDTD_4h.csv")
    shifted = aux_usdtd.copy()
    shifted["close"] = shifted["close"] * 3.0
    cfg["auxiliary"] = {"feeds": {"btcd": {"symbol": "BTC.D", "timeframe": "4h"}}}
    eng = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT,
                                     aux_frames={"btcd": shifted, REGIME_FEED: aux_usdtd})
    assert sorted(f.name for f in eng.aux_feeds) == ["btcd", REGIME_FEED]
    seen = []

    class Spy(SMCStrategy):
        def on_candle(self, ctx):
            seen.append(ctx)
            return super().on_candle(ctx)

    eng.strategy = Spy.from_config(cfg)
    res_multi = eng.run_provider(sample_provider())
    last = seen[-1]
    assert set(last.aux) == {"btcd", REGIME_FEED} and last.aux[REGIME_FEED] is last.regime
    assert isinstance(last.aux["btcd"], AuxPoint) and last.aux["btcd"].close == pytest.approx(3.0 * last.regime.close)
    # an extra feed no strategy consumes changes nothing
    res_single = BacktestEngine.from_config(sample_cfg(), SMCStrategy.from_config(cfg), data_root=ROOT).run_provider(sample_provider())
    pd.testing.assert_frame_equal(res_multi.equity_curve, res_single.equity_curve)
    # the PaperTrader builds the same feeds from config
    t = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=tmp_path / "m", data_root=ROOT,
                    aux_feeds=eng.aux_feeds)
    assert [f.name for f in t.aux_feeds] == [f.name for f in eng.aux_feeds]
    t.close()


def test_reserved_feed_name_and_missing_file_rejected():
    cfg = sample_cfg()
    cfg["auxiliary"] = {"feeds": {REGIME_FEED: {"symbol": "X", "timeframe": "4h"}}}
    with pytest.raises(ValueError):
        BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT)
    cfg["auxiliary"] = {"feeds": {"dxy": {"symbol": "DXY", "timeframe": "1d"}}}
    with pytest.raises(FileNotFoundError):
        BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT)
    cfg["auxiliary"]["feeds"]["dxy"]["enabled"] = False
    assert [f.name for f in BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT).aux_feeds] == [REGIME_FEED]


# ------------------------------------------------------------ error handling
def test_malformed_candles_rejected_without_side_effects(tmp_path):
    strat = ScriptedStrategy({0: Signal(BUY, stop_loss=90.0)})
    t = trader(cfg_small(), strat, tmp_path)
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    bad = [Candle(ts, 100, 99, 98, 100, 1),                 # high < open
           Candle(ts, 100, float("nan"), 98, 100, 1),        # NaN
           Candle(ts, 100, 101, 99, 100, -1),                # negative volume
           Candle(ts, 0, 101, 99, 100, 1),                   # non-positive price
           Candle(ts, "x", 101, 99, 100, 1)]                 # non-numeric
    for c in bad:
        assert t.process_candle(c).status == STATUS_MALFORMED
    assert t.bar_index == -1 and t.smc.n == 0 and strat.seen == [] and t.broker.trades == []
    assert not t.halted
    rows = pd.read_csv(tmp_path / "s" / "candles.csv")
    assert list(rows["status"]) == [STATUS_MALFORMED] * 5
    t.close()


def test_temporary_data_failure_creates_no_order_and_state_is_recoverable(tmp_path):
    class Boom(ScriptedStrategy):
        def on_candle(self, ctx):
            if ctx.index == 3:
                raise RuntimeError("feed glitch")
            return super().on_candle(ctx)

    sigs = {2: Signal(BUY, stop_loss=90.0)}
    cs = candles(frame(flat(100, 6)))
    t = trader(cfg_small(), Boom(sigs), tmp_path)
    for c in cs[:3]:
        t.process_candle(c)
    assert t.pending_entry is not None
    rep = t.process_candle(cs[3])
    assert rep.status == STATUS_ERROR and t.halted and t.pending_entry is None
    assert t.process_candle(cs[4]).status == STATUS_HALTED
    state = json.load(open(tmp_path / "s" / "state.json"))
    assert state["halted"] and state["cursor"]["bar_index"] == 2 and state["pending_entry"] is None
    t.close()
    # NOTE: bar 3 did fill the pending BUY in memory before the strategy raised, but nothing was persisted;
    # the restart resumes from the last good candle (bar 2) with no position and no pending order.
    t2 = trader(cfg_small(), ScriptedStrategy({}), tmp_path)
    assert t2.warnings and t2.bar_index == 2 and t2.pos is None and t2.pending_entry is None
    assert t2.broker.trades == [] and t2.broker.position == 0.0
    assert t2.process_candle(cs[3]).status == STATUS_OK
    assert t2.pos is None                       # the dropped pending order was NOT resurrected
    t2.close()


def test_exceptions_never_bypass_risk_validation(tmp_path):
    """Even when the strategy misbehaves after a BUY, the only path into the broker is the validator."""
    calls = []
    cs = candles(frame(flat(100, 3) + [(100, 101, 99, 100)] * 3))
    t = trader(cfg_small(), ScriptedStrategy({2: Signal(BUY, stop_loss=90.0)}), tmp_path)
    orig = t.validator.validate_with_state

    def spy(*a, **k):
        calls.append(a)
        return orig(*a, **k)

    t.validator.validate_with_state = spy
    for c in cs:
        t.process_candle(c)
    assert len(calls) == 1 and len(t.broker.trades) == 1
    t.close()


# --------------------------------------------------- consistency with backtester
def test_consistency_fixture_strategy_with_fees_and_slippage(tmp_path):
    cfg = sample_cfg()
    cfg["usdtd"]["enabled"] = False
    prov = sample_provider()
    bt = BacktestEngine.from_config(cfg, FixedIntervalTestStrategy.from_config(cfg), data_root=ROOT).run_provider(prov)
    pt = PaperTrader(cfg, FixedIntervalTestStrategy.from_config(cfg), state_dir=tmp_path / "f", data_root=ROOT)
    reps = pt.run_replay(prov)
    assert all(r.status == STATUS_OK for r in reps) and len(reps) == 200
    assert_same_as_backtest(bt, pt)
    assert bt.metrics["trades"] == 5 and len(pt.journal.trades) == 5
    assert pt.broker.total_fees() == pytest.approx(bt.metrics["broker_total_fees"])
    pt.close()


@pytest.mark.parametrize("usdtd", [True, False])
def test_consistency_smc_strategy(tmp_path, usdtd):
    cfg = sample_cfg(**{"strategy.filters.ema_trend.enabled": False, "usdtd.enabled": usdtd})
    prov = sample_provider()
    bt = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ROOT).run_provider(prov)
    pt = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=tmp_path / "smc", data_root=ROOT)
    pt.run_replay(prov)
    assert_same_as_backtest(bt, pt)
    assert bt.metrics["trades"] >= 2
    pt.close()


def test_consistency_default_config_zero_trades_same_diagnostics(tmp_path):
    cfg = sample_cfg()
    prov = sample_provider()
    s_bt, s_pt = SMCStrategy.from_config(cfg), SMCStrategy.from_config(cfg)
    bt = BacktestEngine.from_config(cfg, s_bt, data_root=ROOT).run_provider(prov)
    pt = PaperTrader(cfg, s_pt, state_dir=tmp_path / "d", data_root=ROOT)
    pt.run_replay(prov)
    assert_same_as_backtest(bt, pt)
    assert s_bt.diag == s_pt.diag
    pt.close()


def test_candle_log_has_per_bar_journal_fields(tmp_path):
    cfg = sample_cfg(**{"strategy.filters.ema_trend.enabled": False})
    pt = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=tmp_path / "log", data_root=ROOT)
    pt.run_replay(sample_provider())
    pt.close()
    log = pd.read_csv(tmp_path / "log" / "candles.csv")
    assert len(log) == 200
    for col in ("regime", "smc_events", "strategy_state", "signal_side", "risk_decision", "fill_side",
                "position_qty", "equity", "drawdown_pct", "gate_failures", "poi_kind"):
        assert col in log.columns
    assert set(log["regime"].dropna()) <= {"RISING", "FALLING", "NEUTRAL", "UNKNOWN"}
    assert (log["signal_side"] == "buy").sum() >= 2 and (log["fill_side"] == "buy").sum() >= 2
    assert log["poi_kind"].notna().sum() >= 1 and log["smc_events"].notna().sum() >= 1
    assert log["equity"].iloc[-1] == pytest.approx(pt.broker.equity(pt.history[-1].close))
