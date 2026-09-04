"""SMCStrategy: gates, geometry, exits, regime tightening, next-open gap safety, anti-lookahead."""

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from smc_scenarios import IND, IND_COLS, contexts, flat, frame, pullback_scenario  # noqa: E402
from src.backtesting import BUY, EXIT, BacktestConfig, BacktestEngine  # noqa: E402
from src.data import CSVMarketData  # noqa: E402
from src.main import load_config  # noqa: E402
from src.risk import RiskConfig, RiskDecision, TradeValidator  # noqa: E402
from src.strategy.poi import FVG, ORDER_BLOCK, POI  # noqa: E402
from src.strategy.regime import Regime, RegimeState  # noqa: E402
from src.strategy.smc_engine import SMCConfig, SMCEngine  # noqa: E402
from src.strategy.smc_strategy import (  # noqa: E402
    ARMED,
    COOLDOWN,
    IDLE,
    IN_TRADE,
    REJ_D1,
    REJ_D2,
    REJ_D3,
    TP_FIXED,
    TP_HYBRID,
    TP_STRUCTURE,
    SMCStrategy,
    SMCStrategyConfig,
)

BASE = dict(ema_trend_enabled=False, ema_extension_enabled=False, volume_enabled=False, **IND_COLS)


def strat(**kw):
    return SMCStrategy(SMCStrategyConfig(**{**BASE, **kw}))


def regime(r: Regime):
    return lambda i: RegimeState(r, r, 4.0, 4.0, 4.0, 0.0, 0.0, 3, "t", 0)


def signals(rows, s, **kw):
    return [(i, sig) for i, sig, _ in contexts(rows, s, **kw) if sig is not None]


# ------------------------------------------------------------------ happy path
def test_happy_path_buy_on_ob_pullback():
    s = strat()
    sigs = signals(pullback_scenario(), s)
    assert len(sigs) == 1
    i, sig = sigs[0]
    assert i == 11 and sig.side == BUY
    atr = contexts(pullback_scenario())[11][2].indicators["atr_3"]
    assert sig.stop_loss == pytest.approx(98.0 - 0.25 * atr)
    assert sig.take_profit == pytest.approx(104.0 + 2.0 * (104.0 - sig.stop_loss))  # fixed-RR fallback
    assert "order_block" in sig.reason and REJ_D3 in sig.reason
    assert s.state == IN_TRADE and s.diag.buy_signals == 1


def test_long_only_stop_always_below_close():
    for rc in (102.5, 103.0, 104.0, 108.0):
        for i, sig in signals(pullback_scenario(rejection_close=rc), strat()):
            assert sig.side == BUY and sig.stop_loss < rc and sig.take_profit > rc


# -------------------------------------------------------------- structure gates
def test_no_bullish_bos_no_signal():
    rows = flat(100, 15)
    s = strat()
    assert signals(rows, s) == [] and s.state == IDLE and s.diag.setups_armed == 0


def test_setup_age_expires():
    rows = pullback_scenario()[:11] + flat(114, 10) + [pullback_scenario()[11]]
    s = strat(setup_max_age_bars=5)
    assert signals(rows, s) == []
    assert s.diag.gate_failures.get("A_setup_age") == 1 and s.state == IDLE


def test_break_candle_itself_is_not_an_entry():
    s = strat()
    res = contexts(pullback_scenario()[:8], s)
    assert all(sig is None for _, sig, _ in res)


def test_poi_mitigated_by_close_below_zone_kills_setup():
    rows = pullback_scenario()[:11] + [(112, 112.5, 96, 97)] + [pullback_scenario()[11]]
    s = strat()
    assert signals(rows, s) == [] and s.state == IDLE


# ------------------------------------------------------------- rejection gate D
def test_no_rejection_no_signal():
    # bar 11: bearish candle closing in lower half, inside the zone
    rows = pullback_scenario()[:11] + [(112, 112.5, 99, 99.5)]
    s = strat()
    assert signals(rows, s) == [] and s.diag.gate_failures.get("D_rejection") == 1


def test_d1_bullish_close_upper_half_inside_zone():
    # open 99.2 low 98.2 high 101.8 close 101.5 -> bullish, upper half, still inside OB [98,102]
    rows = pullback_scenario()[:11] + [(99.2, 101.8, 98.2, 101.5)]
    sigs = signals(rows, strat())
    assert len(sigs) == 1 and REJ_D1 in sigs[0][1].reason


def test_require_rejection_false_allows_bearish_touch():
    rows = pullback_scenario()[:11] + [(112, 112.5, 99, 101.9)]   # bearish candle, lower half, inside zone
    assert signals(rows, strat()) == []
    assert len(signals(rows, strat(require_rejection=False))) == 1


def test_d2_sweep_tightens_stop_below_wick():
    # a confirmed swing low at 5 = 90. Build a candle that sweeps 90 and closes back inside a deeper POI.
    # Simplest: make the OB itself deep by shifting scenario so OB low ~ 90.5, then sweep to 89.
    rows = (flat(100, 2) + [(100, 110, 99, 101)] + flat(100, 2) + [(100, 101, 90, 100)]
            + [(101, 102, 90.5, 91)]            # 6 bearish OB [90.5, 102]
            + [(91, 113, 90.6, 112, 150.0)]      # 7 BOS
            + [(112, 116, 108, 114)] + [(114, 115, 113, 114)] * 2
            + [(112, 112.5, 89.0, 100.0)])      # 11 sweep 90 (wick 89) and close 100 inside OB
    s = strat(max_stop_atr=10.0, poi_max_atr_multiple=10.0)
    sigs = signals(rows, s)
    assert len(sigs) == 1 and REJ_D2 in sigs[0][1].reason
    atr = contexts(rows)[11][2].indicators["atr_3"]
    assert sigs[0][1].stop_loss == pytest.approx(89.0 - 0.25 * atr)


# ------------------------------------------------------------------ filters E
def test_ema_trend_filter_blocks_when_below_ema200():
    s = strat(ema_trend_enabled=True)
    ctx11 = contexts(pullback_scenario())[11][2]
    # in this synthetic scenario close 104 < ema_8 (107.2) -> blocked
    assert ctx11.candle.close < ctx11.indicators["ema_8"]
    assert signals(pullback_scenario(), s) == [] and s.diag.gate_failures.get("E1_ema_trend") == 1


def test_ema_extension_filter_blocks_chasing():
    rows = pullback_scenario(rejection_close=112.4, pull_low=101.9)  # touch zone then close ~8 ATR above zone
    s = strat(ema_extension_enabled=True, ema_extension_atr=-1.0)     # require close <= ema - 1 ATR (never true here)
    assert signals(rows, s) == [] and s.diag.gate_failures.get("E2_extension") == 1
    assert len(signals(rows, strat())) == 1  # same rows pass with the filter off


def test_volume_filter_dead_volume_blocked_unless_bos_was_strong():
    rows = pullback_scenario(vol=10.0)  # rejection candle ratio ~0.1
    s = strat(volume_enabled=True, vol_ratio_min=0.8, bos_vol_ratio_min=5.0)  # BOS ratio 1.5 < 5 -> not enough
    assert signals(rows, s) == [] and s.diag.gate_failures.get("E3_volume") == 1
    s2 = strat(volume_enabled=True, vol_ratio_min=0.8, bos_vol_ratio_min=1.2)  # strong BOS rescues
    assert len(signals(rows, s2)) == 1


# --------------------------------------------------------------- geometry F
def test_stop_too_wide_rejected():
    s = strat(max_stop_atr=0.6)
    assert signals(pullback_scenario(), s) == [] and s.diag.gate_failures.get("F_stop_too_wide") == 1


def test_stop_too_tight_rejected():
    rows = pullback_scenario(rejection_close=99.0)  # D3 not met, so use require_rejection False; distance 1 < 1 ATR
    s = strat(min_stop_atr=1.0, stop_buffer_atr=0.0, require_rejection=False)
    assert signals(rows, s) == [] and s.diag.gate_failures.get("F_stop_too_tight") == 1


def test_target_modes():
    rows = pullback_scenario()
    fixed = signals(rows, strat(tp_mode=TP_FIXED, fixed_rr=3.0))[0][1]
    struct = signals(rows, strat(tp_mode=TP_STRUCTURE, min_risk_reward=1.0))[0][1]
    hybrid = signals(rows, strat(tp_mode=TP_HYBRID, min_risk_reward=1.0, fixed_rr=1.0))[0][1]
    dist = 104.0 - fixed.stop_loss
    assert fixed.take_profit == pytest.approx(104.0 + 3.0 * dist)
    assert struct.take_profit == 116.0           # nearest unbroken swing high with RR >= 1
    assert hybrid.take_profit == pytest.approx(max(116.0, 104.0 + 1.0 * dist))


def test_min_rr_rejects_when_no_target_reaches_it():
    s = strat(tp_mode=TP_STRUCTURE, fixed_rr=1.0, min_risk_reward=2.0)  # fallback fixed 1R < 2R
    assert signals(pullback_scenario(), s) == [] and s.diag.gate_failures.get("F_rr") == 1


# ---------------------------------------------------------------------- exits
def test_bearish_choch_triggers_exit_and_cooldown():
    rows = pullback_scenario() + [(104, 105, 103, 104)] * 3 + [(104, 104.5, 85, 86)]  # close < swing low 90
    s = strat(reentry_cooldown_bars=3)
    pos = {12: True}
    sigs = signals(rows, s, positions=pos)
    assert [x[1].side for x in sigs] == [BUY, EXIT] and sigs[1][1].reason == "bearish_choch"
    assert sigs[1][0] == 15


def test_time_stop_exit_when_not_in_profit():
    rows = pullback_scenario() + [(104, 105, 103, 104)] * 6
    s = strat(max_bars_in_trade=4)
    sigs = signals(rows, s, positions={12: True})
    assert [x[1].side for x in sigs] == [BUY, EXIT] and sigs[1][1].reason.startswith("time_stop")
    assert sigs[1][0] == 16  # entry filled at 12 -> 4 bars later


def test_time_stop_not_triggered_when_in_profit():
    rows = pullback_scenario() + [(104, 125, 103, 124)] * 6   # >= 1R in profit
    s = strat(max_bars_in_trade=4)
    sigs = signals(rows, s, positions={12: True})
    assert [x[1].side for x in sigs] == [BUY]


def test_cooldown_and_max_entries_per_setup():
    rows = pullback_scenario() + [pullback_scenario()[11]] * 4
    s = strat(reentry_cooldown_bars=3, max_entries_per_setup=1)
    sigs = signals(rows, s, positions={12: True, 13: False})
    assert len(sigs) == 1  # one entry per setup, even after the position closes
    s2 = strat(reentry_cooldown_bars=0, max_entries_per_setup=3)
    sigs2 = signals(rows, s2, positions={12: True, 13: False})
    assert len(sigs2) == 3   # bars 11, 13, 14: entries capped at 3 per setup (bar 15 touch -> none)
    s3 = strat(reentry_cooldown_bars=2, max_entries_per_setup=3)
    sigs3 = signals(rows, s3, positions={12: True, 13: False})
    assert [i for i, _ in sigs3] == [11, 15]   # exit seen at 13 -> cooldown until 15


def test_rejected_or_unfilled_buy_returns_to_armed():
    rows = pullback_scenario() + [(104, 105, 103, 104)]
    s = strat()
    res = contexts(rows, s, positions={})  # has_position never becomes True
    assert res[11][1].side == BUY
    assert s.state in (ARMED, IDLE) and s.trade is None


# --------------------------------------------------------------- regime gate G
def test_usdtd_disabled_ignores_regime_entirely():
    rows = pullback_scenario()
    a = signals(rows, strat(usdtd_enabled=False), regime_fn=regime(Regime.RISING))
    b = signals(rows, strat(usdtd_enabled=False))
    assert a == b and len(a) == 1


def strip(sigs):
    return [(i, sg.side, sg.stop_loss, sg.take_profit) for i, sg in sigs]


def test_unknown_and_neutral_and_falling_behave_like_no_regime():
    rows = pullback_scenario()
    base = strip(signals(rows, strat(usdtd_enabled=True)))
    for r in (Regime.UNKNOWN, Regime.NEUTRAL, Regime.FALLING):
        assert strip(signals(rows, strat(usdtd_enabled=True), regime_fn=regime(r))) == base
    assert len(base) == 1


def test_rising_skips_first_poi_and_requires_deeper():
    rows = pullback_scenario()
    s = strat(usdtd_enabled=True)
    assert signals(rows, s, regime_fn=regime(Regime.RISING)) == []
    assert s.diag.riskoff_skips == 1 and s.diag.gate_failures.get("G_no_deeper_poi") == 1
    assert s.setup.pois[0].skipped is True


def test_rising_trades_deeper_poi_of_any_kind():
    """RISING skips the first (highest) POI and trades a deeper valid POI of ANY kind."""
    rows = pullback_scenario()[:11] + [(112, 112.5, 100.0, 103.0)]     # bar 11 touches OB -> skipped
    s = strat(usdtd_enabled=True, riskoff_min_confluence=0, riskoff_rr_add=0.0, max_stop_atr=10, poi_max_atr_multiple=10)
    res = contexts(rows, s, regime_fn=regime(Regime.RISING))
    assert res[11][1] is None and s.setup is not None and s.setup.pois[0].skipped
    # inject deeper zones of both kinds into the live setup (as if formed in the impulse) - FVG lowest
    s.setup.pois.append(POI(ORDER_BLOCK, 95.0, 97.0, 7, None))
    s.setup.pois.append(POI(FVG, 90.0, 94.0, 7, None))
    # bar 12: sweeps into the deeper FVG (low 91) and closes back above it (D3) - build ctx by hand
    from src.data.base import Candle
    from src.backtesting.strategy import BacktestContext
    prev = res[11][2]
    c12 = Candle(prev.candle.timestamp + pd.Timedelta(minutes=15), 96.0, 96.5, 91.0, 94.5, 100.0)
    ctx12 = BacktestContext(12, c12, prev.indicators, prev.smc, False, 10_000.0, {}, regime=regime(Regime.RISING)(12))
    sig = s.on_candle(ctx12)
    assert sig is not None and sig.side == BUY and "fvg" in sig.reason
    assert sig.stop_loss < 90.0                     # stop under the deeper zone
    # and an OB-only deeper zone is equally acceptable (search is kind-agnostic)
    s2 = strat(usdtd_enabled=True, riskoff_min_confluence=0, riskoff_rr_add=0.0, max_stop_atr=10, poi_max_atr_multiple=10)
    res2 = contexts(rows, s2, regime_fn=regime(Regime.RISING))
    s2.setup.pois.append(POI(ORDER_BLOCK, 90.0, 94.0, 7, None))
    prev2 = res2[11][2]   # must use s2's own SMC result object (setups are keyed by BOS identity)
    ctx12b = BacktestContext(12, c12, prev2.indicators, prev2.smc, False, 10_000.0, {}, regime=regime(Regime.RISING)(12))
    sig2 = s2.on_candle(ctx12b)
    assert sig2 is not None and "order_block" in sig2.reason


def test_rising_requires_d2_or_d3_rejection():
    rows = pullback_scenario()[:11] + [(99.2, 101.8, 98.2, 101.5)]  # D1 only
    s = strat(usdtd_enabled=True, riskoff_require_deeper_poi=False, riskoff_min_confluence=0)
    assert signals(rows, s, regime_fn=regime(Regime.RISING)) == []
    assert s.diag.gate_failures.get("D_riskoff_rejection") == 1
    assert len(signals(rows, strat(usdtd_enabled=True), regime_fn=regime(Regime.NEUTRAL))) == 1


def test_rising_raises_min_rr_and_confluence():
    rows = pullback_scenario()
    ok = strat(usdtd_enabled=True, riskoff_require_deeper_poi=False, riskoff_min_confluence=0, riskoff_rr_add=0.0)
    assert len(signals(rows, ok, regime_fn=regime(Regime.RISING))) == 1
    rr = strat(usdtd_enabled=True, riskoff_require_deeper_poi=False, riskoff_min_confluence=0, riskoff_rr_add=5.0)
    assert signals(rows, rr, regime_fn=regime(Regime.RISING)) == [] and rr.diag.gate_failures.get("F_rr") == 1
    conf = strat(usdtd_enabled=True, riskoff_require_deeper_poi=False, riskoff_min_confluence=4, riskoff_rr_add=0.0)
    assert signals(rows, conf, regime_fn=regime(Regime.RISING)) == [] and conf.diag.gate_failures.get("G_confluence") == 1


def test_rising_raises_volume_threshold():
    rows = pullback_scenario(vol=95.0)  # ratio ~0.95: passes 0.8, fails 1.0
    kw = dict(usdtd_enabled=True, riskoff_require_deeper_poi=False, riskoff_min_confluence=0, riskoff_rr_add=0.0,
              volume_enabled=True, vol_ratio_min=0.8, riskoff_vol_ratio_min=1.0, bos_vol_ratio_min=9.0)
    assert len(signals(rows, strat(**kw), regime_fn=regime(Regime.NEUTRAL))) == 1
    s = strat(**kw)
    assert signals(rows, s, regime_fn=regime(Regime.RISING)) == [] and s.diag.gate_failures.get("E3_volume") == 1


def test_resume_normal_after_regime_weakens():
    rows = pullback_scenario()[:11] + [(112, 112.5, 100.0, 103.0)] + [(103, 103.5, 100.5, 103.5)]
    # bar 11 RISING (skip OB, no deeper), bar 12 NEUTRAL -> OB still valid but flagged skipped;
    # design: skipped POIs are not retroactively traded, only other valid POIs. So no trade here...
    s = strat(usdtd_enabled=True)
    fn = lambda i: regime(Regime.RISING)(i) if i <= 11 else regime(Regime.NEUTRAL)(i)  # noqa: E731
    assert signals(rows, s, regime_fn=fn) == []
    # ...but once the regime is NEUTRAL again, the same strategy on a fresh pullback scenario trades normally
    # (skipped flags live on the setup; a new setup starts clean).
    s2 = strat(usdtd_enabled=True)
    fn2 = lambda i: regime(Regime.RISING)(i) if i < 5 else regime(Regime.NEUTRAL)(i)  # noqa: E731
    sigs = signals(pullback_scenario(), s2, regime_fn=fn2)
    assert len(sigs) == 1 and sigs[0][0] == 11 and s2.diag.riskoff_skips == 0


# ------------------------------------------------- next-open gap safety (engine)
def engine_for(strategy, **kw):
    cfg = BacktestConfig("BTC/USDT", "15m", 10_000.0, 0.1, 0.0, 1.0, **kw)
    v = TradeValidator(RiskConfig(1.0, 3.0, 10.0, 5, 1))
    return BacktestEngine(cfg, strategy, IND, lambda: SMCEngine(SMCConfig(pivot_strength=2)), v)


def test_next_open_above_stop_normal_entry():
    rows = pullback_scenario() + [(104.5, 106, 103.5, 105)] + [(105, 106, 104, 105)] * 3
    res = engine_for(strat()).run(frame(rows))
    assert len(res.journal.trades) == 1 and res.journal.rejections == []
    t = res.journal.trades[0]
    assert t.entry_index == 12 and t.entry_price == 104.5 and t.stop_loss < t.entry_price


def test_next_open_at_or_below_stop_is_rejected_by_risk_engine():
    sig = signals(pullback_scenario(), strat())[0][1]
    for gap_open in (sig.stop_loss, sig.stop_loss - 1.0):
        rows = pullback_scenario() + [(gap_open, gap_open + 0.5, gap_open - 0.5, gap_open)] + flat(gap_open, 3)
        res = engine_for(strat()).run(frame(rows))
        assert res.journal.trades == []
        assert [r.decision for r in res.journal.rejections] == [RiskDecision.REJECTED_INVALID_STOP.value]
        assert (res.equity_curve["position"] == 0).all()


def test_gap_rejection_leaves_strategy_consistent_and_no_second_validation_path():
    sig = signals(pullback_scenario(), strat())[0][1]
    g = sig.stop_loss - 1.0
    rows = pullback_scenario() + [(g, g + 0.5, g - 0.5, g)] + flat(g, 3)
    s = strat()
    res = engine_for(s).run(frame(rows))
    assert s.trade is None and s.state in (ARMED, IDLE)
    # the rejection came from TradeValidator (its reason string), not from the strategy/engine
    assert "must be below entry" in res.journal.rejections[0].reason


# ---------------------------------------------- backtester integration + regime
def sample_cfg(**over):
    cfg = copy.deepcopy(load_config())
    for k, v in over.items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return cfg


def sample_df():
    cfg = load_config()
    return CSVMarketData(ROOT / cfg["data"]["directory"]).get_ohlcv(cfg["market"]["symbol"], cfg["market"]["timeframe"])


def run_sample(cfg):
    s = SMCStrategy.from_config(cfg)
    return BacktestEngine.from_config(cfg, s, data_root=ROOT).run(sample_df()), s


def test_from_config_and_sample_runs_with_and_without_usdtd():
    on, s_on = run_sample(sample_cfg())
    off, s_off = run_sample(sample_cfg(**{"usdtd.enabled": False}))
    assert on.metrics["bars"] == off.metrics["bars"] == 200
    assert s_on.cfg.usdtd_enabled and not s_off.cfg.usdtd_enabled
    assert s_off.diag.riskoff_skips == 0


def test_disabled_usdtd_identical_to_strategy_without_regime():
    """usdtd.enabled=false must behave exactly like running with no aux data at all."""
    cfg_off = sample_cfg(**{"usdtd.enabled": False, "strategy.filters.ema_trend.enabled": False})
    a, sa = run_sample(cfg_off)
    s = SMCStrategy.from_config(cfg_off)
    eng = BacktestEngine.from_config(cfg_off, s, data_root=ROOT)
    assert eng.regime_config is None and eng.aux_df is None
    b = eng.run(sample_df())
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)
    assert a.journal.to_frame().equals(b.journal.to_frame())
    assert a.metrics["trades"] >= 1  # the disabled path actually trades on the sample


def test_regime_context_is_point_in_time_on_sample():
    cfg = sample_cfg()
    df = sample_df()
    aux = pd.read_csv(ROOT / "data/sample/USDTD_4h.csv")
    aux_ts = pd.to_datetime(aux["timestamp"], utc=True)

    class Spy(SMCStrategy):
        seen = 0

        def on_candle(self, ctx):
            Spy.seen += 1
            if ctx.regime is not None and ctx.regime.source_index >= 0:
                src_close = aux_ts[ctx.regime.source_index] + pd.Timedelta(hours=4)
                assert src_close <= ctx.candle.timestamp + pd.Timedelta(minutes=15)
                if ctx.regime.source_index + 1 < len(aux_ts):
                    assert aux_ts[ctx.regime.source_index + 1] + pd.Timedelta(hours=4) > ctx.candle.timestamp + pd.Timedelta(minutes=15)
            return super().on_candle(ctx)

    BacktestEngine.from_config(cfg, Spy.from_config(cfg), data_root=ROOT).run(df)
    assert Spy.seen == 200


def test_prefix_equality_and_future_usdtd_shock_with_regime():
    cfg = sample_cfg(**{"strategy.filters.ema_trend.enabled": False, "backtesting.close_open_position_at_end": False})
    df = sample_df()
    aux = pd.read_csv(ROOT / "data/sample/USDTD_4h.csv")
    full = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), aux_df=aux).run(df)
    for cut in (90, 150):
        part = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), aux_df=aux).run(df.iloc[:cut])
        pd.testing.assert_frame_equal(part.equity_curve, full.equity_curve.iloc[:cut].reset_index(drop=True))
    # shock USDT.D after the BTC window midpoint: nothing before the first affected BTC bar may change
    mid_ts = pd.to_datetime(df["timestamp"].iloc[120], utc=True)
    shocked = aux.copy()
    mask = pd.to_datetime(shocked["timestamp"], utc=True) >= mid_ts
    shocked.loc[mask, "close"] *= 1.5
    other = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), aux_df=shocked).run(df)
    pd.testing.assert_frame_equal(full.equity_curve.iloc[:120], other.equity_curve.iloc[:120])


def test_deterministic_repeated_runs():
    cfg = sample_cfg(**{"strategy.filters.ema_trend.enabled": False})
    a, _ = run_sample(cfg)
    b, _ = run_sample(cfg)
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)
    assert a.summary_dict() == b.summary_dict()


def test_regime_never_touches_risk_engine():
    """Same proposal -> same validator decision regardless of regime; strategy holds no RiskState."""
    s = strat(usdtd_enabled=True)
    assert not any(k for k in vars(s) if "risk" in k.lower())
    v = TradeValidator(RiskConfig(1.0, 3.0, 10.0, 5, 1))
    from src.risk import TradeProposal
    p = TradeProposal(10_000, 100.0, 98.0)
    assert v.validate(p).decision is RiskDecision.APPROVED  # regime has no channel into this call


def test_strategy_config_from_yaml():
    c = SMCStrategyConfig.from_config(load_config())
    assert c.atr_col == "atr_14" and c.ema_fast_col == "ema_20" and c.ema_mid_col == "ema_50" and c.ema_slow_col == "ema_200"
    assert c.vol_ratio_col == "volume_ratio_20" and c.tp_mode == TP_STRUCTURE and c.min_risk_reward == 2.0
    assert c.usdtd_enabled is True and c.riskoff_rr_add == 0.5 and c.poi_priority == (ORDER_BLOCK, FVG)
    for kw in (dict(tp_mode="x"), dict(min_stop_atr=0), dict(max_stop_atr=0.1, min_stop_atr=0.5), dict(fixed_rr=0)):
        with pytest.raises(ValueError):
            SMCStrategyConfig(**kw)
