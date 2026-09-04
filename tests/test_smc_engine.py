"""Tests for the deterministic SMC engine using tiny hand-crafted candles."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CSVMarketData  # noqa: E402
from src.main import load_config  # noqa: E402
from src.strategy import BEARISH, BULLISH, NEUTRAL, SMCConfig, SMCEngine  # noqa: E402

N = 2  # pivot strength used in most tests so hand-made datasets stay small


def bars(rows):
    """rows: list of (open, high, low, close). Timestamps are t0, t1, ..."""
    return pd.DataFrame(
        {
            "timestamp": [f"t{i}" for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        }
    )


def flat(price, lo=None, hi=None):
    """A neutral candle around `price`."""
    return (price, hi if hi is not None else price + 1, lo if lo is not None else price - 1, price)


def run(rows, **kw):
    cfg = SMCConfig(pivot_strength=kw.pop("n", N), **kw)
    eng = SMCEngine(cfg)
    return eng, eng.analyze(bars(rows))


# A base shape: swing high at index 2 (high 20), swing low at index 5 (low 5).
#            0         1         2          3         4         5          6         7
BASE = [flat(10), flat(11), (12, 20, 11, 13), flat(12), flat(11), (10, 11, 5, 9), flat(10), flat(11)]


# ------------------------------------------------------------------ swings
def test_swing_high_detected_and_confirmed_n_bars_later():
    _, r = run(BASE)
    assert len(r.swing_highs) == 1
    s = r.swing_highs[0]
    assert s.index == 2 and s.price == 20 and s.timestamp == "t2"
    assert s.detected_index == 4 and s.detected_timestamp == "t4"


def test_swing_low_detected_and_confirmed_n_bars_later():
    _, r = run(BASE)
    assert len(r.swing_lows) == 1
    s = r.swing_lows[0]
    assert s.index == 5 and s.price == 5
    assert s.detected_index == 7


def test_pivot_not_confirmed_without_enough_future_candles():
    _, r = run(BASE[:6])  # candle 6 and 7 missing -> swing low at 5 unconfirmable
    assert len(r.swing_highs) == 1
    assert len(r.swing_lows) == 0


def test_strict_inequality_equal_highs_are_not_pivots():
    rows = [flat(10), flat(10), (10, 20, 9, 10), (10, 20, 9, 10), flat(10), flat(10), flat(10)]
    _, r = run(rows)
    assert r.swing_highs == []


def test_pivot_strength_from_config_and_validation():
    assert SMCConfig.from_config(load_config()).pivot_strength == 3
    with pytest.raises(ValueError):
        SMCConfig(pivot_strength=0)
    with pytest.raises(ValueError):
        SMCConfig(break_on="wick")


def test_insufficient_data_produces_nothing():
    eng, r = run([flat(10), flat(11)])
    assert r.swing_highs == r.swing_lows == r.bos_events == r.liquidity_sweeps == []
    assert r.fair_value_gaps == [] and r.order_blocks == [] and r.structure == NEUTRAL
    eng2 = SMCEngine(SMCConfig(pivot_strength=3))
    assert eng2.analyze(bars([])).structure == NEUTRAL


# -------------------------------------------------------------- HH/HL/LH/LL
def test_structure_labels():
    rows = (
        [flat(10), flat(10), (10, 20, 9, 10), flat(10), flat(10)]      # SH 20
        + [(10, 11, 4, 10), flat(10), flat(10)]                          # SL 4
        + [(10, 25, 9, 10), flat(10), flat(10)]                          # SH 25 -> HH
        + [(10, 11, 6, 10), flat(10), flat(10)]                          # SL 6  -> HL
        + [(10, 18, 9, 10), flat(10), flat(10)]                          # SH 18 -> LH
        + [(10, 11, 3, 10), flat(10), flat(10)]                          # SL 3  -> LL
    )
    _, r = run(rows)
    assert [s.label for s in r.swing_highs] == [None, "HH", "LH"]
    assert [s.label for s in r.swing_lows] == [None, "HL", "LL"]


# ---------------------------------------------------------------------- BOS
def test_bullish_bos_on_close_above_confirmed_swing_high():
    rows = BASE + [(11, 22, 10, 21)]  # index 8 closes above 20
    _, r = run(rows)
    assert len(r.bos_events) == 1
    e = r.bos_events[0]
    assert e.direction == BULLISH and e.is_choch is False
    assert e.broken_swing_timestamp == "t2" and e.broken_swing_price == 20
    assert e.break_candle_timestamp == "t8" and e.timestamp == "t8" and e.break_candle_close == 21
    assert e.detected_index == 8
    assert e.structure_before == NEUTRAL and e.structure_after == BULLISH
    assert r.structure == BULLISH


def test_bearish_bos_on_close_below_confirmed_swing_low():
    rows = BASE + [(9, 10, 3, 4)]  # index 8 closes below 5
    _, r = run(rows)
    assert len(r.bos_events) == 1
    e = r.bos_events[0]
    assert e.direction == BEARISH and e.broken_swing_price == 5 and e.break_candle_close == 4
    assert r.structure == BEARISH


def test_wick_through_level_without_close_is_not_bos():
    rows = BASE + [(11, 25, 10, 12)]  # wick to 25 but closes 12
    _, r = run(rows)
    assert r.bos_events == []


def test_bos_cannot_use_unconfirmed_swing():
    # a high at index 2 = 20, but the very next candle closes at 21, before confirmation.
    rows = [flat(10), flat(11), (12, 20, 11, 13), (13, 22, 12, 21), flat(21), flat(21)]
    _, r = run(rows)
    assert r.swing_highs == []   # 21/22 after it means index 2 is not a pivot at all
    assert r.bos_events == []


def test_repeated_break_prevention():
    rows = BASE + [(11, 22, 10, 21), (21, 23, 20, 22), (22, 24, 21, 23)]
    _, r = run(rows)
    assert len(r.bos_events) == 1
    assert r.swing_highs[0].broken is True


def test_bos_on_confirmation_candle_itself():
    # swing high 20 at idx 2 confirmed at idx 4; idx 4 itself closes above 20?
    # Impossible by definition (idx 4 high must be < 20). Use swing low instead:
    # low 5 at idx 2 confirmed at idx 4, idx 4 low must be > 5, so close > 5 too. Also impossible.
    # Therefore the earliest possible break is detected_index + 1: assert that holds.
    rows = BASE + [(11, 22, 10, 21)]
    _, r = run(rows)
    assert r.bos_events[0].detected_index > r.swing_highs[0].detected_index


# --------------------------------------------------------------------- CHoCH
def _bull_then_bear():
    # 1) SH 20 at 2 -> bullish BOS at 8 (structure bullish)
    # 2) SL 5 at 5 -> bearish break at 9 => CHoCH
    return BASE + [(11, 22, 10, 21), (9, 10, 3, 4)]


def test_choch_bullish_to_bearish():
    _, r = run(_bull_then_bear())
    assert [e.direction for e in r.bos_events] == [BULLISH, BEARISH]
    assert [e.is_choch for e in r.bos_events] == [False, True]
    c = r.choch_events[0]
    assert c.structure_before == BULLISH and c.structure_after == BEARISH
    assert r.structure == BEARISH


def test_choch_bearish_to_bullish():
    rows = BASE + [(9, 10, 3, 4), (11, 22, 10, 21)]  # bearish first, then bullish
    _, r = run(rows)
    assert [e.direction for e in r.bos_events] == [BEARISH, BULLISH]
    assert [e.is_choch for e in r.bos_events] == [False, True]
    assert r.structure == BULLISH


def test_continuation_bos_is_not_choch():
    rows = (
        [flat(10), flat(10), (10, 20, 9, 10), flat(10), flat(10), (10, 21, 9, 20.5)]  # SH 20, bull BOS @5
        + [flat(20), flat(20), (20, 30, 19, 20), flat(20), flat(20), (20, 31, 19, 30.5)]  # SH 30, bull BOS @11
    )
    _, r = run(rows)
    assert [e.direction for e in r.bos_events] == [BULLISH, BULLISH]
    assert all(not e.is_choch for e in r.bos_events)


def test_first_bos_from_neutral_is_never_choch():
    _, r = run(BASE + [(11, 22, 10, 21)])
    assert r.bos_events[0].is_choch is False and r.bos_events[0].structure_before == NEUTRAL


def test_choch_state_transition_table():
    # exhaustively check the transition rule via the engine's own logic
    eng = SMCEngine(SMCConfig(pivot_strength=1))
    cases = [(NEUTRAL, BULLISH, False), (NEUTRAL, BEARISH, False), (BULLISH, BULLISH, False),
             (BULLISH, BEARISH, True), (BEARISH, BEARISH, False), (BEARISH, BULLISH, True)]
    from src.strategy.smc_types import Swing
    for before, direction, expect in cases:
        eng.result.structure = before
        eng._ts, eng._c, eng._o, eng._h, eng._l = ["x"], [1.0], [1.0], [1.0], [1.0]
        eng.result.bos_events.clear()
        eng._emit_bos(direction, Swing("high", 0, "x", 1.0, 0, "x"), 0)
        assert eng.result.bos_events[0].is_choch is expect, (before, direction)
        assert eng.result.structure == direction


# ------------------------------------------------------------------ sweeps
def test_bullish_liquidity_sweep():
    rows = BASE + [(9, 10, 4, 6)]  # low 4 < 5, closes 6 >= 5
    _, r = run(rows)
    assert len(r.liquidity_sweeps) == 1 and r.bos_events == []
    s = r.liquidity_sweeps[0]
    assert s.direction == BULLISH and s.swept_level == 5 and s.swept_swing_timestamp == "t5"
    assert s.timestamp == "t8" and s.wick_extreme == 4 and s.close == 6 and s.detected_index == 8
    assert r.swing_lows[0].sweeps == 1 and r.swing_lows[0].broken is False


def test_bearish_liquidity_sweep():
    rows = BASE + [(12, 23, 11, 15)]  # high 23 > 20, closes 15 <= 20
    _, r = run(rows)
    assert len(r.liquidity_sweeps) == 1
    s = r.liquidity_sweeps[0]
    assert s.direction == BEARISH and s.swept_level == 20 and s.wick_extreme == 23 and s.close == 15


def test_breakout_close_is_bos_not_sweep():
    _, r = run(BASE + [(11, 22, 10, 21)])
    assert r.liquidity_sweeps == [] and len(r.bos_events) == 1


def test_close_exactly_on_level_is_sweep_not_bos():
    _, r = run(BASE + [(9, 10, 4, 5)])
    assert len(r.liquidity_sweeps) == 1 and r.bos_events == []


def test_sweep_requires_confirmed_swing():
    # low at 5 (idx 5) would be swept by idx 6 - but it is only confirmed at idx 7
    rows = BASE[:6] + [(9, 10, 4, 6), flat(10)]
    _, r = run(rows)
    assert r.liquidity_sweeps == []


def test_swept_level_can_be_swept_again_and_later_broken():
    rows = BASE + [(9, 10, 4, 6), (9, 10, 4.5, 7), (8, 9, 3, 4)]
    _, r = run(rows)
    assert len(r.liquidity_sweeps) == 2 and r.swing_lows[0].sweeps == 2
    assert len(r.bos_events) == 1 and r.bos_events[0].direction == BEARISH


def test_sweeps_disabled_by_config():
    _, r = run(BASE + [(9, 10, 4, 6)], sweep_enabled=False)
    assert r.liquidity_sweeps == []


# --------------------------------------------------------------------- FVG
def test_bullish_fvg():
    rows = [(10, 11, 9, 10), (11, 15, 10, 14), (14, 16, 13, 15)]  # high[0]=11 < low[2]=13
    _, r = run(rows)
    assert len(r.fair_value_gaps) == 1
    g = r.fair_value_gaps[0]
    assert g.direction == BULLISH and g.timestamp == "t1" and g.lower == 11 and g.upper == 13
    assert g.size == 2 and g.detected_index == 2


def test_bearish_fvg():
    rows = [(15, 16, 14, 15), (14, 15, 10, 11), (11, 12, 9, 10)]  # low[0]=14 > high[2]=12
    _, r = run(rows)
    g = r.fair_value_gaps[0]
    assert g.direction == BEARISH and g.timestamp == "t1" and g.upper == 14 and g.lower == 12


def test_touching_candles_are_not_fvg():
    rows = [(10, 11, 9, 10), (11, 15, 10, 14), (14, 16, 11, 15)]  # low[2] == high[0]
    _, r = run(rows)
    assert r.fair_value_gaps == []


def test_fvg_min_gap_filter():
    rows = [(100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101.5, 103)]  # gap 0.5 = ~0.49%
    _, r = run(rows, fvg_min_gap_pct=1.0)
    assert r.fair_value_gaps == []
    _, r = run(rows, fvg_min_gap_pct=0.1)
    assert len(r.fair_value_gaps) == 1


def test_fvg_disabled_by_config():
    _, r = run([(10, 11, 9, 10), (11, 15, 10, 14), (14, 16, 13, 15)], fvg_enabled=False)
    assert r.fair_value_gaps == []


# ------------------------------------------------------------- order blocks
def test_bullish_order_block_is_last_bearish_candle_before_break():
    # index 6 bearish (11 -> 10), index 7 bullish, index 8 breaks.
    rows = BASE[:6] + [(11, 12, 9, 10), (10, 12, 9.5, 11.5), (11.5, 22, 11, 21)]
    _, r = run(rows)
    assert len(r.order_blocks) == 1
    ob = r.order_blocks[0]
    assert ob.direction == BULLISH and ob.timestamp == "t6"
    assert (ob.open, ob.high, ob.low, ob.close) == (11, 12, 9, 10)
    assert ob.bos is r.bos_events[0] and ob.detected_index == 8


def test_bearish_order_block_is_last_bullish_candle_before_break():
    # index 6 bullish (9 -> 10.5), index 7 bearish, index 8 breaks down.
    rows = BASE[:6] + [(9, 11, 8.5, 10.5), (10.5, 11, 9, 9.5), (9.5, 10, 3, 4)]
    _, r = run(rows)
    ob = r.order_blocks[0]
    assert ob.direction == BEARISH and ob.timestamp == "t6"
    assert (ob.open, ob.high, ob.low, ob.close) == (9, 11, 8.5, 10.5)
    assert ob.bos.direction == BEARISH


def test_order_block_skips_dojis_and_same_direction_candles():
    rows = BASE[:6] + [(11, 12, 9, 10), (10, 11, 9, 10), (10, 12, 9.5, 11.5), (11.5, 22, 11, 21)]
    _, r = run(rows)
    assert r.order_blocks[0].timestamp == "t6"


def test_no_order_block_if_no_opposite_candle_exists():
    rows = [(10, 11, 9, 10.5), (10.5, 12, 10, 11), (11, 20, 10.5, 12), (12, 13, 11, 12.5),
            (12.5, 13.5, 12, 13), (13, 22, 12.5, 21)]  # all bullish/doji, SH 20 @2 broken @5
    _, r = run(rows)
    assert len(r.bos_events) == 1 and r.order_blocks == []


def test_order_blocks_disabled_by_config():
    _, r = run(BASE + [(11, 22, 10, 21)], order_blocks_enabled=False)
    assert len(r.bos_events) == 1 and r.order_blocks == []


# -------------------------------------------------------------- no look-ahead
def _sample_df():
    cfg = load_config()
    return CSVMarketData(ROOT / cfg["data"]["directory"]).get_ohlcv(cfg["market"]["symbol"], cfg["market"]["timeframe"])


def test_incremental_equals_batch_and_prefix_invariance():
    df = _sample_df()
    full = SMCEngine(SMCConfig(pivot_strength=3)).analyze(df)
    for cut in (30, 77, 150):
        partial = SMCEngine(SMCConfig(pivot_strength=3)).analyze(df.iloc[:cut])
        known = SMCEngine(SMCConfig(pivot_strength=3))
        known.analyze(df)
        snap = known.events_known_at(cut - 1)
        for attr in ("swing_highs", "swing_lows", "bos_events", "liquidity_sweeps", "fair_value_gaps", "order_blocks"):
            p, s = getattr(partial, attr), getattr(snap, attr)
            assert len(p) == len(s), (attr, cut)
            assert [_key(x) for x in p] == [_key(x) for x in s], (attr, cut)
        assert partial.structure == snap.structure
    assert len(full.bos_events) > 0


def _key(ev):
    d = getattr(ev, "__dict__", None) or {}
    return tuple((k, v if not hasattr(v, "__dict__") else _key(v)) for k, v in d.items())


def test_future_candles_never_alter_emitted_events():
    df = _sample_df()
    eng = SMCEngine(SMCConfig(pivot_strength=3))
    snapshots = {}
    for i, row in enumerate(df.itertuples(index=False)):
        r = eng.update(row.timestamp, row.open, row.high, row.low, row.close)
        snapshots[i] = (
            [(s.index, s.price, s.detected_index, s.label) for s in r.swing_highs],
            [(s.index, s.price, s.detected_index, s.label) for s in r.swing_lows],
            list(r.bos_events), list(r.liquidity_sweeps), list(r.fair_value_gaps), list(r.order_blocks),
        )
        # every event ever emitted must have detected_index <= current index
        for coll in (r.swing_highs, r.swing_lows, r.bos_events, r.liquidity_sweeps, r.fair_value_gaps, r.order_blocks):
            assert all(e.detected_index <= i for e in coll)
    # each snapshot must be a prefix of the final state (nothing rewritten or removed)
    final = snapshots[len(df) - 1]
    for i, snap in snapshots.items():
        for a, b in zip(snap, final):
            assert b[: len(a)] == a, f"event history rewritten after candle {i}"


def test_pivot_known_only_at_i_plus_n():
    df = _sample_df()
    for n in (1, 3, 5):
        r = SMCEngine(SMCConfig(pivot_strength=n)).analyze(df)
        assert r.swing_highs and r.swing_lows
        assert all(s.detected_index == s.index + n for s in r.swing_highs + r.swing_lows)
        assert all(e.detected_index >= 0 for e in r.bos_events)
        for e in r.bos_events:
            # the broken swing must have been confirmed at or before the break candle
            swing = next(s for s in r.swing_highs + r.swing_lows if s.timestamp == e.broken_swing_timestamp
                         and s.price == e.broken_swing_price)
            assert swing.detected_index <= e.detected_index


def test_engine_from_config_runs_on_sample():
    r = SMCEngine.from_config(load_config()).analyze(_sample_df())
    assert r.structure in (BULLISH, BEARISH, NEUTRAL)
    assert len(r.order_blocks) <= len(r.bos_events)
