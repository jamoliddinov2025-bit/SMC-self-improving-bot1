"""Unit tests for the indicator engine (EMA, ATR, volume)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CSVMarketData  # noqa: E402
from src.indicators import IndicatorConfig, IndicatorEngine, atr, ema, true_range, volume_ratio, volume_sma  # noqa: E402
from src.main import load_config  # noqa: E402


def make_df(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    closes = [float(c) for c in closes]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": closes,
        "high": highs if highs is not None else [c + 1 for c in closes],
        "low": lows if lows is not None else [c - 1 for c in closes],
        "close": closes,
        "volume": volumes if volumes is not None else [100.0] * n,
    })


# --------------------------------------------------------------------- EMA
def test_ema_hand_calculated():
    # period 3 -> alpha 0.5. Seed = mean(1,2,3) = 2.
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)   # 3.0
    assert out.iloc[4] == pytest.approx(0.5 * 5 + 0.5 * 3.0)   # 4.0


def test_ema_constant_series_is_constant():
    out = ema(pd.Series([7.0] * 30), 10)
    assert np.allclose(out.iloc[9:], 7.0)


def test_ema_period_1_equals_price():
    s = pd.Series([3.0, 5.0, 2.0])
    assert ema(s, 1).tolist() == pytest.approx(s.tolist())


def test_ema_warmup_and_insufficient_data():
    s = pd.Series(np.arange(10, dtype=float))
    out = ema(s, 20)
    assert out.isna().all()
    assert len(out) == 10
    out = ema(s, 5)
    assert out.iloc[:4].isna().all() and out.iloc[4:].notna().all()


def test_ema_preserves_index_and_name():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    out = ema(pd.Series([1, 2, 3, 4, 5], index=idx), 2)
    assert out.index.equals(idx)
    assert out.name == "ema_2"


def test_ema_invalid_period():
    with pytest.raises(ValueError):
        ema(pd.Series([1.0]), 0)


# --------------------------------------------------------------------- ATR
def test_true_range_hand_calculated():
    df = make_df(closes=[10, 12, 9], highs=[11, 15, 10], lows=[9, 11, 7])
    tr = true_range(df)
    assert tr.iloc[0] == pytest.approx(2.0)                # no prev close -> high - low
    assert tr.iloc[1] == pytest.approx(max(4, 5, 1))       # 15-11, |15-10|, |11-10|
    assert tr.iloc[2] == pytest.approx(max(3, 2, 5))       # 10-7, |10-12|, |7-12|


def test_atr_hand_calculated():
    # constant TR of 2 -> ATR should be 2 everywhere after warm-up
    df = make_df(closes=[10] * 20)  # high = 11, low = 9 -> TR 2
    out = atr(df, 14)
    assert out.iloc[:13].isna().all()
    assert np.allclose(out.iloc[13:], 2.0)


def test_atr_wilder_smoothing_step():
    # period 2: seed = mean(TR0, TR1); ATR2 = (ATR1*(2-1) + TR2)/2
    df = make_df(closes=[10, 10, 10], highs=[12, 14, 20], lows=[10, 10, 10])
    tr = true_range(df)  # [2, 4, 10]
    assert tr.tolist() == pytest.approx([2, 4, 10])
    out = atr(df, 2)
    assert out.iloc[1] == pytest.approx(3.0)
    assert out.iloc[2] == pytest.approx((3.0 * 1 + 10) / 2)


def test_atr_insufficient_data():
    out = atr(make_df(closes=[1, 2, 3]), 14)
    assert out.isna().all() and len(out) == 3


def test_atr_is_never_negative():
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 300).cumsum()
    df = make_df(c, highs=c + rng.uniform(0, 2, 300), lows=c - rng.uniform(0, 2, 300))
    assert (atr(df, 14).dropna() >= 0).all()


# ------------------------------------------------------------------ Volume
def test_volume_sma_hand_calculated():
    v = pd.Series([10.0, 20.0, 30.0, 40.0])
    out = volume_sma(v, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == pytest.approx([15.0, 25.0, 35.0])


def test_volume_ratio_hand_calculated():
    v = pd.Series([10.0, 20.0, 30.0, 60.0])
    out = volume_ratio(v, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(30 / 20)
    assert out.iloc[3] == pytest.approx(60 / ((20 + 30 + 60) / 3))


def test_volume_ratio_zero_average_gives_nan():
    out = volume_ratio(pd.Series([0.0, 0.0, 0.0, 5.0]), 2)
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(5 / 2.5)


def test_volume_warmup():
    out = volume_sma(pd.Series([1.0] * 5), 10)
    assert out.isna().all()


# ------------------------------------------------------------------ Engine
def test_engine_config_from_yaml():
    cfg = load_config()
    ic = IndicatorConfig.from_config(cfg)
    assert ic.ema_periods == [20, 50, 200]
    assert ic.atr_period == cfg["indicators"]["atr"]["period"]
    assert ic.volume_ma_period == cfg["indicators"]["volume"]["ma_period"]
    assert ic.warmup_bars == 199


def test_engine_adds_aliases_missing_from_periods():
    cfg = {"indicators": {"ema": {"periods": [20], "fast": 50, "slow": 200},
                          "atr": {"period": 14}, "volume": {"ma_period": 20}}}
    assert IndicatorConfig.from_config(cfg).ema_periods == [20, 50, 200]


def test_engine_columns_aligned_with_input():
    df = make_df(np.arange(1, 31, dtype=float))
    eng = IndicatorEngine(IndicatorConfig(ema_periods=[3, 5], atr_period=4, volume_ma_period=3))
    out = eng.compute(df)
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert out["timestamp"].equals(df["timestamp"])
    for col in eng.columns:
        assert col in out.columns
    # warm-up rows
    assert out["ema_5"].iloc[:4].isna().all() and out["ema_5"].iloc[4:].notna().all()
    assert out["atr_4"].iloc[:3].isna().all() and out["atr_4"].iloc[3:].notna().all()
    assert out["volume_ratio_3"].iloc[:2].isna().all()


def test_engine_does_not_mutate_input():
    df = make_df([1.0, 2.0, 3.0])
    before = df.copy()
    IndicatorEngine(IndicatorConfig(ema_periods=[2], atr_period=2, volume_ma_period=2)).compute(df)
    pd.testing.assert_frame_equal(df, before)


def test_engine_rejects_missing_columns():
    with pytest.raises(ValueError, match="volume"):
        IndicatorEngine(IndicatorConfig()).compute(pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]}))


def test_engine_on_sample_data():
    cfg = load_config()
    df = CSVMarketData(ROOT / cfg["data"]["directory"]).get_ohlcv(cfg["market"]["symbol"], cfg["market"]["timeframe"])
    out = IndicatorEngine.from_config(cfg).compute(df)
    assert out["ema_20"].iloc[19:].notna().all()
    assert out["ema_200"].iloc[:199].isna().all() and out["ema_200"].notna().sum() == 1
    assert (out["atr_14"].dropna() > 0).all()
    assert out["volume_ratio_20"].dropna().mean() == pytest.approx(1.0, abs=0.2)


# ------------------------------------------------------------- No look-ahead
def test_no_lookahead_indicators_unchanged_when_future_appended():
    """Values at candle N must be identical whether or not candles after N exist."""
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 1, 120).cumsum()
    df_full = make_df(c, highs=c + rng.uniform(0, 2, 120), lows=c - rng.uniform(0, 2, 120),
                      volumes=rng.uniform(50, 300, 120))
    eng = IndicatorEngine(IndicatorConfig(ema_periods=[5, 20], atr_period=7, volume_ma_period=10))
    full = eng.compute(df_full)
    for cut in (25, 60, 119):
        partial = eng.compute(df_full.iloc[:cut].copy())
        pd.testing.assert_frame_equal(partial[eng.columns], full[eng.columns].iloc[:cut])


def test_no_lookahead_future_shock_does_not_change_past():
    df = make_df(np.linspace(100, 110, 50))
    eng = IndicatorEngine(IndicatorConfig(ema_periods=[5], atr_period=5, volume_ma_period=5))
    base = eng.compute(df)
    shocked = df.copy()
    shocked.loc[40:, ["open", "high", "low", "close"]] *= 5
    shocked.loc[40:, "volume"] *= 50
    out = eng.compute(shocked)
    pd.testing.assert_frame_equal(base.iloc[:40], out.iloc[:40])
    assert not np.allclose(base["ema_5"].iloc[40:], out["ema_5"].iloc[40:])
