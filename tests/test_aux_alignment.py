"""Point-in-time alignment of an auxiliary (USDT.D 4h) series to 15m primary bars."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.aux_data import align_aux_indices, aux_filename, coerce_aux_frame, timeframe_to_timedelta  # noqa: E402


def ts(s):
    return pd.Series(pd.to_datetime(s, utc=True))


def test_timeframe_parsing():
    assert timeframe_to_timedelta("15m") == pd.Timedelta(minutes=15)
    assert timeframe_to_timedelta("4h") == pd.Timedelta(hours=4)
    assert timeframe_to_timedelta("1d") == pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        timeframe_to_timedelta("fortnight")


def test_aux_candle_visible_only_after_its_close():
    primary = ts(pd.date_range("2024-01-01 11:00", periods=24, freq="15min", tz="UTC"))  # 11:00 .. 16:45
    aux = ts(["2024-01-01 04:00", "2024-01-01 08:00", "2024-01-01 12:00"])              # 4h opens
    idx = align_aux_indices(primary, "15m", aux, "4h").tolist()
    closes = (primary + pd.Timedelta(minutes=15)).dt.strftime("%H:%M").tolist()
    # 08:00 candle closes 12:00 -> visible from the bar closing at 12:00 (open 11:45)
    # 12:00 candle closes 16:00 -> visible from the bar closing at 16:00 (open 15:45)
    for c, i in zip(closes, idx):
        if c < "12:00":
            assert i == 0
        elif c < "16:00":
            assert i == 1
        else:
            assert i == 2


def test_bars_before_first_aux_close_map_to_minus_one():
    primary = ts(pd.date_range("2024-01-01 00:00", periods=20, freq="15min", tz="UTC"))
    aux = ts(["2024-01-01 00:00"])  # closes 04:00 -> never visible within 00:00..05:00? last bar closes 05:00
    idx = align_aux_indices(primary, "15m", aux, "4h").tolist()
    assert idx[:15] == [-1] * 15 and idx[15:] == [0] * 5  # bar 15 opens 03:45, closes 04:00


def test_gaps_in_aux_series_hold_last_closed():
    primary = ts(pd.date_range("2024-01-01 00:00", periods=96, freq="15min", tz="UTC"))  # 1 day
    aux = ts(["2023-12-31 20:00", "2024-01-01 08:00"])   # missing 00:00 and 04:00 candles
    idx = align_aux_indices(primary, "15m", aux, "4h").tolist()
    assert set(idx[:47]) == {0}          # until 08:00 candle closes at 12:00 (bar opening 11:45 = index 47)
    assert set(idx[47:]) == {1}


def test_no_future_index_ever_mapped_property():
    primary = ts(pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC"))
    aux = ts(pd.date_range("2023-12-30", periods=60, freq="4h", tz="UTC"))
    idx = align_aux_indices(primary, "15m", aux, "4h")
    p_close = primary + pd.Timedelta(minutes=15)
    a_close = aux + pd.Timedelta(hours=4)
    for i, j in enumerate(idx):
        if j >= 0:
            assert a_close[j] <= p_close[i]
            if j + 1 < len(aux):
                assert a_close[j + 1] > p_close[i]


def test_empty_inputs():
    assert align_aux_indices(ts([]), "15m", ts(["2024-01-01"]), "4h").tolist() == []
    assert align_aux_indices(ts(["2024-01-01"]), "15m", ts([]), "4h").tolist() == [-1]


def test_unsorted_aux_rejected():
    with pytest.raises(ValueError):
        align_aux_indices(ts(["2024-01-01"]), "15m", ts(["2024-01-02", "2024-01-01"]), "4h")


def test_filename_and_frame_coercion():
    assert aux_filename("USDT.D", "4h") == "USDTD_4h.csv"
    df = pd.DataFrame({"Timestamp": ["2024-01-02", "2024-01-01", "2024-01-01"], "Close": [2, 1, 1], "open": [0, 0, 0]})
    out = coerce_aux_frame(df)
    assert list(out.columns) == ["timestamp", "close"] and out["close"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        coerce_aux_frame(pd.DataFrame({"timestamp": ["2024-01-01"]}))


# ------------------------------------------------------------ generic feeds (Step 8)
from src.backtesting.aux_data import AuxFeed, AuxPoint, aux_specs_from_config  # noqa: E402
from src.main import load_config  # noqa: E402


def test_aux_replayer_matches_vectorised_alignment():
    primary = ts(pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC"))
    aux = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC"),
                        "close": [1.0, 2.0, 3.0, 4.0]})
    expected = align_aux_indices(primary, "15m", aux["timestamp"], "4h").tolist()
    rep = AuxFeed("x", "X", "4h", aux).replayer()
    got = []
    for t in primary:
        st = rep.advance(t + pd.Timedelta(minutes=15))
        got.append(-1 if st is None else st.index)
        assert st is None or (isinstance(st, AuxPoint) and st.close == aux["close"][st.index])
    assert got == expected
    assert got[14] == -1 and got[15] == 0        # bar opening 03:45 closes 04:00 -> candle [00:00,04:00) visible


def test_aux_replayer_equal_close_timestamps_visible_and_never_regresses():
    aux = pd.DataFrame({"timestamp": ts(["2024-01-01 00:00"]), "close": [1.0]})
    rep = AuxFeed("x", "X", "4h", aux).replayer()
    assert rep.advance(pd.Timestamp("2024-01-01 03:59", tz="UTC")) is None
    st = rep.advance(pd.Timestamp("2024-01-01 04:00", tz="UTC"))   # equal close time -> visible
    assert st.index == 0
    assert rep.advance(pd.Timestamp("2024-01-01 02:00", tz="UTC")).index == 0   # cursor never moves backwards


def test_aux_specs_from_config_reserved_name_and_extra_feeds():
    cfg = load_config()
    specs = aux_specs_from_config(cfg)
    assert [s.name for s in specs] == ["usdtd"] and specs[0].consumer == "usdtd_regime"
    cfg["auxiliary"] = {"feeds": {"total3": {"symbol": "TOTAL3", "timeframe": "1d", "enabled": False}}}
    specs = aux_specs_from_config(cfg)
    assert [(s.name, s.symbol, s.timeframe, s.enabled, s.consumer) for s in specs[1:]] == \
           [("total3", "TOTAL3", "1d", False, None)]
