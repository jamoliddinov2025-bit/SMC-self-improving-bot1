"""Step 10: timeframe helpers and data-quality validation (V1-V9) on crafted frames. Offline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_source import close_frame, ohlcv_frame  # noqa: E402
from src.data.timeframes import from_ms, gap_runs, iso, off_grid_mask, timeframe_to_ms, to_ms  # noqa: E402
from src.data.validate import ValidationConfig, alignment_preview, validate_frame  # noqa: E402


def codes(rep, severity=None):
    return sorted({i.code for i in rep.issues if severity is None or i.severity == severity})


# ------------------------------------------------------------ timeframes
def test_timeframe_ms_and_roundtrip():
    assert timeframe_to_ms("15m") == 900_000 and timeframe_to_ms("4h") == 14_400_000 and timeframe_to_ms("1d") == 86_400_000
    with pytest.raises(ValueError):
        timeframe_to_ms("15x")
    ms = to_ms("2024-01-01T00:15:00Z")
    assert ms == 1_704_068_100_000 and iso(from_ms(ms)) == "2024-01-01T00:15:00Z"
    assert to_ms(pd.Timestamp("2024-01-01 00:15", tz="Asia/Tashkent")) == ms - 5 * 3_600_000


def test_grid_and_gap_helpers():
    ts = pd.Series(pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:15Z", "2024-01-01T01:00Z", "2024-01-01T01:07Z"]))
    assert off_grid_mask(ts, "15m").tolist() == [False, False, False, True]
    runs = gap_runs(ts.iloc[:3], "15m")
    assert len(runs) == 1 and iso(runs[0][0]) == "2024-01-01T00:30:00Z" and iso(runs[0][1]) == "2024-01-01T00:45:00Z" and runs[0][2] == 2


# ------------------------------------------------------------ validation
def test_clean_frame_is_ok():
    rep = validate_frame(ohlcv_frame(500), "x", "15m")
    assert rep.status == "ok" and rep.rows == 500 and rep.expected_rows == 500 and rep.missing_bars == 0
    assert "ok" in rep.summary_line() and "No issues" in rep.to_markdown()


def test_v1_schema_and_numeric():
    assert codes(validate_frame(ohlcv_frame(10).drop(columns=["volume"]), "x", "15m")) == ["V1"]
    df = ohlcv_frame(10).astype({"close": object}); df.loc[3, "close"] = "abc"
    assert codes(validate_frame(df, "x", "15m")) == ["V1"]
    assert codes(validate_frame(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), "x", "15m")) == ["V1"]


def test_v2_timezone_and_grid():
    df = ohlcv_frame(10)
    naive = df.copy(); naive["timestamp"] = naive["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    rep = validate_frame(naive, "x", "15m")
    assert codes(rep) == ["V2"] and "naive" in rep.issues[0].message
    epoch_ms = df["timestamp"].astype("datetime64[ms, UTC]").astype("int64")
    ms = df.copy(); ms["timestamp"] = epoch_ms
    assert validate_frame(ms, "x", "15m").status == "ok"                      # epoch ms accepted
    sec = df.copy(); sec["timestamp"] = epoch_ms // 1000
    assert validate_frame(sec, "x", "15m").status == "ok"                     # epoch s accepted
    off = df.copy(); off.loc[4, "timestamp"] = off.loc[4, "timestamp"] + pd.Timedelta("3min")
    rep = validate_frame(off, "x", "15m")
    assert "V2" in codes(rep, "error")
    plus = df.copy(); plus["timestamp"] = plus["timestamp"].dt.tz_convert("Asia/Tashkent").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    assert validate_frame(plus, "x", "15m").status == "ok"                    # offset-aware strings are converted


def test_v3_v4_order_and_duplicates():
    df = ohlcv_frame(20)
    rev = df.iloc[::-1]
    assert "V3" in codes(validate_frame(rev, "x", "15m"), "error")
    dup = pd.concat([df, df.iloc[[5]]])
    rep = validate_frame(dup, "x", "15m")
    v4 = next(i for i in rep.issues if i.code == "V4")
    assert rep.duplicates == 1 and v4.detail["conflicting"] == 0
    conflict = df.iloc[[5]].copy(); conflict["close"] += 1
    rep = validate_frame(pd.concat([df, conflict]), "x", "15m")
    assert next(i for i in rep.issues if i.code == "V4").detail["conflicting"] == 1


def test_v5_gaps_warning_then_error():
    df = ohlcv_frame(1000).drop(index=[100, 101, 102]).reset_index(drop=True)
    rep = validate_frame(df, "x", "15m")
    assert rep.status == "warnings" and rep.missing_bars == 3 and len(rep.gap_runs) == 1
    assert rep.gap_runs[0] == {"from": "2023-01-02T01:00:00Z", "to": "2023-01-02T01:30:00Z", "missing": 3}
    big = ohlcv_frame(1000).drop(index=range(100, 200)).reset_index(drop=True)
    assert validate_frame(big, "x", "15m").status == "failed"
    assert validate_frame(big, "x", "15m", cfg=ValidationConfig(max_gap_pct=20)).status == "warnings"
    assert validate_frame(df, "x", "15m", cfg=ValidationConfig(fail_on_warnings=True)).status == "failed"


def test_v6_ohlc_sanity():
    df = ohlcv_frame(20); df.loc[3, "low"] = df.loc[3, "high"] + 1
    assert "V6" in codes(validate_frame(df, "x", "15m"), "error")
    df = ohlcv_frame(20); df.loc[4, "volume"] = -1
    assert "V6" in codes(validate_frame(df, "x", "15m"), "error")
    df = ohlcv_frame(20); df.loc[4, ["open", "high", "low", "close"]] = 0
    assert "V6" in codes(validate_frame(df, "x", "15m"), "error")


def test_v7_outliers_are_warnings_only():
    df = ohlcv_frame(400); df.loc[300, "high"] = df.loc[300, "low"] + 100
    rep = validate_frame(df, "x", "15m")
    assert rep.status == "warnings" and "V7" in codes(rep, "warning")
    df = ohlcv_frame(400); df.loc[100:130, "volume"] = 0
    assert any("zero-volume" in i.message for i in validate_frame(df, "x", "15m").issues)
    df = ohlcv_frame(400); df.loc[100:130, ["open", "high", "low", "close"]] = 50.0
    assert any("flat" in i.message for i in validate_frame(df, "x", "15m").issues)


def test_v8_last_candle_must_be_closed():
    df = ohlcv_frame(20)
    last_open = df["timestamp"].iloc[-1]
    assert validate_frame(df, "x", "15m", now=last_open + pd.Timedelta("15min")).status == "ok"
    rep = validate_frame(df, "x", "15m", now=last_open + pd.Timedelta("14min"))
    assert "V8" in codes(rep, "error")


def test_v9_close_series_and_alignment_preview():
    aux = close_frame(50)
    assert validate_frame(aux, "u", "4h", kind="close").status == "ok"
    bad = aux.copy(); bad.loc[3, "close"] = 0
    assert "V9" in codes(validate_frame(bad, "u", "4h", kind="close"), "error")
    prim = ohlcv_frame(200)
    a = alignment_preview(prim["timestamp"], "15m", aux["timestamp"], "4h")
    assert a["first_visible_primary_index"] == 15      # first 4h candle closes at 04:00 == close of 15m bar #15
    assert 0 < a["coverage_pct"] < 100 and a["aux_ends_before_primary"] is False


def test_report_serialises():
    rep = validate_frame(ohlcv_frame(1000).drop(index=[10]).reset_index(drop=True), "x", "15m")
    d = rep.to_dict()
    assert d["status"] == "warnings" and d["missing_bars"] == 1 and [i["code"] for i in d["issues"]] == ["V5"]
    assert "| V5 |" in rep.to_markdown()
