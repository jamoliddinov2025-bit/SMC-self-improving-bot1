"""Timeframe arithmetic for historical series (pure functions, no I/O)."""

import re

import pandas as pd

_TF = re.compile(r"^(\d+)([mhdw])$")
_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def timeframe_to_ms(tf: str) -> int:
    m = _TF.match(tf.strip().lower())
    if not m:
        raise ValueError(f"unsupported timeframe {tf!r} (expected e.g. 15m, 4h, 1d)")
    return int(m.group(1)) * _MS[m.group(2)]


def to_ms(ts) -> int:
    """UTC pandas Timestamp / ISO string -> epoch milliseconds."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


def iso(ts) -> str:
    """Canonical storage format: 2024-01-01T00:15:00Z."""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def series_to_ms(ts: pd.Series) -> pd.Series:
    """Datetime series (any resolution) -> epoch milliseconds (int64)."""
    return pd.to_datetime(ts, utc=True).astype("datetime64[ms, UTC]").astype("int64")


def off_grid_mask(ts: pd.Series, tf: str) -> pd.Series:
    """True where an open timestamp is not a multiple of the timeframe (UTC epoch grid)."""
    return (series_to_ms(ts) % timeframe_to_ms(tf)) != 0


def gap_runs(ts: pd.Series, tf: str):
    """Missing grid points between consecutive rows: list of (from_ts, to_ts, missing_bars)."""
    ms = series_to_ms(ts).to_numpy()
    step = timeframe_to_ms(tf)
    runs = []
    for prev, nxt in zip(ms[:-1], ms[1:]):
        missing = int((nxt - prev) // step) - 1
        if missing > 0:
            runs.append((from_ms(prev + step), from_ms(nxt - step), missing))
    return runs
