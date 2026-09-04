"""Deterministic in-memory candle source for Step 10 tests (no network). Helper, not a test module."""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch.source import OHLCVSource, SourceError  # noqa: E402
from src.data.timeframes import series_to_ms, timeframe_to_ms  # noqa: E402


class FakeSource(OHLCVSource):
    """Serves a frame page by page like an exchange; can fail N times, revise rows or pad the last page."""

    online = False

    def __init__(self, df: pd.DataFrame, kind: str = "ohlcv", max_page: int = 500, fail_times: int = 0,
                 include_forming: bool = False, tf: str = "15m"):
        self.id = "fake"
        self.kind = kind
        self.max_page = max_page
        self.df = df.reset_index(drop=True)
        self._ms = series_to_ms(self.df["timestamp"]).to_numpy()
        self._vals = self.df.drop(columns=["timestamp"]).to_numpy()
        self.fail_times = fail_times
        self.calls: List[tuple] = []
        self.include_forming = include_forming
        self.tf_ms = timeframe_to_ms(tf)

    def fetch(self, symbol, timeframe, since_ms, limit):
        self.calls.append((int(since_ms), int(limit)))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise SourceError("simulated network failure")
        start = int(self._ms.searchsorted(int(since_ms), side="left"))
        stop = start + min(int(limit), self.max_page)
        rows = [[int(t)] + [float(x) for x in v] for t, v in zip(self._ms[start:stop], self._vals[start:stop])]
        if self.include_forming and stop >= len(self._ms) and rows:
            last = rows[-1]
            rows.append([last[0] + self.tf_ms] + list(last[1:]))   # a candle whose close time is in the future
        return rows

    def revise(self, index: int, factor: float = 1.01) -> None:
        self._vals = self._vals.copy()
        self._vals[index, :4] = self._vals[index, :4] * factor


def ohlcv_frame(n: int = 2000, start: str = "2023-01-01", tf: str = "15min") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=tf, tz="UTC")
    base = 100 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame({"timestamp": ts, "open": base, "high": base + 0.5, "low": base - 0.5,
                         "close": base + 0.1, "volume": 10.0 + (pd.Series(range(n)) % 7)})


def close_frame(n: int = 300, start: str = "2023-01-01", tf: str = "4h") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=tf, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "close": 4.0 + pd.Series(range(n), dtype=float) * 0.001})


def now_after(df: pd.DataFrame, tf_ms: int = 900_000) -> pd.Timestamp:
    """A `now` at which every row of df is closed."""
    return df["timestamp"].iloc[-1] + pd.Timedelta(tf_ms, unit="ms")
