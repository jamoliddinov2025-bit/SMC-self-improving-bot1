"""Average True Range (Wilder).

True Range at candle N:
    max(high - low, |high - prev_close|, |low - prev_close|)
The first candle has no previous close, so its TR is simply high - low.

ATR is seeded with the simple mean of the first `period` true ranges, then
smoothed with Wilder's method: ATR_n = (ATR_{n-1} * (period - 1) + TR_n) / period.
The first `period - 1` values are NaN (warm-up). No future data is used.
"""

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)
    return tr.rename("true_range")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(df).to_numpy()
    out = np.full(len(tr), np.nan)
    if len(tr) < period:
        return pd.Series(out, index=df.index, name=f"atr_{period}")

    out[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=df.index, name=f"atr_{period}")
