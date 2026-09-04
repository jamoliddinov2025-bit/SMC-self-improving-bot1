"""Exponential Moving Average.

Standard definition: alpha = 2 / (period + 1). The first EMA value is seeded
with the simple average of the first `period` closes, so the first
`period - 1` values are NaN (warm-up). Every value at index N depends only on
closes at indices <= N.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return pd.Series(out, index=series.index, name=f"ema_{period}")

    alpha = 2.0 / (period + 1)
    out[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index, name=f"ema_{period}")
