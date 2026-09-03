"""Simple volume metrics.

- volume_sma:   simple moving average of volume over `period` (NaN during warm-up)
- volume_ratio: current volume / volume_sma  (>1 = above-average volume)

Both use only the current and previous candles.
"""

import numpy as np
import pandas as pd


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    return volume.astype(float).rolling(period, min_periods=period).mean().rename(f"volume_sma_{period}")


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    sma = volume_sma(volume, period)
    ratio = volume.astype(float) / sma.replace(0.0, np.nan)
    return ratio.rename(f"volume_ratio_{period}")
