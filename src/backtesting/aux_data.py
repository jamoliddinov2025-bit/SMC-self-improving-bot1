"""Point-in-time alignment of an auxiliary series (e.g. USDT.D 4h) to the primary bar series.

For primary bar i (a fully CLOSED candle with open timestamp T_i and close time
T_i + primary_tf), the aligned auxiliary index is the LAST aux candle whose
close time (open + aux_tf) is <= the primary bar's close time. A 4h aux candle
opening at 12:00 is therefore first visible to primary bars closing at/after 16:00.
Bars with no such candle map to -1.
"""

import re
from typing import Union

import pandas as pd

_TF = re.compile(r"^(\d+)([mhdw])$")
_UNIT = {"m": "min", "h": "h", "d": "D", "w": "W"}


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    m = _TF.match(tf.strip().lower())
    if not m:
        raise ValueError(f"unsupported timeframe {tf!r} (expected e.g. 15m, 4h, 1d)")
    return pd.Timedelta(int(m.group(1)), unit=_UNIT[m.group(2)])


def _utc(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, utc=True)
    return out


def align_aux_indices(primary_ts: pd.Series, primary_tf: str, aux_ts: pd.Series, aux_tf: str) -> pd.Series:
    """Return, for each primary bar, the index into `aux_ts` of the latest fully closed aux candle (or -1)."""
    if len(primary_ts) == 0:
        return pd.Series([], dtype=int)
    p_close = _utc(primary_ts) + timeframe_to_timedelta(primary_tf)
    if len(aux_ts) == 0:
        return pd.Series([-1] * len(primary_ts), index=primary_ts.index, dtype=int)
    a_close = _utc(aux_ts) + timeframe_to_timedelta(aux_tf)
    if not a_close.is_monotonic_increasing:
        raise ValueError("auxiliary timestamps must be sorted ascending")
    # searchsorted(side='right') - 1 == number of aux closes <= primary close, minus one
    pos = a_close.searchsorted(p_close.to_numpy(), side="right") - 1
    return pd.Series(pos, index=primary_ts.index, dtype=int)


def aux_filename(symbol: str, timeframe: str) -> str:
    """USDT.D 4h -> USDTD_4h.csv (same convention as CSVMarketData)."""
    return f"{symbol.replace('/', '').replace('.', '')}_{timeframe}.csv"


def coerce_aux_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a full OHLCV frame or a (timestamp, close) frame; return sorted (timestamp, close)."""
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" not in cols or "close" not in cols:
        raise ValueError("auxiliary frame needs 'timestamp' and 'close' columns")
    out = pd.DataFrame({"timestamp": _utc(df[cols["timestamp"]]), "close": df[cols["close"]].astype(float)})
    return out.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
