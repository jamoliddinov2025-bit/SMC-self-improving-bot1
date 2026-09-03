"""Local CSV / replay market-data provider.

Runs fully offline. CSV files must contain the columns:
    timestamp, open, high, low, close, volume

Files are resolved as `<directory>/<SYMBOL>_<timeframe>.csv`, where the symbol
separator "/" is replaced by "" (e.g. BTC/USDT 15m -> data/BTCUSDT_15m.csv).
An explicit file path can also be passed to bypass this convention.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.data.base import OHLCV_COLUMNS, MarketDataProvider


class CSVMarketData(MarketDataProvider):
    def __init__(self, directory: Union[str, Path] = "data", file_path: Optional[Union[str, Path]] = None):
        self.directory = Path(directory)
        self.file_path = Path(file_path) if file_path else None

    @staticmethod
    def filename_for(symbol: str, timeframe: str) -> str:
        return f"{symbol.replace('/', '')}_{timeframe}.csv"

    def resolve_path(self, symbol: str, timeframe: str) -> Path:
        return self.file_path or self.directory / self.filename_for(symbol, timeframe)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.resolve_path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"Market data file not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} is missing required columns: {missing}")

        df = df[OHLCV_COLUMNS].copy()
        df["timestamp"] = _parse_timestamps(df["timestamp"])
        for col in OHLCV_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

        df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if limit is not None:
            df = df.tail(limit).reset_index(drop=True)
        return df


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Accept ISO strings, or epoch seconds / milliseconds."""
    if pd.api.types.is_numeric_dtype(series):
        unit = "ms" if series.iloc[0] > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True)
