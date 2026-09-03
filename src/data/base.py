"""Exchange-agnostic market-data interface.

Any provider (local CSV replay now, CCXT later) implements `MarketDataProvider`
and returns candles as a pandas DataFrame with the standard OHLCV columns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    """Interface every market-data source must implement."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame with OHLCV_COLUMNS, sorted by timestamp ascending.

        `limit` returns only the most recent N candles when given.
        """

    def iter_candles(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> Iterator[Candle]:
        """Yield candles one by one (useful for replay / paper trading loops)."""
        df = self.get_ohlcv(symbol, timeframe, limit)
        for row in df.itertuples(index=False):
            yield Candle(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
