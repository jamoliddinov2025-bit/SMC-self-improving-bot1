"""Offline source that serves rows from a local CSV / CSV.GZ file.

Used to import third-party dumps (TradingView / exchange archives) - the USDT.D 4h close series in
particular - through exactly the same Downloader -> validate -> SeriesStore path as exchange data,
and by tests as a deterministic "fake exchange". Never touches the network.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.dataset import normalise
from src.data.fetch.source import OHLCVSource, Row, SourceError
from src.data.timeframes import series_to_ms, to_ms


class LocalFileSource(OHLCVSource):
    online = False

    def __init__(self, path, kind: str = "ohlcv", max_page: int = 1000, frame: Optional[pd.DataFrame] = None):
        self.path = Path(path) if path else None
        self.id = f"file:{self.path.name}" if self.path else "file:<frame>"
        self.kind = kind
        self.max_page = max_page
        if frame is None:
            if self.path is None or not self.path.exists():
                raise SourceError(f"import file not found: {self.path}")
            frame = pd.read_csv(self.path)
        try:
            df = normalise(frame, kind)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"{self.id}: cannot parse as {kind} series: {exc}") from exc
        self._ms = series_to_ms(df["timestamp"]).to_numpy()
        self._values = df.drop(columns=["timestamp"]).to_numpy()

    def fetch(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> List[Row]:
        start = int(self._ms.searchsorted(int(since_ms), side="left"))
        stop = start + min(int(limit), self.max_page)
        return [[int(t)] + [float(x) for x in v] for t, v in zip(self._ms[start:stop], self._values[start:stop])]

    @classmethod
    def from_frame(cls, df: pd.DataFrame, kind: str = "ohlcv", max_page: int = 1000, name: str = "frame") -> "LocalFileSource":
        src = cls(None, kind=kind, max_page=max_page, frame=df)
        src.id = f"file:{name}"
        return src

    def first_ms(self) -> Optional[int]:
        return int(self._ms[0]) if len(self._ms) else None

    @staticmethod
    def ms(ts) -> int:
        return to_ms(ts)
