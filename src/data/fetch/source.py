"""Candle sources for the historical downloader.

`OHLCVSource` is the only interface the Downloader knows. New exchanges are added by
registering another source id in `make_source()` (all CCXT exchanges already work through
`CCXTPublicSource`); consumers (Downloader, CLI, tests) never change. There is deliberately
no multi-exchange aggregation: one series comes from one source.

Rows are CCXT-shaped lists: [open_time_ms, open, high, low, close, volume]. Close-only
series (e.g. an imported USDT.D file) use [open_time_ms, close].
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

Row = List[float]


class SourceError(Exception):
    pass


class OHLCVSource(ABC):
    id: str = "abstract"
    kind: str = "ohlcv"          # "ohlcv" | "close"
    online: bool = False         # True only for sources that touch the network
    max_page: int = 1000

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> List[Row]:
        """Return up to `limit` rows with open_time >= since_ms, ascending. Empty list == nothing more."""

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "online": self.online, "max_page": self.max_page}

    def close(self) -> None:
        """Release resources (no-op by default)."""


def parse_source_spec(spec: str):
    """'ccxt:binance' -> ('ccxt', 'binance'); 'file:/path/x.csv' -> ('file', '/path/x.csv'); 'binance' -> ('ccxt','binance')."""
    if ":" in spec:
        scheme, rest = spec.split(":", 1)
        return scheme.strip().lower(), rest.strip()
    return "ccxt", spec.strip().lower()


def make_source(spec: str, kind: str = "ohlcv", fetch_cfg: Optional[Dict[str, Any]] = None) -> OHLCVSource:
    """Factory used by the CLI. `file:` sources are offline; `ccxt:` sources are the only online ones."""
    scheme, target = parse_source_spec(spec)
    if scheme == "file":
        from src.data.fetch.csv_source import LocalFileSource
        return LocalFileSource(target, kind=kind)
    if scheme == "ccxt":
        from src.data.fetch.ccxt_source import CCXTPublicSource
        return CCXTPublicSource(target, fetch_cfg=fetch_cfg)
    raise SourceError(f"unknown source scheme {scheme!r} (expected ccxt:<exchange> or file:<path>)")
