"""Historical candle acquisition (download / import). The ONLY package allowed to reach the network,
and only through `ccxt_source.CCXTPublicSource` public market-data endpoints. Nothing in
src/backtesting, src/execution, src/strategy, src/risk or src/improvement may import this package."""

from src.data.fetch.downloader import DownloadError, Downloader, DownloadResult, FetchConfig
from src.data.fetch.source import OHLCVSource, SourceError, make_source, parse_source_spec

__all__ = ["Downloader", "DownloadResult", "DownloadError", "FetchConfig", "OHLCVSource", "SourceError",
           "make_source", "parse_source_spec"]
