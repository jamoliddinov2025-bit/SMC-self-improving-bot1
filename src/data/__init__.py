"""Market data layer: exchange-agnostic OHLCV providers plus the offline historical-data toolkit
(src.data.dataset / validate / timeframes). Network acquisition lives ONLY in src.data.fetch, which is
never imported from here."""

from src.data.base import OHLCV_COLUMNS, Candle, MarketDataProvider
from src.data.csv_provider import CSVMarketData
from src.data.dataset import DatasetError, DatasetMarketData, dataset_identity, load_manifest, verify_dataset

__all__ = ["OHLCV_COLUMNS", "Candle", "MarketDataProvider", "CSVMarketData", "DatasetMarketData", "DatasetError",
           "dataset_identity", "load_manifest", "verify_dataset"]
