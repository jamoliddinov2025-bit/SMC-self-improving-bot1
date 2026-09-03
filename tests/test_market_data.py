"""Unit tests for the CSV market-data provider."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import OHLCV_COLUMNS, Candle, CSVMarketData, MarketDataProvider  # noqa: E402
from src.main import load_config  # noqa: E402

CSV_BODY = """timestamp,open,high,low,close,volume
2024-01-01T00:30:00Z,102,103,101,102.5,30
2024-01-01T00:00:00Z,100,101,99,100.5,10
2024-01-01T00:15:00Z,101,102,100,101.5,20
"""


@pytest.fixture
def csv_dir(tmp_path):
    (tmp_path / "BTCUSDT_15m.csv").write_text(CSV_BODY)
    return tmp_path


def test_filename_convention():
    assert CSVMarketData.filename_for("BTC/USDT", "15m") == "BTCUSDT_15m.csv"
    assert CSVMarketData.filename_for("ETH/USDT", "4h") == "ETHUSDT_4h.csv"


def test_loads_csv_with_standard_columns(csv_dir):
    df = CSVMarketData(csv_dir).get_ohlcv("BTC/USDT", "15m")
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["close"].dtype == float


def test_rows_sorted_ascending_by_timestamp(csv_dir):
    df = CSVMarketData(csv_dir).get_ohlcv("BTC/USDT", "15m")
    assert df["timestamp"].is_monotonic_increasing
    assert df["open"].tolist() == [100.0, 101.0, 102.0]


def test_limit_returns_most_recent(csv_dir):
    df = CSVMarketData(csv_dir).get_ohlcv("BTC/USDT", "15m", limit=2)
    assert len(df) == 2
    assert df["open"].tolist() == [101.0, 102.0]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVMarketData(tmp_path).get_ohlcv("BTC/USDT", "15m")


def test_missing_column_raises(tmp_path):
    (tmp_path / "BTCUSDT_15m.csv").write_text("timestamp,open,high,low,close\n2024-01-01,1,1,1,1\n")
    with pytest.raises(ValueError, match="volume"):
        CSVMarketData(tmp_path).get_ohlcv("BTC/USDT", "15m")


def test_explicit_file_path_and_header_normalisation(tmp_path):
    p = tmp_path / "whatever.csv"
    p.write_text("Timestamp, Open ,HIGH,low,Close,Volume\n2024-01-01T00:00:00Z,1,2,0.5,1.5,9\n")
    df = CSVMarketData(file_path=p).get_ohlcv("ANY/PAIR", "1d")
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.iloc[0]["high"] == 2.0


def test_epoch_millisecond_timestamps(tmp_path):
    (tmp_path / "BTCUSDT_1h.csv").write_text(
        "timestamp,open,high,low,close,volume\n1704067200000,1,2,0.5,1.5,9\n"
    )
    df = CSVMarketData(tmp_path).get_ohlcv("BTC/USDT", "1h")
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_duplicate_timestamps_dropped(tmp_path):
    (tmp_path / "BTCUSDT_1h.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T00:00:00Z,1,2,0.5,1.5,9\n2024-01-01T00:00:00Z,1,2,0.5,1.5,9\n"
    )
    assert len(CSVMarketData(tmp_path).get_ohlcv("BTC/USDT", "1h")) == 1


def test_iter_candles_yields_candle_objects(csv_dir):
    candles = list(CSVMarketData(csv_dir).iter_candles("BTC/USDT", "15m"))
    assert len(candles) == 3
    assert all(isinstance(c, Candle) for c in candles)
    assert candles[0].close == 100.5 and candles[-1].volume == 30.0


def test_provider_implements_interface(csv_dir):
    assert isinstance(CSVMarketData(csv_dir), MarketDataProvider)


def test_bundled_sample_data_loads_with_config():
    cfg = load_config()
    df = CSVMarketData(ROOT / cfg["data"]["directory"]).get_ohlcv(
        cfg["market"]["symbol"], cfg["market"]["timeframe"]
    )
    assert len(df) == 200
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
