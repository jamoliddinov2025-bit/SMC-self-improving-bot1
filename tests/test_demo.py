"""Integration test: main.py demo runs end to end on local data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import load_config, run_demo  # noqa: E402


def test_demo_runs_offline_and_closes_flat():
    result = run_demo(load_config(), bars=10)
    broker = result["broker"]
    assert len(result["candles"]) == 10
    assert [t["side"] for t in broker.trade_history()] == ["buy", "sell"]
    assert broker.position == 0.0
    assert broker.equity(result["last_price"]) == pytest.approx(broker.cash)
    assert broker.total_fees() > 0
