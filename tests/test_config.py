"""Basic sanity checks for the scaffold and configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.main import load_config  # noqa: E402


def test_config_loads():
    config = load_config()
    assert config["mode"] in {"paper", "backtest"}
    assert config["mode"] != "live"


def test_config_sections_present():
    config = load_config()
    for section in ("market", "strategy", "indicators", "risk", "execution", "backtesting", "improvement"):
        assert section in config


def test_risk_limits_are_sane():
    risk = load_config()["risk"]
    assert 0 < risk["risk_per_trade_pct"] <= 2
    assert risk["min_risk_reward"] >= 1


def test_package_structure():
    for pkg in ("strategy", "indicators", "risk", "execution", "backtesting", "improvement"):
        assert (ROOT / "src" / pkg / "__init__.py").exists()
