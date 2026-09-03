"""SMC Self-Improving Bot - entry point.

PAPER TRADING ONLY. This scaffold does not connect to any exchange
and does not implement the trading strategy yet.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
ALLOWED_MODES = {"paper", "backtest"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(ALLOWED_MODES)}")
    return config


def main() -> None:
    config = load_config()
    print("SMC Self-Improving Bot (scaffold)")
    print(f"  mode      : {config['mode']}")
    print(f"  symbol    : {config['market']['symbol']}")
    print(f"  timeframe : {config['market']['timeframe']}")
    print("Strategy not implemented yet. See README for development stages.")


if __name__ == "__main__":
    main()
