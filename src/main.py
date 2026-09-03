"""SMC Self-Improving Bot - entry point.

PAPER TRADING ONLY. This does not connect to any exchange.
The demo below contains NO trading strategy: it only replays local CSV candles
through the PaperBroker with a fixed buy-then-sell to prove the plumbing works.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CSVMarketData  # noqa: E402
from src.execution import PaperBroker  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config.yaml"
ALLOWED_MODES = {"paper", "backtest"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(ALLOWED_MODES)}")
    return config


def run_demo(config: dict, bars: int = 20) -> dict:
    """Replay the last `bars` candles; buy on the first, sell on the last.

    This is a plumbing test, not a strategy.
    """
    symbol = config["market"]["symbol"]
    timeframe = config["market"]["timeframe"]
    data = CSVMarketData(directory=ROOT / config["data"]["directory"])
    broker = PaperBroker.from_config(config)

    candles = list(data.iter_candles(symbol, timeframe, limit=bars))
    if len(candles) < 2:
        raise ValueError("Need at least 2 candles for the demo")

    first, last = candles[0], candles[-1]
    # Demo action: spend 10% of cash on the first candle's close.
    spend = broker.cash * 0.10
    qty = spend / (first.close * (1 + broker.fee_rate))
    broker.buy(first.close, qty, timestamp=first.timestamp)
    broker.sell(last.close, broker.position, timestamp=last.timestamp)

    return {"broker": broker, "candles": candles, "last_price": last.close}


def main() -> None:
    config = load_config()
    print("SMC Self-Improving Bot - paper demo (no strategy)")
    print(f"  mode      : {config['mode']}")
    print(f"  symbol    : {config['market']['symbol']}")
    print(f"  timeframe : {config['market']['timeframe']}")
    print(f"  data      : {config['data']['directory']}")

    result = run_demo(config)
    broker, candles = result["broker"], result["candles"]
    print(f"\nReplayed {len(candles)} candles: {candles[0].timestamp} -> {candles[-1].timestamp}")
    print("\nTrades:")
    for t in broker.trade_history():
        print(f"  {t['timestamp']}  {t['side']:4s} {t['quantity']:.6f} @ {t['price']:.2f}"
              f"  fee={t['fee']:.4f}  pnl={t['realized_pnl']:+.4f}")
    print("\nPortfolio:")
    for k, v in broker.portfolio(result["last_price"]).items():
        print(f"  {k:16s}: {v:.4f}")


if __name__ == "__main__":
    main()
