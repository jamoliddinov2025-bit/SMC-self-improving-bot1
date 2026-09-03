"""SMC Self-Improving Bot - entry point.

PAPER TRADING ONLY. This does not connect to any exchange.

    python src/main.py            broker demo: replay candles, fixed buy-then-sell
    python src/main.py backtest [--strategy smc|fixture] [--no-usdtd]
                                  run the backtest engine on the local SYNTHETIC sample
                                  CSVs. 'fixture' = FixedIntervalTestStrategy (plumbing
                                  only). Results on synthetic data are meaningless.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting import BacktestEngine, FixedIntervalTestStrategy  # noqa: E402
from src.data import CSVMarketData  # noqa: E402
from src.execution import PaperBroker  # noqa: E402
from src.strategy.smc_strategy import SMCStrategy  # noqa: E402

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


def build_strategy(config: dict, name: str):
    if name == "smc":
        return SMCStrategy.from_config(config)
    if name == "fixture":
        return FixedIntervalTestStrategy.from_config(config)
    raise ValueError(f"unknown strategy {name!r} (smc | fixture)")


def run_backtest(config: dict, strategy_name: str = None):
    """Deterministic sample backtest on synthetic data (plumbing check only). Returns (result, strategy)."""
    name = strategy_name or config.get("backtesting", {}).get("strategy", "fixture")
    data = CSVMarketData(directory=ROOT / config["data"]["directory"])
    strategy = build_strategy(config, name)
    engine = BacktestEngine.from_config(config, strategy, run_id=f"sample_{name}", data_root=ROOT)
    result = engine.run_provider(data)
    if config.get("backtesting", {}).get("save_results"):
        result.save(ROOT / config["backtesting"]["results_directory"])
    return result, strategy


def main() -> None:
    config = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        args = sys.argv[2:]
        name = args[args.index("--strategy") + 1] if "--strategy" in args else None
        if "--no-usdtd" in args:
            config.setdefault("usdtd", {})["enabled"] = False
        name = name or config.get("backtesting", {}).get("strategy", "fixture")
        usdtd_on = bool(config.get("usdtd", {}).get("enabled", False))
        print(f"SMC Self-Improving Bot - backtest  strategy={name}  usdtd={'on' if usdtd_on else 'off'}")
        print("NOTE: SYNTHETIC sample data; numbers are NOT trading performance.")
        result, strategy = run_backtest(config, name)
        print(result.format_summary())
        if hasattr(strategy, "diag"):
            d = strategy.diag
            print(f"  strategy diag   : setups={d.setups_armed} buys={d.buy_signals} exits={d.exit_signals} "
                  f"riskoff_skips={d.riskoff_skips} gate_failures={d.gate_failures}")
        return
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
