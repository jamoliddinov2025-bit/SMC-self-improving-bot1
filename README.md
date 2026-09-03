# SMC Self-Improving Bot

A simple automated cryptocurrency **spot** trading bot based on Smart Money Concepts (SMC),
with strict risk management, backtesting, paper trading, and controlled strategy improvement.

> **PAPER TRADING ONLY.** This project does not connect to a real exchange, does not use
> real API keys, and does not implement live trading. `mode` in the config only accepts
> `paper` or `backtest`.

## Planned strategy components

- **SMC (primary strategy)**: Break of Structure (BOS), Change of Character (CHoCH),
  liquidity sweeps, order blocks, fair value gaps (FVG)
- **Indicators (filters/confirmation)**: EMA, ATR, volume
- **Risk management**: fixed % risk per trade, ATR-based stops, minimum R:R, daily loss limit
- **Backtesting** and **paper trading** with full trade logging
- **Performance evaluation** and **controlled strategy improvement**

## Project structure

```
config/
  config.yaml          # all bot settings (paper mode only)
data/
  sample/              # bundled offline CSV replay data (BTCUSDT_15m.csv)
src/
  main.py              # entry point (paper demo, no strategy)
  data/                # market-data interface + local CSV replay provider
  strategy/            # SMCEngine: swings, BOS, CHoCH, sweeps, FVG, order blocks (analysis only, no signals)
  indicators/          # EMA, ATR, volume + IndicatorEngine (pure functions, no signals)
  risk/                # position sizing, TradeValidator (kill switches), RiskState
  execution/           # PaperBroker (simulated spot broker) and trade history
  backtesting/         # backtest engine and performance metrics
  improvement/         # controlled parameter improvement
tests/
requirements.txt
README.md
.gitignore
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
pytest
```

## Development stages

| Stage | Goal | Status |
|-------|------|--------|
| 0 | Project scaffold and configuration | ✅ done |
| 1 | Market data (CSV replay provider) + PaperBroker | ✅ done |
| 2 | Indicators: EMA, ATR, volume | ✅ done |
| 3 | SMC market structure: swing points, BOS, CHoCH | ✅ done |
| 4 | Liquidity sweeps, order blocks, fair value gaps | ✅ done |
| 5 | Signal generation (entry / stop / target) | planned |
| 6 | Risk management module | ✅ done |
| 7 | Backtesting engine + performance evaluation | planned |
| 8 | Paper-trading loop (strategy + broker) + trade logging | planned |
| 9 | Controlled strategy improvement (bounded parameter changes, manual approval) | planned |

## Known limitations

- **Loss-streak lock has no automatic unlock yet.** When `consecutive_losses >= risk.max_consecutive_losses`
  the `TradeValidator` rejects all new trades. Since only a winning trade resets the streak and no trade can be
  opened while locked, the lock would otherwise be permanent. The explicit extension point is
  `RiskState.reset_loss_streak(reason)`; a cooldown-based unlock (time/bars since last loss) is planned for a
  later stage and will call this hook. Until then the reset is manual. A new day does **not** clear the streak.

Live trading is intentionally out of scope.
