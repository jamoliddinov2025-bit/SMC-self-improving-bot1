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
  backtesting/         # point-in-time BacktestEngine, Strategy protocol, journal, metrics
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
python src/main.py            # broker plumbing demo
python src/main.py backtest   # backtest engine demo on the sample CSV (fixture strategy)
pytest
```

### Backtesting

`BacktestEngine` replays candles strictly one at a time: entries fill at the **next candle open**
(+ slippage), stops fill at the stop price (or the open on a gap), targets fill without slippage, and a
stop+target conflict in one candle is resolved **stop-first**. All of this is configurable under
`backtesting:` in `config/config.yaml`. Proposals pass through the Risk Engine before `PaperBroker`
executes them; every rejection is journaled. Strategies implement `on_candle(ctx) -> Signal | None` and
only ever see the current bar's indicators and SMC events already confirmed by that bar.

> `python src/main.py backtest` uses `FixedIntervalTestStrategy`, a **test fixture** with no edge, on
> **synthetic** sample data. Its output verifies the pipeline and says nothing about trading performance.

## Development stages

| Step | Goal | Status |
|------|------|--------|
| 1 | Project scaffold and configuration | ✅ done |
| 2 | Market data (CSV replay provider) + PaperBroker | ✅ done |
| 3 | Indicator engine: EMA, ATR, volume | ✅ done |
| 4 | SMC market-structure engine: swings, BOS, CHoCH, liquidity sweeps, FVGs, order blocks (analysis only) | ✅ done |
| 5 | Risk engine: position sizing, trade validator, kill switches, risk state | ✅ done |
| 6 | Backtesting engine + performance evaluation (fixture strategy only) | ✅ done |
| 7 | SMC trading strategy: signal generation (entry / stop / target) | planned |
| 8 | Paper-trading loop (strategy + broker) + trade logging | planned |
| 9 | Controlled strategy improvement (bounded parameter changes, manual approval) | planned |

## Known limitations

- **Loss-streak lock has no automatic unlock yet.** When `consecutive_losses >= risk.max_consecutive_losses`
  the `TradeValidator` rejects all new trades. Since only a winning trade resets the streak and no trade can be
  opened while locked, the lock would otherwise be permanent. The explicit extension point is
  `RiskState.reset_loss_streak(reason)`; a cooldown-based unlock (time/bars since last loss) is planned for a
  later stage and will call this hook. Until then the reset is manual. A new day does **not** clear the streak.

Live trading is intentionally out of scope.
