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
  sample/              # SYNTHETIC offline data: BTCUSDT_15m.csv, USDTD_4h.csv (see data/sample/README.md)
src/
  main.py              # entry point (paper demo, no strategy)
  data/                # market-data interface + local CSV replay provider
  strategy/            # SMCEngine (structure analysis), POI logic, USDT.D regime, SMCStrategy (signals)
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
python src/main.py backtest [--strategy smc|fixture] [--no-usdtd]   # backtest on the synthetic sample
pytest
```

### Backtesting

`BacktestEngine` replays candles strictly one at a time: entries fill at the **next candle open**
(+ slippage), stops fill at the stop price (or the open on a gap), targets fill without slippage, and a
stop+target conflict in one candle is resolved **stop-first**. All of this is configurable under
`backtesting:` in `config/config.yaml`. Proposals pass through the Risk Engine before `PaperBroker`
executes them; every rejection is journaled. Strategies implement `on_candle(ctx) -> Signal | None` and
only ever see the current bar's indicators and SMC events already confirmed by that bar.

> The bundled sample data is **synthetic**. Any backtest output from it (fixture or SMC strategy) verifies
> the pipeline only and says nothing about trading performance. `FixedIntervalTestStrategy` is a test fixture.

### SMC strategy (long-only spot)

`SMCStrategy` implements `Strategy.on_candle`. SMC is the primary signal source; indicators only confirm.

| Component | Role | Rule |
|---|---|---|
| BOS / CHoCH | intent | structure must be bullish; the latest bullish BOS (≤ `setup_max_age_bars` old) is the *trigger* |
| Order Block / FVG | where | POI = the OB created by the trigger BOS, or a bullish FVG formed in its impulse (OB first). A POI is mitigated once a candle **closes** below it |
| Touch + rejection | when | candle trades into the POI and shows D1 bullish close in the upper half, D2 a bullish liquidity sweep, or D3 a close back above the zone |
| Liquidity sweep | confluence | optional; adds confluence and moves the stop under the sweep wick |
| Swing highs | target | nearest unbroken swing high giving RR ≥ `risk.min_risk_reward` (mode `structure`), else fixed RR |
| EMA 200/50/20 | filter | close > EMA200 and EMA50 > EMA200; close ≤ EMA20 + k·ATR (no chasing) |
| ATR | geometry | stop = zone low (or sweep wick) − `buffer_atr`·ATR; stop distance within [`min_stop_atr`, `max_stop_atr`]·ATR |
| Volume ratio | filter | trigger candle ≥ `ratio_min`, or the BOS candle was ≥ `bos_ratio_min` |

Exits: the backtester fills stop/target; the strategy adds `EXIT` on a bearish CHoCH and a time stop
(`max_bars_in_trade` without ≥ 1R open profit). Signals are generated on the close of bar *i* and filled at the
open of bar *i+1*; if that open gaps to or below the stop the **Risk Engine** rejects the proposal
(`REJECTED_INVALID_STOP`) — there is no second validation path.

### USDT.D regime filter (optional, `usdtd.enabled`)

A **hypothesis**, not an assumption: rising USDT dominance ≈ risk-off. Computed on the USDT.D close series
only, aligned so a 4h candle is visible **only after its close time**:

- `RISING`  if close > EMA50 and EMA20 slope > +`slope_threshold_pct` and ROC > +`roc_threshold_pct`
- `FALLING` if the mirror conditions hold; otherwise `NEUTRAL`; warm-up / missing data → `UNKNOWN`
- A regime change is adopted only after `confirm_bars` consecutive bars agree (hysteresis).

Only **RISING** changes behaviour, and only by tightening strategy gates: the setup's first (highest) POI is
skipped and a **deeper** valid POI of any kind is required ("wait for the dip"); rejection must be D2/D3;
confluence score ≥ `min_confluence_score`; RR requirement +`rr_add`; volume threshold raised. `FALLING`,
`NEUTRAL`, `UNKNOWN` and `enabled: false` are all identical to running without USDT.D. The regime never
touches `RiskState` or bypasses `TradeValidator`.

## Development stages

| Step | Goal | Status |
|------|------|--------|
| 1 | Project scaffold and configuration | ✅ done |
| 2 | Market data (CSV replay provider) + PaperBroker | ✅ done |
| 3 | Indicator engine: EMA, ATR, volume | ✅ done |
| 4 | SMC market-structure engine: swings, BOS, CHoCH, liquidity sweeps, FVGs, order blocks (analysis only) | ✅ done |
| 5 | Risk engine: position sizing, trade validator, kill switches, risk state | ✅ done |
| 6 | Backtesting engine + performance evaluation (fixture strategy only) | ✅ done |
| 7 | SMC trading strategy (long-only) + optional USDT.D regime filter | ✅ done |
| 8 | Paper-trading loop (strategy + broker) + trade logging | planned |
| 9 | Controlled strategy improvement (bounded parameter changes, manual approval) | planned |

## Known limitations

- **Sample data is synthetic.** With only 200 BTC bars the EMA200 exists on a single bar, so the default SMC
  strategy takes ~0 trades on the sample; that is the EMA-trend filter working, not a bug. Real historical
  data is a separate task.
- **No stop modification yet** (break-even / trailing): the backtester has no "amend stop" signal.
- USDT.D ↔ BTC relationship is unverified; defaults are conservative and every effect is switchable.

- **Loss-streak lock has no automatic unlock yet.** When `consecutive_losses >= risk.max_consecutive_losses`
  the `TradeValidator` rejects all new trades. Since only a winning trade resets the streak and no trade can be
  opened while locked, the lock would otherwise be permanent. The explicit extension point is
  `RiskState.reset_loss_streak(reason)`; a cooldown-based unlock (time/bars since last loss) is planned for a
  later stage and will call this hook. Until then the reset is manual. A new day does **not** clear the streak.

Live trading is intentionally out of scope.
