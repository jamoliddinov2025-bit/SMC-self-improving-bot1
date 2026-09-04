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
  history/             # REAL downloaded series + frozen datasets (git-ignored; see data/history/README.md)
src/
  main.py              # entry point: demo | backtest | paper | improve | proposal | data | benchmark
  data/                # market-data interface, CSV replay provider, historical pipeline (validate, dataset, cli)
  data/fetch/          # the ONLY network code: public ccxt OHLCV source + local-file import + downloader
  strategy/            # SMCEngine (structure analysis), POI logic, USDT.D regime, SMCStrategy (signals)
  indicators/          # EMA, ATR, volume + IndicatorEngine (pure functions, no signals)
  risk/                # position sizing, TradeValidator (kill switches), RiskState
  execution/           # PaperBroker, PaperTrader (replay-driven paper loop), state store, candle log
  backtesting/         # point-in-time BacktestEngine, Strategy protocol, aux feeds, journal, metrics
  improvement/         # OFFLINE controlled-improvement framework (walk-forward parameter analysis, proposals)
  benchmark/           # READ-ONLY real-historical baseline of the current strategy (report + metrics)
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
python src/main.py paper [--reset] [--candles N] [--strategy smc|fixture] [--no-usdtd]   # paper-trading loop
python src/main.py improve [--dry-run] [--max-candidates N]   # offline improvement analysis (needs improvement.enabled)
python src/main.py proposal show P-1                          # read-only
python src/main.py proposal apply P-1 --confirm P-1           # writes config/config.proposed.P-1.yaml only
python src/main.py data list|download|update|validate|inspect|export   # real historical data pipeline
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

### Auxiliary data feeds (generic, point-in-time)

`BacktestEngine` and `PaperTrader` accept any number of **named auxiliary series** (`AuxFeed`), each a
`(timestamp, close)` frame with its own timeframe. Every feed is advanced with the rule *aux candle close
time ≤ primary bar close time* and exposed to the strategy as `ctx.aux[<name>]`. The `usdtd:` block defines
the feed named `usdtd`, whose consumer is the `USDTDRegimeDetector` (also exposed as `ctx.regime`); extra
feeds (e.g. BTC.D, TOTAL3, DXY) can be declared under `auxiliary.feeds` and are exposed as raw
`AuxPoint`s. **USDT.D is currently the only feed any strategy consumes.**

### Paper trading (`python src/main.py paper`)

`PaperTrader.process_candle(candle)` runs the backtester's per-bar steps once for **one closed candle**:
advance aux feeds → fill the pending entry at this open through `TradeValidator` → `PaperBroker` (the
backtester's own `try_enter`) → mechanical stop/target/requested-exit → `SMCEngine.update` → indicator
row → `strategy.on_candle(BacktestContext)` → queue BUY for the next open / flag EXIT → risk
mark-to-market → journals → atomic state write. Feeding a replay through it reproduces the
`BacktestEngine` result exactly (tested for the fixture and SMC strategies, with and without USDT.D).

The current CLI is **replay-driven** (it feeds the sample CSV); there is no exchange connection. Re-running
the command resumes from `data/paper/state.json` and skips candles already processed.

Files under `paper.state_directory` (default `data/paper/`, git-ignored):

| File | Content |
|---|---|
| `state.json` | account, risk state (daily PnL, peak, streak), open position, pending entry, cursor, strategy state, config hash — written atomically after every candle |
| `history.csv` | every accepted candle; replayed on restart to rebuild SMC structure and indicators (`paper.warmup_bars`, 0 = all for an exact resume) |
| `candles.csv` | one row per candle: OHLCV, EMA/ATR/volume ratio, regime, new SMC events, strategy state, selected POI, gate failures, signal, risk decision, fill, position, cash/equity/PnL/drawdown/streak |
| `trades.csv`, `rejections.csv` | the `TradeJournal` (round trips and every risk rejection) |

Safety rules: malformed candles (non-finite, bad OHLC ordering, negative volume) are rejected before anything
runs; duplicate / older timestamps are skipped; any exception halts the trader **without** writing state and
drops the pending order, so a restart resumes from the last good candle. The only path into the broker is the
Risk Engine. A `state.json` written with a different trading configuration is refused unless
`paper.allow_config_change: true` (or `--reset`).

### Controlled improvement (`python src/main.py improve`) — human approval required

An **offline analysis tool**, not a self-modifying system. It never edits `src/`, never edits or overwrites
`config/config.yaml`, never touches paper-trading state and never deploys anything. Its only output is a
report directory `data/improvement/<run_id>/` containing `report.md`, `ranking.csv`, `folds.csv`,
`summary.json` and `proposals/P-<n>.yaml`.

- **Parameter optimisation only**, over a hard-coded whitelist of strategy-shape parameters
  (`src/improvement/space.py`): setup age, POI size, rejection threshold, cooldown, EMA/volume filter switches and
  thresholds, stop buffer / min / max ATR, target mode / fixed RR, time stop, CHoCH exit. Risk limits, fees,
  slippage, balance, symbol/timeframe, indicator periods and USDT.D thresholds are **not tunable**.
- Bounds: whitelist ranges/steps, `max_parameter_change_pct` (default 10 % vs the current value), at most
  `max_params_changed_per_proposal` (2) changed parameters, invariants `min_stop_atr ≤ max_stop_atr` and
  `fixed_rr ≥ risk.min_risk_reward`. Candidates are **declared grid values only**: a grid neighbour that
  exceeds the change cap is skipped (never interpolated), so a parameter may legitimately have zero legal
  candidates — the report lists those under "parameters with no legal grid candidate".
- Search v1: deterministic **single-parameter coordinate descent** (no randomness / ML / Bayesian / gradients).
  The `Stage` abstraction leaves room for a future pairwise stage.
- Evaluation: the existing `BacktestEngine` + `SMCStrategy` + `compute_metrics`, with warm-up history fed before
  each slice (warm-up trades are discarded and the slice is re-based). USDT.D alignment is the normal
  point-in-time aux-feed path.
- Data: final **20 % holdout is sealed** — never used for search or ranking; only the top-N survivors are
  evaluated on it once. Development window → **4 anchored walk-forward folds**, 20 % OOS each.
- Constraints (all must pass): total OOS trades ≥ `min_trades_before_change`, per-fold minimum trades, every
  OOS fold positive, median & min OOS score above baseline, OOS/IS expectancy ratio, drawdown limit, minimum
  improvement, neighbourhood stability; then the holdout check. The **baseline runs through the identical
  pipeline**. If the baseline itself has too few OOS trades the run **aborts** with a clear message — the bundled
  200-bar synthetic sample does exactly that, by design.
- Score: median over OOS folds of `avg_R × √trades − dd_penalty × maxDD%`.

Applying a proposal is a separate manual step: `proposal show` prints the overlay, evidence and diff and writes
nothing; `proposal apply <id> --confirm <id>` requires the exact ID twice, refuses if `config.yaml` changed since
the run, and writes **only** `config/config.proposed.<id>.yaml` for you to review and copy by hand.

### Real historical data (`python src/main.py data ...`)

The bot ships with a synthetic sample so everything runs offline; real market data is brought in through a
small pipeline that keeps every consumer unchanged: a **dataset is just a directory** with the canonical
file names (`BTCUSDT_15m.csv[.gz]`, `USDTD_4h.csv[.gz]`) plus a `manifest.json`, so `data.directory` can
point at it and `BacktestEngine`, `PaperTrader` and the improvement framework read it through the same
`CSVMarketData` / `build_aux_feeds` path as before.

```bash
pip install ccxt                                                # optional, download/update only
python src/main.py data download --symbol BTC/USDT --timeframe 15m --from 2020-01-01   # public ccxt OHLCV
python src/main.py data download --symbol USDT.D --timeframe 4h --kind close \
                                 --source file:/path/usdt_dominance_4h.csv --from 2020-01-01   # v1: file import
python src/main.py data update                                  # incremental, re-pulls an overlap, refuses revisions
python src/main.py data validate                                # V1-V10 report (gaps, dupes, order, tz, OHLC, ...)
python src/main.py data inspect                                 # ranges, hashes, aux alignment preview
python src/main.py data export --dataset btc-15m-2020-2025      # frozen, hash-pinned data/history/datasets/<id>/
# then set  data.directory: data/history/datasets/btc-15m-2020-2025/  in config/config.yaml
```

- **Sources**: `ccxt:<exchange>` (default `binance`; any ccxt id such as `bybit`/`okx` works — one series comes
  from one source, no aggregation) and `file:<path>` for offline imports. Only `load_markets` and `fetch_ohlcv`
  (public market-data endpoints) are ever called; API keys/secrets are refused and no environment variables
  are read. Network code lives exclusively in `src/data/fetch/` and is only reachable from the `data` command.
- **USDT.D** has no public OHLCV endpoint: v1 imports a CSV (`timestamp,close`) through the same pipeline.
- **Offline BTC import**: official Binance Vision monthly archives (`BTCUSDT-15m-YYYY-MM.zip`, downloaded in a
  browser) are accepted by the same `file:` source - a directory, single file or glob. The 12-column kline
  layout is mapped to the canonical `timestamp,open,high,low,close,volume` (extra columns discarded, ZIP read
  in memory with a single safe CSV member), then goes through the identical validation/export path; no
  network or API key is involved. The external Binance `.CHECKSUM` is not verified; the repository hashes the
  canonical dataset it produces, and that frozen manifest is the artifact of record (see `data/history/README.md`).
- **Quality**: gaps, duplicates (incl. conflicting), out-of-order rows, naive/off-grid timestamps, OHLC
  violations, outliers, unclosed last candle, hash mismatches — reported in `validation.json/.md`. Gaps are
  **never filled**.
- **Reproducibility**: `manifest.json` pins every file's sha256 and a `dataset_sha256`; loading a tampered
  dataset is refused, the improvement report quotes the dataset id/hash, and the paper-trader `state.json`
  records it as additive metadata (behaviour unchanged).
- Storage stays CSV (gzip for datasets); no Parquet, no database.

### Real-historical baseline benchmark (`python src/main.py benchmark`)

A **read-only** measurement of the *current* `SMCStrategy` on a frozen, hash-verified dataset. It exists to
answer "what does the bot actually do on real history?" honestly - it never tunes, proposes or changes anything.

```bash
python src/main.py data download --symbol BTC/USDT --timeframe 15m --from 2020-01-01   # Step 10 (network)
python src/main.py data download --symbol USDT.D --timeframe 4h --source file:/path/usdtd.csv
python src/main.py data export --dataset btc15m-2020-latest
python src/main.py benchmark --dataset btc15m-2020-latest --dry-run   # verify hashes + validate only
python src/main.py benchmark --dataset btc15m-2020-latest             # -> data/benchmarks/<id>/
```

- Input must be a frozen dataset directory (`manifest.json`); `data/sample/` and other plain folders are refused.
  Datasets exported with `data export --synthetic` are labelled **SYNTHETIC / FIXTURE BASELINE**; anything
  else is labelled **REAL-HISTORICAL BASELINE**. Nothing is ever fabricated, filled or repaired.
- Gate: file hashes must match the manifest (tampering aborts) and Step 10 validation must have no critical
  problem (OHLC violations, off-grid/naive timestamps, duplicates, unclosed last candle, missing required aux
  feed). Warnings such as gaps are printed and written to `validation.json`; `benchmark.fail_on_warnings: true`
  makes them fatal too.
- Output (`data/benchmarks/<benchmark_id>/`, git-ignored): `manifest.json` (dataset id + sha256, per-file
  hashes, strategy, trading config hash and full immutable config snapshot, symbol/timeframe/range, equity,
  fee/slippage assumptions, repo commit, timestamp), `metrics.json`, `trades.csv`, `rejections.csv`,
  `equity_curve.csv`, `validation.json`, `report.md`.
- Metrics separate **BUY signals -> risk-approved -> executed (next-open fill) -> closed**, and add median R,
  max consecutive losses, average trade duration, estimated slippage cost and the strategy gate diagnostics
  to the engine metrics. Below `benchmark.min_trades_for_statistics` closed trades, ratio metrics (win rate,
  expectancy, R, profit factor) are reported as **unavailable** rather than invented.
- The engine, strategy, risk rules and `config/config.yaml` are used as-is and left untouched; the paper
  trader is not involved. Historical results are not a forecast: *past performance does not guarantee
  future performance*.

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
| 8 | Paper-trading loop (replay-driven, resumable) + per-candle logging + generic aux feeds | ✅ done |
| 9 | Controlled improvement framework (offline walk-forward parameter analysis, ranked proposals, manual approval) | ✅ done |
| 10 | Real historical data pipeline (public ccxt download, validation, hash-pinned datasets, file import for USDT.D) | ✅ done |
| 11 | Real-data baseline benchmark (read-only measurement of the current strategy on a frozen dataset) | ✅ done |

## Known limitations

- **Sample data is synthetic.** With only 200 BTC bars the EMA200 exists on a single bar, so the default SMC
  strategy takes ~0 trades on the sample; that is the EMA-trend filter working, not a bug. Real historical
  data is a separate task.
- **No stop modification yet** (break-even / trailing): the backtester has no "amend stop" signal.
- USDT.D ↔ BTC relationship is unverified; defaults are conservative and every effect is switchable.
- **Improvement needs real history.** The framework aborts on the synthetic sample (too few baseline trades); a
  synthetic long series is used in tests for plumbing only. Its report labels synthetic input explicitly.
- **Paper mode is replay-driven.** `python src/main.py paper` feeds the local CSV; a scheduled/live market-data
  provider is a later, separate task. The loop itself only needs closed candles, one at a time.
- A capped `paper.warmup_bars` rebuilds indicators/SMC from a shorter history on restart, so recursive values
  (EMA, Wilder ATR) can differ slightly from an uninterrupted run; `0` (all history) is exact.

- **Loss-streak lock has no automatic unlock yet.** When `consecutive_losses >= risk.max_consecutive_losses`
  the `TradeValidator` rejects all new trades. Since only a winning trade resets the streak and no trade can be
  opened while locked, the lock would otherwise be permanent. The explicit extension point is
  `RiskState.reset_loss_streak(reason)`; a cooldown-based unlock (time/bars since last loss) is planned for a
  later stage and will call this hook. Until then the reset is manual. A new day does **not** clear the streak.

Live trading is intentionally out of scope.
