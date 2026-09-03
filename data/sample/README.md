# Sample data — SYNTHETIC

Both files here are **synthetic random walks generated with a fixed seed**. They exist so the
project runs offline and tests are deterministic. They are **not** real market data and any
backtest output produced from them says nothing about trading performance.

- `BTCUSDT_15m.csv` — 200 candles, OHLCV
- `USDTD_4h.csv`    — 120 candles, timestamp + close (USDT dominance proxy)
