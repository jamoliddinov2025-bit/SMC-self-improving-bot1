# data/history — real historical market data (not committed)

Everything in this directory is produced by `python src/main.py data ...` and is git-ignored.

```
<source>/<SERIES>/<SERIES>.csv      working copy (BTCUSDT_15m.csv, USDTD_4h.csv, ...)
<source>/<SERIES>/series.json       metadata: rows, range, sha256, fetch log, validation status
<source>/<SERIES>/validation.{json,md}
datasets/<dataset_id>/manifest.json frozen bundle: per-file sha256 + dataset_sha256
datasets/<dataset_id>/<SERIES>.csv.gz
```

Point `data.directory` in `config/config.yaml` at a `datasets/<dataset_id>/` folder to run
backtests, paper replay or the improvement framework on real data. Nothing here is ever
modified by those commands. Gaps are reported, never filled.

## Offline import of official Binance Vision archives (no network, no API key)

When the machine running the bot cannot reach exchange APIs, download the official monthly spot
kline archives in a browser (`https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/15m/`,
files named `BTCUSDT-15m-YYYY-MM.zip`), put them in one folder and import them through the same
`file:` source used for USDT.D:

```bash
python src/main.py data download --symbol BTC/USDT --timeframe 15m \
       --source file:/path/to/binance_vision_archives/ --from 2020-01-01      # dir, single .zip/.csv, or glob
python src/main.py data validate
python src/main.py data export --dataset btc-15m-2020-latest
```

- The 12-column Binance kline layout (header optional) is recognised; only `open_time` (epoch ms, or µs in
  2025+ archives, detected by magnitude) → `timestamp` (UTC, candle OPEN) and `open,high,low,close,volume`
  are kept. `close_time`, quote volume, trade count, taker volumes and `ignore` are discarded.
- ZIPs are read in memory: exactly one top-level `.csv` member is accepted; nested paths, absolute paths,
  `..`, extra CSVs or non-zip files are refused. Nothing is extracted to disk.
- Monthly files are combined in name order. Identical overlapping rows collapse; conflicting duplicates
  abort. Gaps, unsorted rows, OHLC violations and a still-forming last candle are handled by the normal
  Step 10 validation / forming-candle rule - never by the archive's file name.
- Provenance: archive contents are imported offline; the original Binance Vision `.CHECKSUM` file is **not**
  verified by this import (check it manually if you need to authenticate the download). The repository hashes
  the resulting canonical files only - archive names, paths and archive hashes are not part of any hash. The
  frozen `datasets/<id>/` manifest (`sha256` per file, `dataset_sha256`) is the artifact of record and is what
  the Step 11 benchmark verifies.
