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
