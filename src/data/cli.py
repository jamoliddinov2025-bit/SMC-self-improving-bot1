"""`python src/main.py data <command>` - historical data pipeline commands.

    data list                                         series + datasets on disk (offline)
    data download --symbol S --timeframe TF [--source ccxt:binance|file:PATH] [--kind ohlcv|close]
                  [--from ISO] [--to ISO] [--dry-run]                                      (network for ccxt:)
    data update   [--series BTCUSDT_15m ...] [--dry-run]     extend configured/downloaded series (network for ccxt:)
    data validate [--series ID | --dataset ID]               re-run V1-V10, write validation.json/.md (offline)
    data inspect  [--series ID | --dataset ID]               range, rows, gaps, hashes, aux alignment (offline)
    data export   --dataset ID [--from ISO] [--to ISO] [--series ID ...] [--no-compress] [--overwrite]
                                                             freeze a hash-pinned dataset dir (offline)

Only `download`/`update` with a ccxt: source touch the network, via src/data/fetch/ccxt_source.py.
These commands never write to config/config.yaml, src/, data/paper/ or data/improvement/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.dataset import (MANIFEST_NAME, DatasetError, SeriesStore, export_dataset, load_manifest, read_series_csv,
                              series_id, verify_dataset)
from src.data.fetch import DownloadError, Downloader, FetchConfig, SourceError, make_source, parse_source_spec
from src.data.validate import ValidationConfig, alignment_preview, validate_frame

REGIME_SYMBOL_KEY = ("usdtd", "symbol")


def _flag(args: List[str], name: str, default=None):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else default


def _multi(args: List[str], name: str) -> List[str]:
    out, take = [], False
    for a in args:
        if a == name:
            take = True
            continue
        if a.startswith("--"):
            take = False
        elif take:
            out.append(a)
    return out


class DataCLI:
    def __init__(self, config: Dict[str, Any], root: Path, out=print):
        self.config, self.root, self.out = config, Path(root), out
        h = config.get("history", {}) or {}
        self.hroot = self.root / h.get("root", "data/history/")
        self.default_source = h.get("default_source", "ccxt:binance")
        self.series_cfg: List[Dict[str, Any]] = list(h.get("series", []) or [])
        self.fcfg = FetchConfig.from_config(config)
        self.vcfg = ValidationConfig.from_config(config)
        self.compress = bool((h.get("export", {}) or {}).get("compress", True))

    # ------------------------------------------------------------ helpers
    def _source_dir_name(self, source_spec: str) -> str:
        scheme, target = parse_source_spec(source_spec)
        return target if scheme == "ccxt" else "file"

    def _cfg_for(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        for s in self.series_cfg:
            if s.get("symbol") == symbol and s.get("timeframe") == timeframe:
                return s
        return {}

    def _store(self, symbol: str, timeframe: str, kind: str, source_spec: str) -> SeriesStore:
        return SeriesStore(self.hroot, self._source_dir_name(source_spec), symbol, timeframe, kind)

    def _find_series(self, sid: str) -> Optional[SeriesStore]:
        """Locate a downloaded series by id (BTCUSDT_15m) under any source directory; config wins on ties."""
        hits = []
        if self.hroot.exists():
            for src_dir in sorted(p for p in self.hroot.iterdir() if p.is_dir() and p.name != "datasets"):
                meta_path = src_dir / sid / "series.json"
                if meta_path.exists():
                    m = json.loads(meta_path.read_text())
                    hits.append(SeriesStore(self.hroot, src_dir.name, m["symbol"], m["timeframe"], m.get("kind", "ohlcv")))
        for h in hits:
            c = self._cfg_for(h.symbol, h.timeframe)
            if c and self._source_dir_name(c.get("source", self.default_source)) == h.source:
                return h
        return hits[0] if hits else None

    def _banner(self, source) -> None:
        if getattr(source, "online", False):
            self.out(f"  source          : {source.id} (ccxt PUBLIC market-data endpoints; no credentials, no trading API)")
        else:
            self.out(f"  source          : {source.id} (local file, offline)")

    # ------------------------------------------------------------ commands
    def run(self, args: List[str]) -> int:
        if not args:
            self.out(__doc__)
            return 2
        cmd, rest = args[0], args[1:]
        fn = {"list": self.cmd_list, "download": self.cmd_download, "update": self.cmd_update,
              "validate": self.cmd_validate, "inspect": self.cmd_inspect, "export": self.cmd_export}.get(cmd)
        if fn is None:
            self.out(__doc__)
            return 2
        try:
            return fn(rest)
        except (SourceError, DownloadError, DatasetError, FileNotFoundError, ValueError) as exc:
            self.out(f"  REFUSED         : {exc}")
            return 1

    def cmd_list(self, args: List[str]) -> int:
        self.out(f"history root: {self.hroot}")
        found = False
        if self.hroot.exists():
            for src_dir in sorted(p for p in self.hroot.iterdir() if p.is_dir() and p.name != "datasets"):
                for sd in sorted(p for p in src_dir.iterdir() if (p / "series.json").exists()):
                    m = json.loads((sd / "series.json").read_text())
                    found = True
                    self.out(f"  series  {m['series_id']:<16} {src_dir.name:<10} {m['rows']:>9,} rows  {m['first_open']} -> "
                             f"{m['last_open']}  validation {m.get('validation_status', '?')}  sha256 {m['sha256'][:12]}")
            ds = self.hroot / "datasets"
            if ds.exists():
                for d in sorted(p for p in ds.iterdir() if (p / MANIFEST_NAME).exists()):
                    m = load_manifest(d)
                    found = True
                    self.out(f"  dataset {m['dataset_id']:<16} primary {m['primary']['symbol']} {m['primary']['timeframe']} "
                             f"{m['primary']['rows']:,} rows, {len(m.get('auxiliary', []))} aux  dataset_sha256 {m['dataset_sha256'][:12]}")
        if not found:
            self.out("  (nothing downloaded yet)")
        self.out(f"  configured series: {[series_id(s['symbol'], s['timeframe']) for s in self.series_cfg]}")
        return 0

    def cmd_download(self, args: List[str]) -> int:
        symbol, tf = _flag(args, "--symbol"), _flag(args, "--timeframe")
        if not symbol or not tf:
            raise ValueError("--symbol and --timeframe are required")
        c = self._cfg_for(symbol, tf)
        kind = _flag(args, "--kind", c.get("kind", "ohlcv"))
        spec = _flag(args, "--source", c.get("source", self.default_source))
        if parse_source_spec(spec)[0] == "file" and not parse_source_spec(spec)[1]:
            raise ValueError(f"{symbol} {tf} is a file-import series: pass --source file:<path/to.csv>")
        start = _flag(args, "--from", c.get("start"))
        end = _flag(args, "--to", c.get("end"))
        if not start:
            raise ValueError("--from ISO date is required (or set history.series[].start)")
        source = make_source(spec, kind=kind, fetch_cfg=self.fcfg.as_dict())
        store = self._store(symbol, tf, kind, spec)
        self.out(f"SMC Self-Improving Bot - data download  {symbol} {tf} ({kind})")
        self._banner(source)
        dl = Downloader(source, store, self.fcfg, self.vcfg, progress=self.out)
        res = dl.download(start, end, dry_run="--dry-run" in args)
        source.close()
        return self._report(res, store)

    def cmd_update(self, args: List[str]) -> int:
        wanted = _multi(args, "--series")
        targets: List[SeriesStore] = []
        if wanted:
            for sid in wanted:
                st = self._find_series(sid)
                if st is None:
                    raise DownloadError(f"series {sid} not found under {self.hroot} - download it first")
                targets.append(st)
        else:
            for c in self.series_cfg:
                st = self._find_series(series_id(c["symbol"], c["timeframe"]))
                if st is not None:
                    targets.append(st)
        if not targets:
            raise DownloadError("nothing to update (no downloaded series match)")
        rc = 0
        for st in targets:
            c = self._cfg_for(st.symbol, st.timeframe)
            spec = c.get("source", self.default_source)
            if st.source == "file":
                path = _flag(args, "--source")
                if not path:
                    self.out(f"  {st.id}: file-import series - re-import with `data download --source file:<path>`; skipped")
                    continue
                spec = path
            source = make_source(spec, kind=st.kind, fetch_cfg=self.fcfg.as_dict())
            self.out(f"SMC Self-Improving Bot - data update  {st.id}")
            self._banner(source)
            res = Downloader(source, st, self.fcfg, self.vcfg, progress=self.out).update(dry_run="--dry-run" in args)
            source.close()
            rc = max(rc, self._report(res, st))
        return rc

    def cmd_validate(self, args: List[str]) -> int:
        ds = _flag(args, "--dataset")
        if ds:
            d = self.hroot / "datasets" / ds
            problems = verify_dataset(d)
            m = load_manifest(d)
            if m is None:
                raise DatasetError(f"dataset {ds} not found at {d}")
            rc = 0
            for entry in [m["primary"]] + list(m.get("auxiliary", [])):
                raw = read_series_csv(d / entry["file"], entry["kind"])
                rep = validate_frame(raw, Path(entry["file"]).name.split(".")[0], entry["timeframe"], entry["kind"], self.vcfg)
                self.out("  " + rep.summary_line())
                rc = max(rc, 0 if rep.ok else 1)
            for p in problems:
                self.out(f"  ERROR           : {p}")
            self.out(f"  dataset {ds}: {'ok' if not problems and rc == 0 else 'FAILED'}  dataset_sha256 {m['dataset_sha256']}")
            return 1 if problems else rc
        sids = _multi(args, "--series") or [series_id(c["symbol"], c["timeframe"]) for c in self.series_cfg]
        rc = 0
        for sid in sids:
            st = self._find_series(sid)
            if st is None:
                self.out(f"  {sid}: not downloaded")
                continue
            rep = validate_frame(st.load_raw(), st.id, st.timeframe, st.kind, self.vcfg)
            st.write_validation(rep)
            hash_ok = st.verify_hash()
            self.out("  " + rep.summary_line() + ("" if hash_ok else "  [V10 sha256 MISMATCH vs series.json]"))
            for i in rep.issues:
                self.out(f"      {i.code} {i.severity:<7} x{i.count:<5} {i.message}")
            rc = max(rc, 0 if rep.ok and hash_ok else 1)
        return rc

    def cmd_inspect(self, args: List[str]) -> int:
        ds = _flag(args, "--dataset")
        if ds:
            d = self.hroot / "datasets" / ds
            m = load_manifest(d)
            if m is None:
                raise DatasetError(f"dataset {ds} not found at {d}")
            self.out(json.dumps({k: v for k, v in m.items()}, indent=2, default=str))
            self.out(f"  verification    : {verify_dataset(d) or 'ok'}")
            return 0
        sids = _multi(args, "--series") or [series_id(c["symbol"], c["timeframe"]) for c in self.series_cfg]
        primary = self._find_series(series_id(self.config["market"]["symbol"], self.config["market"]["timeframe"]))
        pdf = primary.load() if primary else None
        for sid in sids:
            st = self._find_series(sid)
            if st is None:
                self.out(f"  {sid}: not downloaded")
                continue
            m = st.meta() or {}
            self.out(f"  {st.id}  source {st.source}  kind {st.kind}  rows {m.get('rows'):,}  {m.get('first_open')} -> {m.get('last_open')}")
            self.out(f"      sha256 {m.get('sha256')}  validation {m.get('validation_status', '?')}  fetches {len(m.get('fetch_log', []))}")
            val = st.dir / "validation.json"
            if val.exists():
                v = json.loads(val.read_text())
                self.out(f"      gaps {len(v['gap_runs'])} runs / {v['missing_bars']} bars ({v['missing_pct']}%)  duplicates {v['duplicates']}")
            if pdf is not None and st.kind == "close":
                a = alignment_preview(pdf["timestamp"], primary.timeframe, st.load()["timestamp"], st.timeframe)
                self.out(f"      alignment vs {primary.id}: first visible bar #{a['first_visible_primary_index']}, "
                         f"coverage {a['coverage_pct']}%, aux ends before primary: {a['aux_ends_before_primary']}")
        return 0

    def cmd_export(self, args: List[str]) -> int:
        ds = _flag(args, "--dataset")
        if not ds:
            raise ValueError("--dataset <id> is required")
        prim_id = series_id(self.config["market"]["symbol"], self.config["market"]["timeframe"])
        wanted = _multi(args, "--series")
        primary = self._find_series(prim_id)
        if primary is None:
            raise DatasetError(f"primary series {prim_id} (market.symbol/timeframe) is not downloaded")
        aux_ids = wanted or [series_id(c["symbol"], c["timeframe"]) for c in self.series_cfg
                             if series_id(c["symbol"], c["timeframe"]) != prim_id]
        aux, names, validation = [], {}, {}
        regime_sid = series_id(self.config.get("usdtd", {}).get("symbol", "USDT.D"), self.config.get("usdtd", {}).get("timeframe", "4h"))
        for sid in aux_ids:
            if sid == prim_id:
                continue
            st = self._find_series(sid)
            if st is None:
                if wanted:
                    raise DatasetError(f"series {sid} is not downloaded")
                self.out(f"  note            : {sid} not downloaded - skipped")
                continue
            aux.append(st)
            names[sid] = "usdtd" if sid == regime_sid else sid
        for st in [primary] + aux:
            validation[st.id] = (st.meta() or {}).get("validation_status", "unknown")
        compress = self.compress and "--no-compress" not in args
        m = export_dataset(self.hroot, ds, primary, aux, start=_flag(args, "--from"), end=_flag(args, "--to"),
                           compress=compress, description=_flag(args, "--description", ""), aux_names=names,
                           validation=validation, overwrite="--overwrite" in args)
        out_dir = self.hroot / "datasets" / ds
        self.out(f"  dataset         : {out_dir}")
        self.out(f"  primary         : {m['primary']['file']}  {m['primary']['rows']:,} rows  {m['primary']['first_open']} -> {m['primary']['last_open']}")
        for a in m["auxiliary"]:
            self.out(f"  auxiliary       : {a['name']:<8} {a['file']}  {a['rows']:,} rows")
        self.out(f"  dataset_sha256  : {m['dataset_sha256']}")
        try:
            rel = out_dir.relative_to(self.root)
        except ValueError:
            rel = out_dir
        self.out(f"  use it          : set data.directory: {rel}/ in config/config.yaml (config.yaml was NOT modified)")
        return 0

    def _report(self, res, store: SeriesStore) -> int:
        if res.dry_run:
            self.out(f"  DRY RUN         : {json.dumps(res.meta.get('plan'), default=str)}  (nothing fetched or written)")
            return 0
        if res.stopped_reason:
            self.out(f"  STOPPED         : {res.stopped_reason}")
        self.out(f"  fetched         : {res.rows_fetched:,} rows in {res.pages} pages  (forming candle dropped: {res.dropped_forming})")
        self.out(f"  series          : {store.csv_path}  rows {res.rows_total:,}  added {res.rows_added:,}  {res.first_open} -> {res.last_open}")
        if res.validation is not None:
            self.out("  validation      : " + res.validation.summary_line())
            for i in res.validation.issues:
                self.out(f"      {i.code} {i.severity:<7} x{i.count:<5} {i.message}")
        return 0 if res.ok else 1
