"""Deterministic paging downloader: source -> validate -> SeriesStore (working copy).

Rules
- Pages advance by `last_open + tf`; a page is never skipped silently - a failing source aborts the run
  after the retries inside the source, and everything fetched so far is still written (resumable).
- The last candle is dropped when its close time is after `now` (still forming) - the same
  point-in-time rule the Step 8 aux feeds use.
- `update()` re-pulls `overlap_bars` before the stored last row; if the overlap disagrees with what is
  on disk beyond `revision_tolerance_pct` the update STOPS and reports instead of rewriting history.
- Gaps are never filled. Duplicates inside a fetch are resolved last-wins (exchange re-sends).
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.data.base import OHLCV_COLUMNS
from src.data.dataset import SeriesStore, columns_for, normalise
from src.data.fetch.source import OHLCVSource, SourceError
from src.data.timeframes import from_ms, iso, timeframe_to_ms, to_ms
from src.data.validate import ValidationConfig, ValidationReport, validate_frame


@dataclass
class FetchConfig:
    page_limit: Optional[int] = None      # None -> source max_page
    overlap_bars: int = 3
    max_retries: int = 5
    backoff_seconds: float = 2.0
    rate_limit: bool = True
    revision_tolerance_pct: float = 1e-6  # allowed |delta| % on overlapping rows before an update refuses
                                          # (default only absorbs CSV float formatting, not real revisions)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FetchConfig":
        f = (config.get("history", {}) or {}).get("fetch", {}) or {}
        return cls(f.get("page_limit"), int(f.get("overlap_bars", 3)), int(f.get("max_retries", 5)),
                   float(f.get("backoff_seconds", 2)), bool(f.get("rate_limit", True)),
                   float(f.get("revision_tolerance_pct", 1e-6)))

    def as_dict(self) -> Dict[str, Any]:
        return {"page_limit": self.page_limit, "overlap_bars": self.overlap_bars, "max_retries": self.max_retries,
                "backoff_seconds": self.backoff_seconds, "rate_limit": self.rate_limit}


@dataclass
class DownloadResult:
    series_id: str
    rows_fetched: int = 0
    rows_added: int = 0
    rows_total: int = 0
    pages: int = 0
    first_open: Optional[str] = None
    last_open: Optional[str] = None
    dropped_forming: int = 0
    dry_run: bool = False
    stopped_reason: Optional[str] = None
    validation: Optional[ValidationReport] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stopped_reason is None and (self.validation is None or self.validation.ok)


class DownloadError(Exception):
    pass


class Downloader:
    def __init__(self, source: OHLCVSource, store: SeriesStore, fetch_cfg: Optional[FetchConfig] = None,
                 validation_cfg: Optional[ValidationConfig] = None, now: Optional[pd.Timestamp] = None,
                 progress: Optional[Callable[[str], None]] = None):
        if source.kind != store.kind:
            raise DownloadError(f"source kind {source.kind!r} does not match series kind {store.kind!r}")
        self.source, self.store = source, store
        self.fcfg = fetch_cfg or FetchConfig()
        self.vcfg = validation_cfg or ValidationConfig()
        self.now = pd.Timestamp(now) if now is not None else pd.Timestamp(dt.datetime.now(dt.timezone.utc))
        self.now = self.now.tz_localize("UTC") if self.now.tzinfo is None else self.now.tz_convert("UTC")
        self.progress = progress or (lambda msg: None)

    # ------------------------------------------------------------------ paging
    def fetch_range(self, since_ms: int, until_ms: Optional[int] = None) -> (pd.DataFrame, int):
        tf_ms = timeframe_to_ms(self.store.timeframe)
        limit = int(self.fcfg.page_limit or self.source.max_page)
        until_ms = until_ms if until_ms is not None else to_ms(self.now)
        rows: List[List[float]] = []
        pages = 0
        cursor = int(since_ms)
        while cursor <= until_ms:
            page = self.source.fetch(self.store.symbol, self.store.timeframe, cursor, limit)
            pages += 1
            if not page:
                break
            page = [r for r in page if r[0] >= cursor]          # sources may return the row before `since`
            if not page:
                break
            rows.extend(r for r in page if r[0] <= until_ms)
            last = int(page[-1][0])
            self.progress(f"  page {pages}: {len(page)} rows up to {iso(from_ms(last))}")
            if last + tf_ms <= cursor:                           # no forward progress -> stop, never loop
                break
            cursor = last + tf_ms
            if len(page) < limit:                                # short page == end of available data
                break
        cols = columns_for(self.store.kind)
        df = pd.DataFrame(rows, columns=["timestamp"] + cols[1:]) if rows else pd.DataFrame(columns=cols)
        if len(df):
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
            df = normalise(df, self.store.kind)
        return df, pages

    def _drop_forming(self, df: pd.DataFrame) -> (pd.DataFrame, int):
        if df.empty:
            return df, 0
        close_time = df["timestamp"] + pd.Timedelta(timeframe_to_ms(self.store.timeframe), unit="ms")
        keep = close_time <= self.now
        return df[keep].reset_index(drop=True), int((~keep).sum())

    # ------------------------------------------------------------------ commands
    def download(self, start, end=None, dry_run: bool = False) -> DownloadResult:
        """Full fetch of [start, end] into the series store (replaces the working copy)."""
        res = DownloadResult(self.store.id, dry_run=dry_run)
        since, until = to_ms(start), (to_ms(end) if end else None)
        if dry_run:
            res.stopped_reason = None
            res.meta = {"plan": {"since": iso(from_ms(since)), "until": iso(from_ms(until)) if until else iso(self.now),
                                 "page_limit": int(self.fcfg.page_limit or self.source.max_page),
                                 "source": self.source.describe()}}
            return res
        df, res.pages = self.fetch_range(since, until)
        res.rows_fetched = int(len(df))
        df, res.dropped_forming = self._drop_forming(df)
        if df.empty:
            res.stopped_reason = "source returned no closed candles for the requested window"
            return res
        return self._commit(df, res, {"action": "download", "since": iso(from_ms(since)),
                                      "until": iso(from_ms(until)) if until else None})

    def update(self, dry_run: bool = False) -> DownloadResult:
        """Extend the working copy from its last row (minus overlap); refuse if the overlap disagrees."""
        res = DownloadResult(self.store.id, dry_run=dry_run)
        if not self.store.exists():
            raise DownloadError(f"series {self.store.id} has not been downloaded yet - run `data download` first")
        existing = self.store.load()
        tf_ms = timeframe_to_ms(self.store.timeframe)
        overlap = min(self.fcfg.overlap_bars, len(existing))
        since_ts = existing["timestamp"].iloc[-overlap] if overlap else existing["timestamp"].iloc[-1] + pd.Timedelta(tf_ms, unit="ms")
        since = to_ms(since_ts)
        if dry_run:
            res.meta = {"plan": {"since": iso(from_ms(since)), "until": iso(self.now), "overlap_bars": overlap}}
            res.rows_total = int(len(existing))
            return res
        new, res.pages = self.fetch_range(since, None)
        res.rows_fetched = int(len(new))
        new, res.dropped_forming = self._drop_forming(new)
        if new.empty:
            res.rows_total = int(len(existing))
            res.first_open, res.last_open = iso(existing["timestamp"].iloc[0]), iso(existing["timestamp"].iloc[-1])
            res.meta = {"note": "nothing new"}
            return res
        # overlap check
        ov = existing.merge(new, on="timestamp", suffixes=("_old", "_new"))
        if len(ov):
            cols = columns_for(self.store.kind)[1:]
            for c in cols:
                delta = ((ov[f"{c}_new"] - ov[f"{c}_old"]).abs() / ov[f"{c}_old"].abs().replace(0, 1)) * 100.0
                if (delta > self.fcfg.revision_tolerance_pct).any():
                    worst = ov.loc[delta.idxmax(), "timestamp"]
                    res.stopped_reason = (f"overlap disagreement on {c} at {iso(worst)} ({delta.max():.6g}% > "
                                          f"{self.fcfg.revision_tolerance_pct}%); history NOT rewritten - "
                                          f"re-download the series if the source revised it")
                    res.rows_total = int(len(existing))
                    return res
        merged = normalise(pd.concat([existing, new], ignore_index=True), self.store.kind)
        res.rows_added = int(len(merged) - len(existing))
        if res.rows_added == 0:                                   # only the overlap came back: leave the file alone
            res.rows_total = int(len(existing))
            res.first_open, res.last_open = iso(existing["timestamp"].iloc[0]), iso(existing["timestamp"].iloc[-1])
            res.meta = {"note": "nothing new"}
            return res
        return self._commit(merged, res, {"action": "update", "since": iso(from_ms(since)), "overlap_bars": overlap})

    def _commit(self, df: pd.DataFrame, res: DownloadResult, entry: Dict[str, Any]) -> DownloadResult:
        report = validate_frame(df, self.store.id, self.store.timeframe, self.store.kind, self.vcfg, now=self.now)
        res.validation = report
        entry.update({"utc": iso(self.now), "rows_fetched": res.rows_fetched, "pages": res.pages,
                      "dropped_forming": res.dropped_forming, "source": self.source.describe(),
                      "validation_status": report.status})
        res.meta = self.store.save(df, fetch_entry=entry, extra={"fetch": self.fcfg.as_dict()})
        self.store.write_validation(report)
        res.rows_total = int(len(df))
        if entry.get("action") == "download":
            res.rows_added = res.rows_total
        res.first_open, res.last_open = res.meta["first_open"], res.meta["last_open"]
        return res
