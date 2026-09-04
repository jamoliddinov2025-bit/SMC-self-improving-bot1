"""Offline data-quality validation for historical series (OHLCV or close-only auxiliary series).

Checks (codes referenced by tests, reports and the CLI):
    V1 schema / numeric parse           V6 OHLC sanity (low <= open,close <= high; volume >= 0; price > 0)
    V2 timezone / grid alignment        V7 outliers: extreme ranges, zero-volume runs, flat runs
    V3 out-of-order rows                V8 last candle must be closed (close time <= `now`)
    V4 duplicates / conflicting dupes   V9 (close series) positive values
    V5 gaps                             V10 hash mismatch (checked by dataset.py, reported through the same class)

Nothing is repaired: missing candles are never filled and problems are reported, not hidden.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.base import OHLCV_COLUMNS
from src.data.timeframes import from_ms, gap_runs, iso, off_grid_mask, timeframe_to_ms

CLOSE_COLUMNS = ["timestamp", "close"]


@dataclass
class ValidationConfig:
    max_gap_pct: float = 0.5            # percentage of expected bars missing -> error above this
    outlier_range_multiple: float = 20  # candle range > k * rolling median range -> warning
    outlier_window: int = 200
    zero_volume_run: int = 12           # consecutive zero-volume bars -> warning
    flat_run: int = 12                  # consecutive identical OHLC bars -> warning
    fail_on_warnings: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ValidationConfig":
        v = (config.get("history", {}) or {}).get("validation", {}) or {}
        return cls(float(v.get("max_gap_pct", 0.5)), float(v.get("outlier_range_multiple", 20)),
                   int(v.get("outlier_window", 200)), int(v.get("zero_volume_run", 12)),
                   int(v.get("flat_run", 12)), bool(v.get("fail_on_warnings", False)))


@dataclass
class Issue:
    code: str
    severity: str        # error | warning
    message: str
    count: int = 1
    detail: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "count": self.count,
                "detail": self.detail}


@dataclass
class ValidationReport:
    name: str
    kind: str
    timeframe: str
    rows: int = 0
    first_open: Optional[str] = None
    last_open: Optional[str] = None
    expected_rows: Optional[int] = None
    missing_bars: int = 0
    gap_runs: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: int = 0
    issues: List[Issue] = field(default_factory=list)
    fail_on_warnings: bool = False

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def status(self) -> str:
        if self.errors or (self.fail_on_warnings and self.warnings):
            return "failed"
        return "warnings" if self.warnings else "ok"

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "timeframe": self.timeframe, "status": self.status,
                "rows": self.rows, "first_open": self.first_open, "last_open": self.last_open,
                "expected_rows": self.expected_rows, "missing_bars": self.missing_bars,
                "missing_pct": round(100.0 * self.missing_bars / self.expected_rows, 4) if self.expected_rows else 0.0,
                "gap_runs": self.gap_runs, "duplicates": self.duplicates,
                "issues": [i.to_dict() for i in self.issues]}

    def summary_line(self) -> str:
        gaps = f"gaps {len(self.gap_runs)} runs / {self.missing_bars} bars"
        if self.expected_rows:
            gaps += f" ({100.0 * self.missing_bars / self.expected_rows:.3f}%)"
        return (f"{self.name}: {self.rows:,} rows, {self.first_open} -> {self.last_open}, {gaps}, "
                f"duplicates {self.duplicates}, status {self.status}")

    def to_markdown(self) -> str:
        L = [f"# Validation report - {self.name}", "", f"- status: **{self.status}**", f"- kind: {self.kind} {self.timeframe}",
             f"- rows: {self.rows:,}  ({self.first_open} -> {self.last_open})",
             f"- expected rows on grid: {self.expected_rows}  missing: {self.missing_bars}",
             f"- duplicates: {self.duplicates}", ""]
        if self.issues:
            L += ["| code | severity | count | message |", "|---|---|---|---|"]
            L += [f"| {i.code} | {i.severity} | {i.count} | {i.message} |" for i in self.issues]
        else:
            L.append("_No issues found._")
        if self.gap_runs:
            L += ["", "## Gap runs (first 50)", "", "| from | to | missing bars |", "|---|---|---|"]
            L += [f"| {g['from']} | {g['to']} | {g['missing']} |" for g in self.gap_runs[:50]]
        L += ["", "_Gaps are reported, never filled._", ""]
        return "\n".join(L)


def _parse_ts(series: pd.Series):
    """Return (parsed UTC series or None, issue or None). Naive ISO strings are assumed UTC only if they end
    with 'Z' or carry an offset; plain naive strings are rejected (timezone ambiguity)."""
    if pd.api.types.is_numeric_dtype(series):
        s = series.astype("int64")
        if len(s) and s.min() > 1e11:
            return pd.to_datetime(s, unit="ms", utc=True), None
        if len(s) and s.max() < 1e11:
            return pd.to_datetime(s, unit="s", utc=True), None
        return None, Issue("V2", "error", "mixed epoch seconds / milliseconds timestamps")
    text = series.astype(str)
    naive = ~text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if naive.any():
        return None, Issue("V2", "error", "timestamps without timezone (naive) - ambiguous, export in UTC with 'Z'",
                           int(naive.sum()), text[naive].head(3).tolist())
    try:
        return pd.to_datetime(text, utc=True), None
    except (ValueError, TypeError) as exc:
        return None, Issue("V1", "error", f"unparseable timestamps: {exc}")


def validate_frame(df: pd.DataFrame, name: str, timeframe: str, kind: str = "ohlcv",
                   cfg: Optional[ValidationConfig] = None, now: Optional[pd.Timestamp] = None) -> ValidationReport:
    """Validate a raw frame as read from CSV (no sorting / dedupe applied beforehand)."""
    cfg = cfg or ValidationConfig()
    rep = ValidationReport(name, kind, timeframe, fail_on_warnings=cfg.fail_on_warnings)
    cols = OHLCV_COLUMNS if kind == "ohlcv" else CLOSE_COLUMNS
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        rep.issues.append(Issue("V1", "error", f"missing columns {missing}"))
        return rep
    df = df[cols]
    rep.rows = int(len(df))
    if rep.rows == 0:
        rep.issues.append(Issue("V1", "error", "empty series"))
        return rep
    for c in cols[1:]:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
        except (ValueError, TypeError):
            rep.issues.append(Issue("V1", "error", f"non-numeric values in column {c!r}"))
            return rep
    if df[cols[1:]].isna().any().any():
        rep.issues.append(Issue("V1", "error", "NaN values", int(df[cols[1:]].isna().sum().sum())))

    ts, issue = _parse_ts(df["timestamp"])
    if issue is not None:
        rep.issues.append(issue)
        return rep
    df["timestamp"] = ts

    # V3 ordering, V4 duplicates
    diffs = df["timestamp"].diff().dropna()
    if (diffs < pd.Timedelta(0)).any():
        rep.issues.append(Issue("V3", "error", "out-of-order timestamps", int((diffs < pd.Timedelta(0)).sum())))
    dup_mask = df["timestamp"].duplicated(keep=False)
    if dup_mask.any():
        d = df[dup_mask]
        conflicting = int(d.groupby("timestamp").nunique().gt(1).any(axis=1).sum())
        rep.duplicates = int(df["timestamp"].duplicated().sum())
        rep.issues.append(Issue("V4", "error", f"duplicate timestamps ({conflicting} with differing values)", rep.duplicates,
                                {"conflicting": conflicting}))
    s = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # V2 grid
    og = off_grid_mask(s["timestamp"], timeframe)
    if og.any():
        rep.issues.append(Issue("V2", "error", f"timestamps not on the {timeframe} UTC grid", int(og.sum()),
                                [iso(t) for t in s.loc[og, "timestamp"].head(3)]))

    rep.first_open, rep.last_open = iso(s["timestamp"].iloc[0]), iso(s["timestamp"].iloc[-1])
    step = timeframe_to_ms(timeframe)
    span = int((s["timestamp"].iloc[-1] - s["timestamp"].iloc[0]).total_seconds() * 1000)
    rep.expected_rows = span // step + 1

    # V5 gaps
    runs = gap_runs(s["timestamp"], timeframe)
    rep.missing_bars = int(sum(r[2] for r in runs))
    rep.gap_runs = [{"from": iso(a), "to": iso(b), "missing": n} for a, b, n in runs]
    if runs:
        pct = 100.0 * rep.missing_bars / rep.expected_rows
        sev = "error" if pct > cfg.max_gap_pct else "warning"
        rep.issues.append(Issue("V5", sev, f"{len(runs)} gap runs, {rep.missing_bars} missing bars ({pct:.3f}% "
                                          f"of grid; limit {cfg.max_gap_pct}%)", len(runs)))

    if kind == "ohlcv":
        bad = ((s["low"] > s[["open", "close"]].min(axis=1)) | (s["high"] < s[["open", "close"]].max(axis=1))
               | (s["low"] > s["high"]))
        if bad.any():
            rep.issues.append(Issue("V6", "error", "OHLC violation (low/high do not bracket open/close)", int(bad.sum())))
        if (s[["open", "high", "low", "close"]] <= 0).any().any():
            rep.issues.append(Issue("V6", "error", "non-positive prices", int((s[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())))
        if (s["volume"] < 0).any():
            rep.issues.append(Issue("V6", "error", "negative volume", int((s["volume"] < 0).sum())))
        # V7
        rng = (s["high"] - s["low"])
        med = rng.rolling(cfg.outlier_window, min_periods=20).median().shift(1)
        out = (med > 0) & (rng > cfg.outlier_range_multiple * med)
        if out.any():
            rep.issues.append(Issue("V7", "warning", f"candle range > {cfg.outlier_range_multiple}x rolling median",
                                    int(out.sum()), [iso(t) for t in s.loc[out, "timestamp"].head(5)]))
        zr = _longest_run(s["volume"] == 0)
        if zr >= cfg.zero_volume_run:
            rep.issues.append(Issue("V7", "warning", f"zero-volume run of {zr} bars", zr))
        flat = (s[["open", "high", "low", "close"]].nunique(axis=1) == 1) & (s["open"].diff() == 0)
        fr = _longest_run(flat)
        if fr >= cfg.flat_run:
            rep.issues.append(Issue("V7", "warning", f"flat (identical OHLC) run of {fr} bars", fr))
    else:
        if (s["close"] <= 0).any():
            rep.issues.append(Issue("V9", "error", "non-positive close values", int((s["close"] <= 0).sum())))

    # V8 last candle closed
    if now is not None:
        close_time = s["timestamp"].iloc[-1] + pd.Timedelta(step, unit="ms")
        now_ts = pd.Timestamp(now)
        now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
        if close_time > now_ts:
            rep.issues.append(Issue("V8", "error", f"last candle {rep.last_open} is not closed yet"))
    return rep


def _longest_run(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def alignment_preview(primary_ts: pd.Series, primary_tf: str, aux_ts: pd.Series, aux_tf: str) -> Dict[str, Any]:
    """Coverage of an auxiliary series relative to the primary using the Step 8 alignment rule."""
    from src.backtesting.aux_data import align_aux_indices   # existing, unchanged rule
    idx = align_aux_indices(primary_ts, primary_tf, aux_ts, aux_tf)
    visible = idx >= 0
    first = int(visible.idxmax()) if visible.any() else None
    aux_end = pd.to_datetime(aux_ts, utc=True).iloc[-1] + pd.Timedelta(timeframe_to_ms(aux_tf), unit="ms")
    prim_end = pd.to_datetime(primary_ts, utc=True).iloc[-1] + pd.Timedelta(timeframe_to_ms(primary_tf), unit="ms")
    return {"primary_bars": int(len(primary_ts)), "first_visible_primary_index": first,
            "coverage_pct": round(100.0 * visible.mean(), 3) if len(primary_ts) else 0.0,
            "aux_ends_before_primary": bool(aux_end < prim_end),
            "aux_last_close_time": iso(aux_end), "primary_last_close_time": iso(prim_end)}
