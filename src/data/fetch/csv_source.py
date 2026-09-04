"""Offline source that serves rows from local CSV / CSV.GZ files or official Binance Vision ZIP archives.

Used to import third-party dumps (TradingView / exchange archives) - the USDT.D 4h close series in
particular - through exactly the same Downloader -> validate -> SeriesStore path as exchange data,
and by tests as a deterministic "fake exchange". Never touches the network.

Accepted `file:` targets
    file:/path/series.csv            canonical header  timestamp,<open,high,low,close,volume | close>
    file:/path/series.csv.gz         same, gzip
    file:/path/BTCUSDT-15m-2024-01.zip    official Binance Vision spot kline archive (one CSV member)
    file:/path/BTCUSDT-15m-2024-01.csv    the same CSV outside its zip
    file:/path/archive_dir/          every *.zip / *.csv / *.csv.gz in the directory, sorted by name
    file:/path/BTCUSDT-15m-*.zip     glob pattern (quoted in the shell)

Binance Vision kline layout (12 columns, header optional):
    open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades,
    taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
Only open_time -> timestamp (UTC) and open/high/low/close/volume are kept; the other columns are
discarded. open_time is epoch milliseconds (microseconds in archives from 2025 onwards) and is
converted by magnitude, never by file name. Values are passed through unchanged.

Combining several files: identical duplicate rows (adjacent monthly archives) are de-duplicated;
duplicates with different values are a CONFLICT and abort the import. Nothing is filled or repaired -
gaps, unsorted rows, OHLC violations and forming candles are left to the Step 10 validation/downloader.
"""

import glob
import io
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.dataset import columns_for, normalise
from src.data.fetch.source import OHLCVSource, Row, SourceError
from src.data.timeframes import series_to_ms, to_ms

BINANCE_KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
                         "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
_MAX_ZIP_MEMBER_BYTES = 512 * 1024 * 1024       # a monthly 1m kline file is ~60 MB; anything larger is not ours
_SUFFIXES = (".csv", ".csv.gz", ".zip")


# ------------------------------------------------------------------ readers
def _is_binance_layout(df: pd.DataFrame) -> bool:
    return df.shape[1] == len(BINANCE_KLINE_COLUMNS)


def binance_kline_to_canonical(raw: pd.DataFrame, kind: str = "ohlcv") -> pd.DataFrame:
    """12-column Binance kline frame (header-less, header row already stripped) -> canonical columns."""
    if not _is_binance_layout(raw):
        raise SourceError(f"expected {len(BINANCE_KLINE_COLUMNS)} Binance kline columns, got {raw.shape[1]}")
    raw = raw.copy()
    raw.columns = BINANCE_KLINE_COLUMNS
    ts = pd.to_numeric(raw["open_time"], errors="coerce")
    if ts.isna().any() or not np.isfinite(ts.to_numpy(dtype=float)).all():
        bad = raw["open_time"][ts.isna()].head(3).tolist()
        raise SourceError(f"non-numeric Binance open_time value(s): {bad}")
    ts = ts.astype("int64")
    # epoch magnitude: >= 1e14 -> microseconds (Binance Vision 2025+), else milliseconds
    ts = np.where(ts >= 10 ** 14, ts // 1000, ts)
    if (ts < 10 ** 11).any():
        raise SourceError("Binance open_time must be epoch milliseconds/microseconds")
    out = pd.DataFrame({"timestamp": pd.to_datetime(ts.astype("int64"), unit="ms", utc=True)})
    for c in columns_for(kind)[1:]:
        out[c] = pd.to_numeric(raw[c], errors="raise").astype(float)
    return out


def _read_csv_any(handle_or_path) -> pd.DataFrame:
    """Read a CSV whose header may be absent; returns a frame with either real header or integer columns."""
    df = pd.read_csv(handle_or_path, header=None, dtype=str, keep_default_na=False)
    if df.empty:
        raise SourceError("empty import file")
    first = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    has_header = not first[0].lstrip("-").replace(".", "", 1).isdigit()
    if has_header:
        df.columns = first
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _to_canonical(df: pd.DataFrame, kind: str, name: str) -> pd.DataFrame:
    """Frame from any accepted layout -> canonical frame (NOT yet sorted/de-duplicated)."""
    try:
        if _is_binance_layout(df) and (all(isinstance(c, (int, np.integer)) for c in df.columns)
                                       or list(df.columns)[:2] == ["open_time", "open"]):
            return binance_kline_to_canonical(df, kind)
        if df.shape[1] < 2 or all(isinstance(c, (int, np.integer)) for c in df.columns):
            raise SourceError(f"unrecognised layout with {df.shape[1]} header-less columns "
                              f"(expected canonical header or the 12-column Binance kline layout)")
        cols = columns_for(kind)
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise SourceError(f"missing column(s) {missing}")
        out = out[cols]
        ts = pd.to_numeric(out["timestamp"], errors="coerce")
        if ts.notna().all():
            out["timestamp"] = pd.to_datetime(ts.astype("int64"), unit="ms" if ts.iloc[0] > 1e11 else "s", utc=True)
        else:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        for c in cols[1:]:
            out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
        return out
    except SourceError as exc:
        raise SourceError(f"{name}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"{name}: cannot parse as {kind} series: {exc}") from exc


def _zip_member(zf: zipfile.ZipFile, archive: Path) -> zipfile.ZipInfo:
    """The single CSV member of a Binance Vision archive; refuses anything unexpected (no extraction)."""
    members = [m for m in zf.infolist() if not m.is_dir()]
    csvs = []
    for m in members:
        n = m.filename
        if not n.lower().endswith(".csv"):
            continue
        p = Path(n)
        if p.is_absolute() or ".." in p.parts or "/" in n or "\\" in n or n.startswith(("/", "\\")):
            raise SourceError(f"{archive.name}: unsafe zip member path {n!r}")
        csvs.append(m)
    if len(csvs) != 1:
        raise SourceError(f"{archive.name}: expected exactly one .csv member, found {len(csvs)} "
                          f"({[m.filename for m in members][:5]})")
    m = csvs[0]
    if m.file_size > _MAX_ZIP_MEMBER_BYTES:
        raise SourceError(f"{archive.name}: member {m.filename} is {m.file_size} bytes, above the import limit")
    return m


def read_import_file(path: Path, kind: str = "ohlcv") -> pd.DataFrame:
    """One CSV / CSV.GZ / ZIP file -> canonical frame (unsorted, duplicates kept)."""
    if not path.exists():
        raise SourceError(f"import file not found: {path}")
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                m = _zip_member(zf, path)
                with zf.open(m, "r") as fh:                     # streamed, read-only, never extracted to disk
                    df = _read_csv_any(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
        except zipfile.BadZipFile as exc:
            raise SourceError(f"{path.name}: not a valid zip archive") from exc
        return _to_canonical(df, kind, f"{path.name}:{m.filename}")
    return _to_canonical(_read_csv_any(path), kind, path.name)


def expand_target(target: str) -> List[Path]:
    """file: target -> ordered list of import files (single file, directory or glob)."""
    p = Path(target)
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.is_file() and q.name.lower().endswith(_SUFFIXES))
        if not files:
            raise SourceError(f"no .csv/.csv.gz/.zip files in {p}")
        return files
    if any(ch in target for ch in "*?["):
        files = sorted(Path(q) for q in glob.glob(target))
        if not files:
            raise SourceError(f"no files match {target}")
        return files
    return [p]


def combine_frames(frames: List[pd.DataFrame], kind: str) -> pd.DataFrame:
    """Concatenate imports; identical duplicate timestamps collapse, conflicting ones abort."""
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    dup = df[df.duplicated("timestamp", keep=False)]
    if len(dup):
        cols = columns_for(kind)[1:]
        for ts, grp in dup.groupby("timestamp", sort=True):
            if (grp[cols].nunique(dropna=False) > 1).any():
                raise SourceError(f"conflicting duplicate candle at {ts.isoformat()}: "
                                  f"{grp[cols].drop_duplicates().to_dict('records')[:2]}")
    return normalise(df, kind)


# ------------------------------------------------------------------ source
class LocalFileSource(OHLCVSource):
    online = False

    def __init__(self, path, kind: str = "ohlcv", max_page: int = 1000, frame: Optional[pd.DataFrame] = None):
        self.path = Path(path) if path else None
        self.id = f"file:{self.path.name}" if self.path else "file:<frame>"
        self.kind = kind
        self.max_page = max_page
        self.files: List[Path] = []
        if frame is None:
            if self.path is None:
                raise SourceError("import file not found: None")
            self.files = expand_target(str(path))
            frames = [read_import_file(f, kind) for f in self.files]
            df = combine_frames(frames, kind)
            if len(self.files) > 1:
                self.id = f"file:{self.files[0].name}..{self.files[-1].name}({len(self.files)} files)"
        else:
            try:
                df = normalise(frame, kind)
            except Exception as exc:  # noqa: BLE001
                raise SourceError(f"{self.id}: cannot parse as {kind} series: {exc}") from exc
        self._ms = series_to_ms(df["timestamp"]).to_numpy()
        self._values = df.drop(columns=["timestamp"]).to_numpy()

    def fetch(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> List[Row]:
        start = int(self._ms.searchsorted(int(since_ms), side="left"))
        stop = start + min(int(limit), self.max_page)
        return [[int(t)] + [float(x) for x in v] for t, v in zip(self._ms[start:stop], self._values[start:stop])]

    @classmethod
    def from_frame(cls, df: pd.DataFrame, kind: str = "ohlcv", max_page: int = 1000, name: str = "frame") -> "LocalFileSource":
        src = cls(None, kind=kind, max_page=max_page, frame=df)
        src.id = f"file:{name}"
        return src

    def first_ms(self) -> Optional[int]:
        return int(self._ms[0]) if len(self._ms) else None

    @staticmethod
    def ms(ts) -> int:
        return to_ms(ts)
