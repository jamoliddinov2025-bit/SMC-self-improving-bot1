"""Historical series and frozen datasets on local disk (CSV canonical, optional .csv.gz).

Layout (root = history.root, default data/history/):
    <root>/<source>/<SERIES>/<SERIES>.csv        mutable working copy, extended by `data update`
    <root>/<source>/<SERIES>/series.json         per-series metadata + fetch log + sha256
    <root>/<source>/<SERIES>/validation.json|md  last validation report
    <root>/datasets/<dataset_id>/manifest.json   frozen, hash-pinned bundle (what config.data.directory
    <root>/datasets/<dataset_id>/<SERIES>.csv[.gz]  points at for real-data backtests / paper / improve)

SERIES is exactly the file stem the existing loaders expect: CSVMarketData.filename_for()
("BTCUSDT_15m") and aux_filename() ("USDTD_4h"), so nothing downstream changes.
"""

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data.base import OHLCV_COLUMNS
from src.data.csv_provider import CSVMarketData
from src.data.timeframes import iso
from src.data.validate import CLOSE_COLUMNS

SERIES_SCHEMA = 1
MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
SERIES_META = "series.json"


# ---------------------------------------------------------------- naming
def series_id(symbol: str, timeframe: str) -> str:
    """BTC/USDT 15m -> BTCUSDT_15m ; USDT.D 4h -> USDTD_4h (same rule as the existing loaders)."""
    return f"{symbol.replace('/', '').replace('.', '')}_{timeframe}"


def columns_for(kind: str) -> List[str]:
    return OHLCV_COLUMNS if kind == "ohlcv" else CLOSE_COLUMNS


# ---------------------------------------------------------------- hashing / io
def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Content hash independent of file compression: canonical CSV text of the frame."""
    return hashlib.sha256(frame_to_csv_text(df).encode("utf-8")).hexdigest()


def frame_to_csv_text(df: pd.DataFrame) -> str:
    out = df.copy()
    out["timestamp"] = [iso(t) for t in pd.to_datetime(out["timestamp"], utc=True)]
    # %.17g round-trips every IEEE double exactly: the stored file reproduces the source values bit-for-bit
    return out.to_csv(index=False, lineterminator="\n", float_format="%.17g")


def write_csv_atomic(df: pd.DataFrame, path: Union[str, Path]) -> str:
    """Write canonical CSV (gzip if the suffix is .gz) via tmp + os.replace. Returns the file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame_to_csv_text(df).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=".series-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            if path.suffix == ".gz":
                # mtime=0 + no filename -> byte-identical gzip output for identical content (reproducible hashes)
                with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                    gz.write(text)
            else:
                f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return sha256_file(path)


def read_series_csv(path: Union[str, Path], kind: str = "ohlcv") -> pd.DataFrame:
    """Read a canonical series file (plain or gzip) WITHOUT cleaning (validation sees the raw rows)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df[columns_for(kind)] if all(c in df.columns for c in columns_for(kind)) else df


def normalise(df: pd.DataFrame, kind: str = "ohlcv") -> pd.DataFrame:
    """Canonical in-memory form: UTC timestamps, float columns, sorted, de-duplicated (last wins)."""
    cols = columns_for(kind)
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out[cols]
    ts = out["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        out["timestamp"] = pd.to_datetime(ts.astype("int64"), unit="ms" if ts.iloc[0] > 1e11 else "s", utc=True)
    else:
        out["timestamp"] = pd.to_datetime(ts, utc=True)
    for c in cols[1:]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".meta-", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- series (working copy)
class SeriesStore:
    """One historical series directory: <root>/<source>/<SERIES>/."""

    def __init__(self, root: Union[str, Path], source: str, symbol: str, timeframe: str, kind: str = "ohlcv"):
        self.root, self.source, self.symbol, self.timeframe, self.kind = Path(root), source, symbol, timeframe, kind
        self.id = series_id(symbol, timeframe)
        self.dir = self.root / source / self.id
        self.csv_path = self.dir / f"{self.id}.csv"
        self.meta_path = self.dir / SERIES_META

    def exists(self) -> bool:
        return self.csv_path.exists()

    def meta(self) -> Optional[Dict[str, Any]]:
        return _read_json(self.meta_path)

    def load_raw(self) -> pd.DataFrame:
        return read_series_csv(self.csv_path, self.kind)

    def load(self) -> pd.DataFrame:
        return normalise(self.load_raw(), self.kind)

    def save(self, df: pd.DataFrame, fetch_entry: Optional[Dict[str, Any]] = None,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = normalise(df, self.kind)
        sha = write_csv_atomic(df, self.csv_path)
        meta = self.meta() or {"schema_version": SERIES_SCHEMA, "fetch_log": []}
        meta.update({"schema_version": SERIES_SCHEMA, "series_id": self.id, "source": self.source,
                     "symbol": self.symbol, "timeframe": self.timeframe, "kind": self.kind,
                     "file": self.csv_path.name, "rows": int(len(df)), "sha256": sha,
                     "content_sha256": sha256_frame(df),
                     "first_open": iso(df["timestamp"].iloc[0]) if len(df) else None,
                     "last_open": iso(df["timestamp"].iloc[-1]) if len(df) else None})
        if fetch_entry:
            meta.setdefault("fetch_log", []).append(fetch_entry)
        if extra:
            meta.update(extra)
        _write_json(self.meta_path, meta)
        return meta

    def verify_hash(self) -> bool:
        m = self.meta()
        return bool(m) and self.exists() and sha256_file(self.csv_path) == m.get("sha256")

    def write_validation(self, report) -> None:
        _write_json(self.dir / "validation.json", report.to_dict())
        (self.dir / "validation.md").write_text(report.to_markdown(), encoding="utf-8")
        meta = self.meta()
        if meta is not None:
            meta["validation_status"] = report.status
            _write_json(self.meta_path, meta)


# ---------------------------------------------------------------- datasets (frozen bundles)
class DatasetError(Exception):
    pass


def dataset_hash(files: Dict[str, str]) -> str:
    """One number for the whole dataset: sha256 over sorted (filename, sha256) pairs."""
    blob = "\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_manifest(directory: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Manifest of a dataset directory, or None when `directory` is a plain CSV folder (e.g. data/sample)."""
    return _read_json(Path(directory) / MANIFEST_NAME)


def verify_dataset(directory: Union[str, Path]) -> List[str]:
    """Return a list of problems (empty == every file matches its manifest hash)."""
    d = Path(directory)
    m = load_manifest(d)
    if m is None:
        return [f"{d / MANIFEST_NAME} not found"]
    problems = []
    files = {}
    for entry in [m["primary"]] + list(m.get("auxiliary", [])):
        p = d / entry["file"]
        if not p.exists():
            problems.append(f"missing file {entry['file']}")
            continue
        actual = sha256_file(p)
        files[entry["file"]] = actual
        if actual != entry["sha256"]:
            problems.append(f"V10 hash mismatch for {entry['file']}: manifest {entry['sha256'][:12]} != file {actual[:12]}")
    if not problems and dataset_hash(files) != m.get("dataset_sha256"):
        problems.append("V10 dataset_sha256 does not match the files")
    return problems


def dataset_identity(directory: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """{'dataset_id', 'dataset_sha256', 'synthetic'} for a dataset dir; None for plain folders."""
    m = load_manifest(directory)
    if m is None:
        return None
    return {"dataset_id": m.get("dataset_id"), "dataset_sha256": m.get("dataset_sha256"),
            "synthetic": bool(m.get("synthetic", False))}


def export_dataset(root: Union[str, Path], dataset_id: str, primary: SeriesStore, auxiliary: List[SeriesStore],
                   start: Optional[str] = None, end: Optional[str] = None, compress: bool = True,
                   description: str = "", aux_names: Optional[Dict[str, str]] = None,
                   validation: Optional[Dict[str, str]] = None, created_utc: Optional[str] = None,
                   overwrite: bool = False, synthetic: bool = False) -> Dict[str, Any]:
    """Freeze series (optionally sliced to [start, end]) into <root>/datasets/<dataset_id>/ with a manifest.
    `synthetic=True` labels fixture/generated data so no consumer can present it as real history."""
    out = Path(root) / "datasets" / dataset_id
    if out.exists() and not overwrite:
        raise DatasetError(f"dataset {dataset_id!r} already exists at {out} (use --overwrite)")
    tmp = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=str(out.parent if out.parent.exists() else Path(root))))
    try:
        files: Dict[str, str] = {}

        def _freeze(store: SeriesStore) -> Dict[str, Any]:
            if not store.exists():
                raise DatasetError(f"series {store.id} not downloaded ({store.csv_path})")
            df = store.load()
            if start:
                df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
            df = df.reset_index(drop=True)
            if df.empty:
                raise DatasetError(f"series {store.id} has no rows in the requested window")
            fname = f"{store.id}.csv" + (".gz" if compress else "")
            sha = write_csv_atomic(df, tmp / fname)
            files[fname] = sha
            return {"symbol": store.symbol, "timeframe": store.timeframe, "kind": store.kind, "file": fname,
                    "sha256": sha, "content_sha256": sha256_frame(df), "rows": int(len(df)),
                    "first_open": iso(df["timestamp"].iloc[0]), "last_open": iso(df["timestamp"].iloc[-1]),
                    "source": store.source, "series_sha256_at_export": (store.meta() or {}).get("sha256")}

        manifest = {
            "schema_version": MANIFEST_SCHEMA, "dataset_id": dataset_id, "description": description,
            "created_utc": created_utc or pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window": {"start": start, "end": end}, "compressed": compress, "synthetic": bool(synthetic),
            "primary": _freeze(primary),
            "auxiliary": [dict(_freeze(a), name=(aux_names or {}).get(a.id, a.id)) for a in auxiliary],
            "validation": validation or {},
        }
        manifest["dataset_sha256"] = dataset_hash(files)
        _write_json(tmp / MANIFEST_NAME, manifest)
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


# ---------------------------------------------------------------- provider with hash verification
class DatasetMarketData(CSVMarketData):
    """CSVMarketData that refuses to serve a dataset directory whose files do not match manifest.json.
    Plain folders without a manifest (e.g. data/sample/) behave exactly like CSVMarketData."""

    def __init__(self, directory: Union[str, Path], verify: bool = True):
        super().__init__(directory)
        self.manifest = load_manifest(directory)
        if verify and self.manifest is not None:
            problems = verify_dataset(directory)
            if problems:
                raise DatasetError(f"dataset at {directory} failed verification: " + "; ".join(problems))

    @property
    def identity(self) -> Optional[Dict[str, Any]]:
        return dataset_identity(self.directory)
