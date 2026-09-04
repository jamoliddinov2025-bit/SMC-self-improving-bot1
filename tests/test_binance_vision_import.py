"""Step 12A - importing official Binance Vision spot kline archives through the existing `file:` source.
Deterministic fixtures only; nothing is downloaded and nothing is fabricated as real data."""

import copy
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.data.cli import DataCLI  # noqa: E402
from src.data.dataset import SeriesStore, load_manifest, normalise  # noqa: E402
from src.data.fetch.csv_source import (BINANCE_KLINE_COLUMNS, LocalFileSource, binance_kline_to_canonical,  # noqa: E402
                                       combine_frames, read_import_file)
from src.data.fetch.downloader import Downloader  # noqa: E402
from src.data.fetch.source import SourceError, make_source  # noqa: E402
from src.data.validate import ValidationConfig, validate_frame  # noqa: E402
from src.main import load_config  # noqa: E402

TF_MS = 15 * 60 * 1000
JAN = pd.Timestamp("2024-01-01", tz="UTC")


def kline_rows(start: pd.Timestamp, n: int, base: float = 42000.0, unit: str = "ms"):
    """Deterministic 12-column Binance-style rows (open_time first, close_time = open + tf - 1)."""
    t0 = int(start.value // 1_000_000)
    rows = []
    for i in range(n):
        o = base + i * 1.5
        ot = t0 + i * TF_MS
        mult = 1000 if unit == "us" else 1
        rows.append([ot * mult, f"{o:.2f}", f"{o + 10:.2f}", f"{o - 10:.2f}", f"{o + 2.25:.2f}", f"{100 + i:.5f}",
                     (ot + TF_MS - 1) * mult, "1.0", 10 + i, "0.5", "0.25", "0"])
    return rows


def write_csv(path: Path, rows, header=False):
    df = pd.DataFrame(rows, columns=BINANCE_KLINE_COLUMNS)
    df.to_csv(path, index=False, header=header)
    return path


def write_zip(path: Path, rows, member=None, header=False, extra=None):
    member = member or path.name.replace(".zip", ".csv")
    buf = io.StringIO()
    pd.DataFrame(rows, columns=BINANCE_KLINE_COLUMNS).to_csv(buf, index=False, header=header)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, buf.getvalue())
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return path


def expected_frame(rows):
    return pd.DataFrame({"timestamp": pd.to_datetime([r[0] if r[0] < 10 ** 14 else r[0] // 1000 for r in rows], unit="ms", utc=True),
                         "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                         "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                         "volume": [float(r[5]) for r in rows]})


# ------------------------------------------------------------------ layout mapping
def test_valid_binance_row_maps_to_canonical_exactly(tmp_path):
    rows = kline_rows(JAN, 8)
    for header in (False, True):
        df = read_import_file(write_csv(tmp_path / f"h{header}.csv", rows, header=header))
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        pd.testing.assert_frame_equal(df, expected_frame(rows))
    assert str(df["timestamp"].dt.tz) == "UTC" and df["timestamp"].iloc[0] == JAN
    assert df["open"].iloc[3] == 42004.5 and df["volume"].iloc[7] == 107.0        # values preserved, extras dropped


def test_microsecond_open_time_is_recognised_by_magnitude(tmp_path):
    rows = kline_rows(pd.Timestamp("2025-03-01", tz="UTC"), 4, unit="us")
    df = read_import_file(write_csv(tmp_path / "us.csv", rows))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-03-01", tz="UTC")
    assert (df["timestamp"].diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_malformed_column_count_is_rejected(tmp_path):
    rows = [r[:11] for r in kline_rows(JAN, 3)]
    pd.DataFrame(rows).to_csv(tmp_path / "eleven.csv", index=False, header=False)
    with pytest.raises(SourceError, match="unrecognised layout with 11"):
        read_import_file(tmp_path / "eleven.csv")
    with pytest.raises(SourceError, match="expected 12"):
        binance_kline_to_canonical(pd.DataFrame(rows))
    rows13 = [r + ["x"] for r in kline_rows(JAN, 3)]
    pd.DataFrame(rows13).to_csv(tmp_path / "thirteen.csv", index=False, header=False)
    with pytest.raises(SourceError):
        read_import_file(tmp_path / "thirteen.csv")


def test_invalid_timestamp_is_rejected(tmp_path):
    rows = kline_rows(JAN, 3)
    rows[1][0] = "2024-01-01T00:15:00Z"
    with pytest.raises(SourceError, match="non-numeric Binance open_time"):
        read_import_file(write_csv(tmp_path / "badts.csv", rows))
    rows = kline_rows(JAN, 3)
    rows[2][0] = 1704067200                               # seconds, not ms: refuse instead of guessing
    with pytest.raises(SourceError, match="epoch milliseconds"):
        read_import_file(write_csv(tmp_path / "secs.csv", rows))


def test_non_numeric_price_is_rejected(tmp_path):
    rows = kline_rows(JAN, 3)
    rows[0][4] = "n/a"
    with pytest.raises(SourceError, match="cannot parse"):
        read_import_file(write_csv(tmp_path / "price.csv", rows))


# ------------------------------------------------------------------ duplicates / ordering / gaps (existing semantics)
def test_identical_duplicate_rows_collapse(tmp_path):
    rows = kline_rows(JAN, 5)
    src = LocalFileSource(write_csv(tmp_path / "dup.csv", rows + [list(rows[2])]))
    got = src.fetch("BTC/USDT", "15m", 0, 100)
    assert len(got) == 5 and [r[0] for r in got] == [r[0] for r in rows]


def test_conflicting_duplicate_fails(tmp_path):
    rows = kline_rows(JAN, 5)
    bad = list(rows[2]); bad[4] = "1.00"
    with pytest.raises(SourceError, match="conflicting duplicate candle at 2024-01-01T00:30:00"):
        LocalFileSource(write_csv(tmp_path / "conflict.csv", rows + [bad]))


def test_unsorted_rows_are_ordered_by_canonical_normalise(tmp_path):
    rows = kline_rows(JAN, 6)
    shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
    src = LocalFileSource(write_csv(tmp_path / "unsorted.csv", shuffled))
    ts = [r[0] for r in src.fetch("BTC/USDT", "15m", 0, 100)]
    assert ts == sorted(ts) == [r[0] for r in rows]


def test_ohlc_violation_passes_import_but_fails_step10_validation(tmp_path):
    rows = kline_rows(JAN, 30)
    rows[10][2] = "1.00"                                     # high below low
    df = read_import_file(write_csv(tmp_path / "ohlc.csv", rows))
    rep = validate_frame(normalise(df), "BTCUSDT_15m", "15m", "ohlcv", ValidationConfig(), now=pd.Timestamp("2024-02-01", tz="UTC"))
    assert not rep.ok and any(i.code == "V6" for i in rep.issues)


def test_gap_is_reported_never_filled(tmp_path):
    rows = kline_rows(JAN, 40)
    del rows[20:23]
    df = normalise(read_import_file(write_csv(tmp_path / "gap.csv", rows)))
    assert len(df) == 37
    rep = validate_frame(df, "BTCUSDT_15m", "15m", "ohlcv", ValidationConfig(max_gap_pct=0.0), now=pd.Timestamp("2024-02-01", tz="UTC"))
    assert rep.missing_bars == 3 and any(i.code == "V5" for i in rep.issues)


# ------------------------------------------------------------------ downloader integration (window, forming candle)
def _store(tmp_path):
    return SeriesStore(tmp_path / "data/history", "file", "BTC/USDT", "15m")


def test_forming_candle_dropped_by_existing_rule_regardless_of_archive_name(tmp_path):
    rows = kline_rows(JAN, 12)                               # archive claims January, "now" is inside it
    now = JAN + pd.Timedelta(minutes=15 * 10 + 5)            # candle #10 (open 02:30) is still forming
    src = LocalFileSource(write_zip(tmp_path / "BTCUSDT-15m-2024-01.zip", rows))
    res = Downloader(src, _store(tmp_path), now=now).download("2024-01-01")
    assert res.ok and res.rows_total == 10 and res.dropped_forming >= 1
    assert pd.Timestamp(res.last_open) == JAN + pd.Timedelta(minutes=15 * 9)
    assert res.validation.ok


def test_from_to_window_limits_imported_rows(tmp_path):
    rows = kline_rows(JAN, 96 * 3)                           # 3 days
    src = LocalFileSource(write_csv(tmp_path / "jan.csv", rows))
    res = Downloader(src, _store(tmp_path), now=pd.Timestamp("2024-02-01", tz="UTC")).download("2024-01-02", "2024-01-02T12:00:00Z")
    assert res.ok and res.rows_total == 49
    df = _store(tmp_path).load()
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2024-01-02T12:00:00Z")


# ------------------------------------------------------------------ monthly archives
def test_multiple_monthly_archives_combine_into_one_series(tmp_path):
    jan = kline_rows(JAN, 96 * 31)
    feb = kline_rows(pd.Timestamp("2024-02-01", tz="UTC"), 96 * 29, base=42000.0 + 96 * 31 * 1.5)
    d = tmp_path / "archives"; d.mkdir()
    write_zip(d / "BTCUSDT-15m-2024-02.zip", feb)
    write_zip(d / "BTCUSDT-15m-2024-01.zip", jan)
    (d / "notes.txt").write_text("ignored")
    src = LocalFileSource(d)                                 # directory -> sorted files
    assert [p.name for p in src.files] == ["BTCUSDT-15m-2024-01.zip", "BTCUSDT-15m-2024-02.zip"]
    res = Downloader(src, _store(tmp_path), now=pd.Timestamp("2024-04-01", tz="UTC")).download("2024-01-01")
    assert res.ok and res.rows_total == 96 * 60 and res.validation.ok and res.validation.missing_bars == 0
    glob_src = make_source(f"file:{d}/BTCUSDT-15m-*.zip")   # glob works too
    assert len(glob_src.files) == 2 and glob_src.first_ms() == src.first_ms()


def test_adjacent_archives_overlap_identical_ok_conflict_fails(tmp_path):
    jan = kline_rows(JAN, 20)
    d = tmp_path / "a"; d.mkdir()
    write_zip(d / "m1.zip", jan[:12]); write_zip(d / "m2.zip", jan[10:])       # 2 identical overlapping rows
    assert len(LocalFileSource(d).fetch("BTC/USDT", "15m", 0, 100)) == 20
    bad = [list(r) for r in jan[10:]]; bad[0][5] = "999.0"
    write_zip(d / "m2.zip", bad)
    with pytest.raises(SourceError, match="conflicting duplicate"):
        LocalFileSource(d)


# ------------------------------------------------------------------ ZIP safety
def test_zip_with_expected_csv_member_streams_without_extraction(tmp_path, monkeypatch):
    rows = kline_rows(JAN, 5)
    z = write_zip(tmp_path / "BTCUSDT-15m-2024-01.zip", rows)
    monkeypatch.setattr(zipfile.ZipFile, "extract", lambda *a, **k: pytest.fail("extract called"))
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *a, **k: pytest.fail("extractall called"))
    pd.testing.assert_frame_equal(read_import_file(z), expected_frame(rows))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["BTCUSDT-15m-2024-01.zip"]   # nothing written next to it


def test_zip_with_unsafe_or_unexpected_members_is_refused(tmp_path):
    rows = kline_rows(JAN, 5)
    with pytest.raises(SourceError, match="unsafe zip member"):
        read_import_file(write_zip(tmp_path / "trav.zip", rows, member="../../evil.csv"))
    with pytest.raises(SourceError, match="unsafe zip member"):
        read_import_file(write_zip(tmp_path / "abs.zip", rows, member="/tmp/evil.csv"))
    with pytest.raises(SourceError, match="unsafe zip member"):
        read_import_file(write_zip(tmp_path / "nested.zip", rows, member="sub/dir.csv"))
    with pytest.raises(SourceError, match="expected exactly one .csv member, found 2"):
        read_import_file(write_zip(tmp_path / "two.zip", rows, extra={"other.csv": "a,b\n1,2\n"}))
    with pytest.raises(SourceError, match="expected exactly one .csv member, found 0"):
        read_import_file(write_zip(tmp_path / "none.zip", rows, member="data.bin"))
    (tmp_path / "notzip.zip").write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(SourceError, match="not a valid zip"):
        read_import_file(tmp_path / "notzip.zip")
    # a non-csv companion (e.g. CHECKSUM) inside the archive is ignored, the csv still loads
    z = write_zip(tmp_path / "chk.zip", rows, extra={"CHECKSUM": "abc"})
    assert len(read_import_file(z)) == 5


# ------------------------------------------------------------------ determinism + full CLI path
def test_repeated_import_and_export_are_byte_identical(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["history"]["series"] = [{"symbol": "BTC/USDT", "timeframe": "15m", "kind": "ohlcv", "source": "file", "start": "2024-01-01"}]
    rows = kline_rows(JAN, 96 * 10)
    z1 = write_zip(tmp_path / "BTCUSDT-15m-2024-01.zip", rows)
    now = "2024-02-01T00:00:00Z"
    hashes = []
    for run in ("one", "two"):
        ws = tmp_path / run; ws.mkdir()
        cli = DataCLI(cfg, ws, out=lambda _: None, now=pd.Timestamp(now)) if "now" in DataCLI.__init__.__code__.co_varnames \
            else DataCLI(cfg, ws, out=lambda _: None)
        assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{z1}", "--from", "2024-01-01", "--to", "2024-01-11"]) == 0
        assert cli.run(["export", "--dataset", "bv", "--synthetic"]) == 0
        m = load_manifest(ws / "data/history/datasets/bv")
        hashes.append((m["primary"]["sha256"], m["dataset_sha256"], m["primary"]["rows"]))
    assert hashes[0] == hashes[1] and hashes[0][2] == 96 * 10
    # a copy of the same archive under a different name gives the same data hash (name/metadata not hashed)
    z2 = tmp_path / "renamed-archive.zip"; z2.write_bytes(z1.read_bytes())
    ws = tmp_path / "three"; ws.mkdir()
    cli = DataCLI(cfg, ws, out=lambda _: None)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{z2}", "--from", "2024-01-01", "--to", "2024-01-11"]) == 0
    assert cli.run(["export", "--dataset", "bv", "--synthetic"]) == 0
    m3 = load_manifest(ws / "data/history/datasets/bv")
    assert (m3["primary"]["sha256"], m3["dataset_sha256"]) == hashes[0][:2]
    csv_gz = hashlib.sha256((ws / "data/history/datasets/bv/BTCUSDT_15m.csv.gz").read_bytes()).hexdigest()
    assert csv_gz == m3["primary"]["sha256"]


# ------------------------------------------------------------------ backward compatibility
def test_canonical_csv_and_csv_gz_imports_unchanged(tmp_path):
    df = expected_frame(kline_rows(JAN, 10))
    df.to_csv(tmp_path / "c.csv", index=False)
    df.to_csv(tmp_path / "c.csv.gz", index=False, compression="gzip")
    a = LocalFileSource(tmp_path / "c.csv").fetch("BTC/USDT", "15m", 0, 100)
    b = LocalFileSource(tmp_path / "c.csv.gz").fetch("BTC/USDT", "15m", 0, 100)
    assert a == b and len(a) == 10 and a[0][1] == 42000.0
    close = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC"), "close": [4.0, 4.1, 4.2, 4.3, 4.4]})
    close.to_csv(tmp_path / "u.csv", index=False)
    u = LocalFileSource(tmp_path / "u.csv", kind="close").fetch("USDT.D", "4h", 0, 100)
    assert len(u) == 5 and u[1] == [int(close["timestamp"][1].value // 1_000_000), 4.1]
    from src.data.timeframes import series_to_ms
    epoch = df.copy(); epoch["timestamp"] = series_to_ms(epoch["timestamp"])
    epoch.to_csv(tmp_path / "e.csv", index=False)
    assert LocalFileSource(tmp_path / "e.csv").fetch("BTC/USDT", "15m", 0, 100) == a
    assert LocalFileSource.from_frame(df).fetch("BTC/USDT", "15m", 0, 100) == a
    with pytest.raises(SourceError, match="not found"):
        LocalFileSource(tmp_path / "missing.csv")
    with pytest.raises(SourceError, match="missing column"):
        pd.DataFrame({"timestamp": [1], "open": [1]}).to_csv(tmp_path / "short.csv", index=False)
        LocalFileSource(tmp_path / "short.csv")


def test_no_network_code_in_csv_source():
    import ast
    tree = ast.parse((ROOT / "src/data/fetch/csv_source.py").read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
           {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"ccxt", "requests", "urllib", "http", "socket", "aiohttp"}
