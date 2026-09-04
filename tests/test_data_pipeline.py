"""Step 10: downloader (paging, forming candle, retry, resume/update, revision refusal, dry-run), series store,
dataset export / manifest / hash verification, .csv.gz loading, CCXT source credential refusal. Offline."""

import gzip
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_source import FakeSource, close_frame, now_after, ohlcv_frame  # noqa: E402
from src.backtesting.aux_data import build_aux_feeds  # noqa: E402
from src.data import CSVMarketData, DatasetError, DatasetMarketData, dataset_identity, verify_dataset  # noqa: E402
from src.data.dataset import SeriesStore, dataset_hash, export_dataset, sha256_file, write_csv_atomic  # noqa: E402
from src.data.fetch.ccxt_source import CCXTPublicSource  # noqa: E402
from src.data.fetch.csv_source import LocalFileSource  # noqa: E402
from src.data.fetch.downloader import DownloadError, Downloader, FetchConfig  # noqa: E402
from src.data.fetch.source import SourceError, make_source, parse_source_spec  # noqa: E402
from src.main import load_config  # noqa: E402

DF = ohlcv_frame(2000)
NOW = now_after(DF)


def _store(tmp_path, kind="ohlcv", symbol="BTC/USDT", tf="15m"):
    return SeriesStore(tmp_path / "history", "fake", symbol, tf, kind)


# ------------------------------------------------------------------ paging
def test_download_pages_exactly_without_overlap_or_loss(tmp_path):
    src = FakeSource(DF, max_page=300)
    st = _store(tmp_path)
    res = Downloader(src, st, now=NOW).download(DF["timestamp"].iloc[0])
    assert res.ok and res.rows_total == 2000 and res.rows_fetched == 2000
    assert res.pages == 7 and [c[0] for c in src.calls][:2] == [int(DF["timestamp"].iloc[0].value // 10**6),
                                                                 int(DF["timestamp"].iloc[300].value // 10**6)]
    saved = st.load()
    pd.testing.assert_frame_equal(saved, DF, check_dtype=False)
    meta = st.meta()
    assert meta["rows"] == 2000 and meta["sha256"] == sha256_file(st.csv_path) and meta["fetch_log"][0]["action"] == "download"
    assert (st.dir / "validation.json").exists() and (st.dir / "validation.md").exists()
    assert res.validation.status == "ok"


def test_download_respects_window_and_drops_forming_candle(tmp_path):
    src = FakeSource(DF, max_page=500, include_forming=True)
    st = _store(tmp_path)
    res = Downloader(src, st, now=NOW).download(DF["timestamp"].iloc[100], DF["timestamp"].iloc[699])
    assert res.rows_total == 600 and st.load()["timestamp"].iloc[0] == DF["timestamp"].iloc[100]
    # a full download whose last page carries a still-forming candle
    res = Downloader(src, st, now=NOW).download(DF["timestamp"].iloc[0])
    assert res.dropped_forming == 1 and res.rows_total == 2000


def test_source_failures_retry_inside_ccxt_source_but_downloader_aborts_on_persistent_error(tmp_path):
    src = FakeSource(DF, fail_times=1)
    with pytest.raises(SourceError):
        Downloader(src, _store(tmp_path), now=NOW).download(DF["timestamp"].iloc[0])
    assert not _store(tmp_path).exists()   # nothing half-written


def test_dry_run_writes_nothing(tmp_path):
    src = FakeSource(DF)
    st = _store(tmp_path)
    res = Downloader(src, st, now=NOW).download(DF["timestamp"].iloc[0], dry_run=True)
    assert res.dry_run and res.meta["plan"]["page_limit"] == 500 and src.calls == [] and not (tmp_path / "history").exists()


# ------------------------------------------------------------------ update
def test_update_extends_by_exactly_new_rows_and_is_idempotent(tmp_path):
    half = DF.iloc[:1200]
    st = _store(tmp_path)
    Downloader(FakeSource(half), st, now=now_after(half)).download(DF["timestamp"].iloc[0])
    res = Downloader(FakeSource(DF), st, FetchConfig(overlap_bars=3), now=NOW).update()
    assert res.ok and res.rows_added == 800 and res.rows_total == 2000
    assert res.rows_fetched == 803                                   # overlap re-pulled
    pd.testing.assert_frame_equal(st.load(), DF, check_dtype=False)
    again = Downloader(FakeSource(DF), st, now=NOW).update()
    assert again.rows_added == 0 and again.rows_total == 2000 and again.meta.get("note") == "nothing new"
    assert len(st.meta()["fetch_log"]) == 2


def test_update_refuses_when_overlap_was_revised(tmp_path):
    st = _store(tmp_path)
    Downloader(FakeSource(DF.iloc[:1000]), st, now=now_after(DF.iloc[:1000])).download(DF["timestamp"].iloc[0])
    before = st.csv_path.read_bytes()
    src = FakeSource(DF)
    src.revise(999, 1.05)                                            # last stored candle changed at the source
    res = Downloader(src, st, FetchConfig(overlap_bars=3), now=NOW).update()
    assert not res.ok and "overlap disagreement" in res.stopped_reason and "NOT rewritten" in res.stopped_reason
    assert st.csv_path.read_bytes() == before


def test_update_requires_prior_download(tmp_path):
    with pytest.raises(DownloadError):
        Downloader(FakeSource(DF), _store(tmp_path), now=NOW).update()


# ------------------------------------------------------------------ files & store
def test_atomic_write_and_gzip_are_reproducible(tmp_path):
    a = write_csv_atomic(DF, tmp_path / "a.csv.gz")
    b = write_csv_atomic(DF, tmp_path / "b.csv.gz")
    assert a == b                                                    # mtime=0 -> byte-identical gzip
    assert not list(tmp_path.glob(".series-*"))
    with gzip.open(tmp_path / "a.csv.gz", "rt") as f:
        assert f.readline().strip() == "timestamp,open,high,low,close,volume"
    pd.testing.assert_frame_equal(CSVMarketData(file_path=tmp_path / "a.csv.gz").get_ohlcv("x", "15m"), DF, check_dtype=False)


def test_csv_provider_falls_back_to_gz_and_aux_feeds_read_gz(tmp_path):
    d = tmp_path / "ds"
    write_csv_atomic(DF, d / "BTCUSDT_15m.csv.gz")
    write_csv_atomic(close_frame(), d / "USDTD_4h.csv.gz")
    p = CSVMarketData(d)
    assert p.resolve_path("BTC/USDT", "15m").name == "BTCUSDT_15m.csv.gz" and len(p.get_ohlcv("BTC/USDT", "15m")) == 2000
    cfg = load_config()
    cfg["data"]["directory"] = "ds"
    feeds = build_aux_feeds(cfg, data_root=tmp_path)
    assert feeds and len(feeds[0].frame) == 300
    # plain .csv still wins when both exist
    write_csv_atomic(DF.iloc[:10], d / "BTCUSDT_15m.csv")
    assert p.resolve_path("BTC/USDT", "15m").name == "BTCUSDT_15m.csv"


def test_local_file_source_imports_close_series_through_downloader(tmp_path):
    aux = close_frame(300)
    aux.to_csv(tmp_path / "usdtd.csv", index=False)
    src = make_source(f"file:{tmp_path / 'usdtd.csv'}", kind="close")
    assert isinstance(src, LocalFileSource) and not src.online
    st = _store(tmp_path, kind="close", symbol="USDT.D", tf="4h")
    res = Downloader(src, st, now=now_after(aux, 14_400_000)).download("2023-01-01")
    assert res.ok and res.rows_total == 300 and st.csv_path.name == "USDTD_4h.csv"
    with pytest.raises(DownloadError):
        Downloader(src, _store(tmp_path), now=NOW)                    # kind mismatch (close vs ohlcv)


# ------------------------------------------------------------------ datasets
@pytest.fixture
def series(tmp_path):
    prim = _store(tmp_path)
    Downloader(FakeSource(DF), prim, now=NOW).download(DF["timestamp"].iloc[0])
    aux = _store(tmp_path, kind="close", symbol="USDT.D", tf="4h")
    a = close_frame(300)
    Downloader(FakeSource(a, kind="close"), aux, now=now_after(a, 14_400_000)).download("2023-01-01")
    return tmp_path / "history", prim, aux


def test_export_manifest_hashes_and_verification(series):
    root, prim, aux = series
    m = export_dataset(root, "ds1", prim, [aux], aux_names={"USDTD_4h": "usdtd"}, created_utc="2025-01-01T00:00:00Z")
    d = root / "datasets" / "ds1"
    assert sorted(p.name for p in d.iterdir()) == ["BTCUSDT_15m.csv.gz", "USDTD_4h.csv.gz", "manifest.json"]
    assert m["dataset_id"] == "ds1" and m["primary"]["sha256"] == sha256_file(d / "BTCUSDT_15m.csv.gz")
    assert m["auxiliary"][0]["name"] == "usdtd" and m["synthetic"] is False
    assert m["dataset_sha256"] == dataset_hash({"BTCUSDT_15m.csv.gz": m["primary"]["sha256"],
                                                "USDTD_4h.csv.gz": m["auxiliary"][0]["sha256"]})
    assert verify_dataset(d) == []
    assert dataset_identity(d) == {"dataset_id": "ds1", "dataset_sha256": m["dataset_sha256"], "synthetic": False}
    # identical export -> identical hash (reproducible); window slice -> different hash
    m2 = export_dataset(root, "ds2", prim, [aux], aux_names={"USDTD_4h": "usdtd"}, created_utc="2025-06-01T00:00:00Z")
    assert m2["dataset_sha256"] == m["dataset_sha256"]
    m3 = export_dataset(root, "ds3", prim, [aux], start="2023-01-05", end="2023-01-10", compress=False)
    assert m3["dataset_sha256"] != m["dataset_sha256"] and m3["primary"]["file"] == "BTCUSDT_15m.csv"
    assert m3["primary"]["first_open"] == "2023-01-05T00:00:00Z"
    with pytest.raises(DatasetError):
        export_dataset(root, "ds1", prim, [aux])                      # exists, no overwrite


def test_tampered_dataset_is_detected_and_refused(series):
    root, prim, aux = series
    export_dataset(root, "ds1", prim, [aux], compress=False)
    d = root / "datasets" / "ds1"
    DatasetMarketData(d)                                              # ok
    with open(d / "BTCUSDT_15m.csv", "a") as f:
        f.write("2030-01-01T00:00:00Z,1,1,1,1,1\n")
    problems = verify_dataset(d)
    assert problems and problems[0].startswith("V10 hash mismatch")
    with pytest.raises(DatasetError):
        DatasetMarketData(d)
    assert len(DatasetMarketData(d, verify=False).get_ohlcv("BTC/USDT", "15m")) == 2001


def test_plain_folders_without_manifest_behave_like_before():
    root = Path(__file__).resolve().parents[1]
    p = DatasetMarketData(root / "data/sample")
    assert p.manifest is None and p.identity is None
    pd.testing.assert_frame_equal(p.get_ohlcv("BTC/USDT", "15m"), CSVMarketData(root / "data/sample").get_ohlcv("BTC/USDT", "15m"))
    assert dataset_identity(root / "data/sample") is None


# ------------------------------------------------------------------ ccxt source (stubbed, offline)
class _StubExchange:
    has = {"fetchOHLCV": True}
    version = "stub"

    def __init__(self, rows, fail=0):
        self.rows, self.fail, self.calls = rows, fail, []
        self.apiKey = self.secret = None

    def load_markets(self):
        return {"BTC/USDT": {}}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        if self.fail:
            self.fail -= 1
            raise RuntimeError("429 rate limited")
        return [r for r in self.rows if r[0] >= since][:limit]


def test_ccxt_source_refuses_credentials_and_only_uses_public_calls():
    rows = [[1_704_067_200_000 + i * 900_000, 1, 2, 0.5, 1.5, 3] for i in range(10)]
    with pytest.raises(SourceError):
        CCXTPublicSource("binance", options={"apiKey": "x", "secret": "y"}, client=_StubExchange(rows))
    bad = _StubExchange(rows); bad.apiKey = "leaked"
    with pytest.raises(SourceError):
        CCXTPublicSource("binance", client=bad)
    stub = _StubExchange(rows, fail=2)
    src = CCXTPublicSource("binance", client=stub, fetch_cfg={"max_retries": 5, "backoff_seconds": 0, "page_limit": 4})
    src._sleep = lambda s: None
    out = src.fetch("BTC/USDT", "15m", rows[0][0], 100)
    assert len(out) == 4 and out[0][0] == rows[0][0] and src.online
    assert src.describe()["endpoints"] == ["load_markets", "fetch_ohlcv"] and src.describe()["credentials"] is False
    assert all(c[0] == "BTC/USDT" for c in stub.calls)
    stub2 = _StubExchange(rows, fail=9)
    src2 = CCXTPublicSource("binance", client=stub2, fetch_cfg={"max_retries": 3, "backoff_seconds": 0})
    src2._sleep = lambda s: None
    with pytest.raises(SourceError, match="giving up after 3"):
        src2.fetch("BTC/USDT", "15m", rows[0][0], 10)


def test_ccxt_source_without_ccxt_installed_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "ccxt", None)                    # import ccxt -> ImportError
    with pytest.raises(SourceError, match="ccxt is not installed"):
        CCXTPublicSource("binance")


def test_source_spec_parsing_and_factory():
    assert parse_source_spec("ccxt:bybit") == ("ccxt", "bybit") and parse_source_spec("okx") == ("ccxt", "okx")
    assert parse_source_spec("file:/tmp/x.csv") == ("file", "/tmp/x.csv")
    with pytest.raises(SourceError):
        make_source("ftp:nowhere")
    with pytest.raises(SourceError):
        make_source("file:/definitely/missing.csv")


def test_ccxt_source_never_reads_environment(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "should-not-be-used")
    monkeypatch.setenv("BINANCE_SECRET", "should-not-be-used")
    captured = {}

    class _Ex(_StubExchange):
        def __init__(self, options):
            super().__init__([])
            captured.update(options)

    fake_ccxt = types.SimpleNamespace(binance=_Ex, __version__="stub")
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    CCXTPublicSource("binance")
    assert "apiKey" not in captured and "secret" not in captured and captured == {"enableRateLimit": True}
