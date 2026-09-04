"""Step 10 integration: `data` CLI end-to-end (offline file source), consumers reading a frozen dataset
unchanged (BacktestEngine == PaperTrader replay == in-memory frame), additive PaperTrader metadata,
improvement labelling, network confinement, and the "nothing else changes" guarantees."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_source import close_frame  # noqa: E402
from improvement_data import improvement_cfg, synthetic_frame  # noqa: E402
from src.backtesting import BacktestEngine  # noqa: E402
from src.data import CSVMarketData, DatasetMarketData  # noqa: E402
from src.data.cli import DataCLI  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402
from src.execution.paper_trader import PaperTrader  # noqa: E402
from src.main import CONFIG_PATH, load_config  # noqa: E402
from src.strategy.smc_strategy import SMCStrategy  # noqa: E402

PRIMARY = synthetic_frame(3000)          # has real SMC trades; timestamps 2023-01-01 15m
AUX = close_frame(300)                   # 2023-01-01 4h


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A scratch project root with real-data config, CSV imports, and a frozen dataset built via the CLI."""
    ws = tmp_path_factory.mktemp("ws")
    (ws / "in").mkdir()
    PRIMARY.to_csv(ws / "in/btc.csv", index=False)
    AUX.to_csv(ws / "in/usdtd.csv", index=False)
    cfg = improvement_cfg()                       # sample config + strategy tweaks that produce trades
    cfg["usdtd"]["enabled"] = True
    cfg["history"] = {"root": "data/history/", "default_source": "ccxt:binance",
                      "series": [{"symbol": "BTC/USDT", "timeframe": "15m", "kind": "ohlcv", "source": "ccxt:binance", "start": "2023-01-01"},
                                 {"symbol": "USDT.D", "timeframe": "4h", "kind": "close", "source": "file"}],
                      "fetch": {"overlap_bars": 3}, "validation": {"max_gap_pct": 0.5}, "export": {"compress": True}}
    lines = []
    cli = DataCLI(cfg, ws, out=lines.append)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{ws / 'in/btc.csv'}", "--from", "2023-01-01"]) == 0
    assert cli.run(["download", "--symbol", "USDT.D", "--timeframe", "4h", "--source", f"file:{ws / 'in/usdtd.csv'}", "--from", "2023-01-01"]) == 0
    assert cli.run(["export", "--dataset", "ds-test"]) == 0
    return ws, cfg, lines


# ------------------------------------------------------------------ CLI
def test_cli_builds_series_and_dataset_offline(workspace):
    ws, cfg, lines = workspace
    text = "\n".join(lines)
    assert "local file, offline" in text and "ccxt" not in text.split("data download")[1].split("\n")[1]
    assert (ws / "data/history/file/BTCUSDT_15m/BTCUSDT_15m.csv").exists()
    assert (ws / "data/history/file/USDTD_4h/series.json").exists()
    ds = ws / "data/history/datasets/ds-test"
    m = load_manifest(ds)
    assert m["primary"]["file"] == "BTCUSDT_15m.csv.gz" and m["auxiliary"][0]["name"] == "usdtd"
    assert m["validation"] == {"BTCUSDT_15m": "ok", "USDTD_4h": "ok"}
    assert "config.yaml was NOT modified" in text


def test_cli_list_validate_inspect_are_offline_and_clean(workspace):
    ws, cfg, _ = workspace
    out = []
    cli = DataCLI(cfg, ws, out=out.append)
    assert cli.run(["list"]) == 0 and any("dataset ds-test" in l for l in out)
    out.clear(); assert cli.run(["validate"]) == 0 and "status ok" in out[0]
    out.clear(); assert cli.run(["validate", "--dataset", "ds-test"]) == 0 and "dataset ds-test: ok" in out[-1]
    out.clear(); assert cli.run(["inspect"]) == 0 and any("alignment vs BTCUSDT_15m" in l for l in out)
    out.clear(); assert cli.run(["inspect", "--dataset", "ds-test"]) == 0 and "verification    : ok" in out[-1]
    out.clear(); assert cli.run(["update", "--dry-run"]) == 0                 # file series are skipped with a note
    out.clear(); assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--dry-run"]) in (0, 1)
    assert any("ccxt is not installed" in l or "DRY RUN" in l for l in out)   # network path only via ccxt
    out.clear(); assert cli.run(["bogus"]) == 2
    out.clear(); assert cli.run(["export"]) == 1 and "REFUSED" in out[0]
    out.clear(); assert cli.run(["export", "--dataset", "ds-test"]) == 1 and "already exists" in out[0]


def test_cli_validate_flags_tampered_series(workspace):
    ws, cfg, _ = workspace
    p = ws / "data/history/file/USDTD_4h/USDTD_4h.csv"
    original = p.read_bytes()
    try:
        with open(p, "a") as f:
            f.write("2023-01-01T00:00:00Z,4.0\n")                              # duplicate row
        out = []
        assert DataCLI(cfg, ws, out=out.append).run(["validate", "--series", "USDTD_4h"]) == 1
        assert any("V4" in l for l in out) and any("V10 sha256 MISMATCH" in l for l in out)
    finally:
        p.write_bytes(original)
        DataCLI(cfg, ws, out=lambda s: None).run(["validate", "--series", "USDTD_4h"])


# ------------------------------------------------------------------ consumers
def test_backtest_and_paper_replay_on_frozen_dataset_equal_in_memory_run(workspace):
    ws, cfg, _ = workspace
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["directory"] = "data/history/datasets/ds-test/"
    provider = CSVMarketData(ws / cfg["data"]["directory"])
    assert provider.resolve_path("BTC/USDT", "15m").suffix == ".gz"
    bt = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ws).run_provider(provider)
    mem = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ws).run(PRIMARY)
    assert bt.metrics["trades"] == mem.metrics["trades"] >= 3
    for k, v in mem.metrics.items():
        assert bt.metrics[k] == pytest.approx(v, rel=1e-9, abs=1e-9), k      # CSV round-trip, float noise only
    assert [t.entry_timestamp for t in bt.journal.trades] == [t.entry_timestamp for t in mem.journal.trades]
    # paper replay over the first 1200 dataset bars == backtest over the same bars (Step 8 equivalence, real file path)
    short = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ws).run(PRIMARY.iloc[:1200].reset_index(drop=True))
    trader = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=ws / "data/paper", data_root=ws, reset=True)
    trader.run_replay(provider, limit=1200)
    trader.close()
    closed = [t for t in short.journal.trades if t.exit_reason != "end_of_data"]
    assert len(trader.journal.trades) == len(closed) >= 1
    assert [t.net_pnl for t in trader.journal.trades] == pytest.approx([t.net_pnl for t in closed])


def test_paper_state_carries_dataset_metadata_additively(workspace):
    ws, cfg, _ = workspace
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["directory"] = "data/history/datasets/ds-test/"
    m = load_manifest(ws / cfg["data"]["directory"])
    state = json.load(open(ws / "data/paper/state.json"))
    assert state["dataset"] == {"dataset_id": "ds-test", "dataset_sha256": m["dataset_sha256"], "synthetic": False}
    # resume works and metadata is preserved; a plain folder yields null metadata and the same schema
    t2 = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=ws / "data/paper", data_root=ws)
    assert t2.dataset_info["dataset_id"] == "ds-test"
    t2.close()
    cfg["data"]["directory"] = "data/sample/"
    t3 = PaperTrader(load_config(), SMCStrategy.from_config(load_config()), state_dir=ws / "data/paper2", data_root=ROOT, reset=True)
    t3.run_replay(CSVMarketData(ROOT / "data/sample"), limit=5); t3.close()
    s3 = json.load(open(ws / "data/paper2/state.json"))
    assert s3["dataset"] is None and s3["schema_version"] == state["schema_version"]


def test_improvement_uses_manifest_synthetic_flag_and_dataset_label(workspace, monkeypatch):
    ws, cfg, _ = workspace
    import src.main as main_mod
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["directory"] = "data/history/datasets/ds-test/"
    cfg["improvement"]["results_directory"] = str(ws / "imp")
    cfg["improvement"]["data"]["min_trades_per_fold"] = 500       # abort fast: we only test labelling
    monkeypatch.setattr(main_mod, "ROOT", ws)
    res = main_mod.run_improve(cfg, run_id="lbl")
    assert res.aborted
    summary = json.load(open(ws / "imp/lbl/summary.json"))
    assert summary["synthetic_data"] is False and "dataset ds-test" in json.dumps(summary["data"])
    assert "SYNTHETIC" not in (ws / "imp/lbl/report.md").read_text()


# ------------------------------------------------------------------ safety
def test_network_package_is_confined_and_ccxt_only_in_ccxt_source():
    for py in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(py.read_text())
        rel = py.relative_to(ROOT / "src").as_posix()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                if n == "ccxt" or n.startswith("ccxt."):
                    assert rel == "data/fetch/ccxt_source.py", rel
                if n.startswith("src.data.fetch"):
                    assert rel.startswith("data/fetch/") or rel == "data/cli.py", rel
                assert not any(n.startswith(m) for m in ("requests", "urllib", "http.client", "aiohttp", "websocket", "socket")), (rel, n)
    # nothing outside src/data imports the CLI either
    for py in (ROOT / "src").rglob("*.py"):
        if not py.as_posix().startswith(str(ROOT / "src/data")) and py.name != "main.py":
            assert "src.data.cli" not in py.read_text() and "src.data.fetch" not in py.read_text(), py


def test_no_trading_or_private_endpoints_referenced():
    text = (ROOT / "src/data/fetch/ccxt_source.py").read_text()
    for forbidden in ("create_order", "createOrder", "fetch_balance", "fetchBalance", "cancel_order", "withdraw",
                      "fetch_my_trades", "privateGet", "privatePost", "sign(", "os.environ", "getenv"):
        assert forbidden not in text, forbidden


def test_pipeline_leaves_config_src_and_paper_state_untouched(tmp_path):
    cfg_before = CONFIG_PATH.read_bytes()
    src_before = _tree_hash(ROOT / "src")
    paper = tmp_path / "data/paper"; paper.mkdir(parents=True)
    (paper / "state.json").write_text('{"sentinel": 1}')
    PRIMARY.iloc[:500].to_csv(tmp_path / "btc.csv", index=False)
    cfg = load_config()
    out = []
    cli = DataCLI(cfg, tmp_path, out=out.append)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{tmp_path / 'btc.csv'}", "--from", "2023-01-01"]) == 0
    assert cli.run(["export", "--dataset", "d1"]) == 0
    cli.run(["validate"]); cli.run(["inspect"]); cli.run(["list"])
    assert CONFIG_PATH.read_bytes() == cfg_before and _tree_hash(ROOT / "src") == src_before
    assert (paper / "state.json").read_text() == '{"sentinel": 1}'
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert all(w.startswith("data/history/") or w in ("btc.csv", "data/paper/state.json") for w in written), written


def test_sample_workflows_unchanged():
    """The default config still points at data/sample and the fixture backtest baseline is untouched."""
    cfg = load_config()
    assert cfg["data"]["directory"] == "data/sample/"
    from src.backtesting.demo_strategy import FixedIntervalTestStrategy
    res = BacktestEngine.from_config(cfg, FixedIntervalTestStrategy.from_config(cfg), data_root=ROOT).run_provider(
        CSVMarketData(ROOT / "data/sample"))
    assert res.metrics["trades"] == 5 and round(res.metrics["total_return_pct"], 2) == -3.57
    assert "history" in cfg and cfg["history"]["default_source"] == "ccxt:binance"
