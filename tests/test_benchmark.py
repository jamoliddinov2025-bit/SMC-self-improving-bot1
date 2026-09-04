"""Step 11 - baseline benchmark: config, dataset identity, validation gating, metrics, unavailable statistics,
deterministic reports, point-in-time protections, backtest/paper consistency, CLI, tampered-data refusal,
and the read-only guarantees (no config / strategy / risk mutation, no proposal application)."""

import copy
import datetime as dt
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
from src.backtesting import EXIT_END, BacktestEngine  # noqa: E402
from src.backtesting.strategy import BacktestContext  # noqa: E402
from src.benchmark import (BenchmarkConfig, BenchmarkRunner, DISCLAIMER, LABEL_REAL, LABEL_SYNTHETIC,  # noqa: E402
                           config_snapshot, snapshot_hash)
from src.data import DatasetMarketData  # noqa: E402
from src.data.cli import DataCLI  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402
from src.execution.paper_trader import PaperTrader  # noqa: E402
from src.main import CONFIG_PATH, load_config  # noqa: E402
from src.strategy.smc_strategy import SMCStrategy  # noqa: E402

PRIMARY = synthetic_frame(3000)          # deterministic fixture (seed 7); produces real SMC trades
AUX = close_frame(300)
NOW = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
DS = "data/history/datasets/ds-bench/"


def _bench_cfg():
    cfg = improvement_cfg()
    cfg["improvement"]["enabled"] = False          # benchmark never needs it
    cfg["usdtd"]["enabled"] = True
    cfg["history"] = {"root": "data/history/", "default_source": "ccxt:binance",
                      "series": [{"symbol": "BTC/USDT", "timeframe": "15m", "kind": "ohlcv", "source": "ccxt:binance", "start": "2023-01-01"},
                                 {"symbol": "USDT.D", "timeframe": "4h", "kind": "close", "source": "file"}],
                      "fetch": {"overlap_bars": 3}, "validation": {"max_gap_pct": 0.5}, "export": {"compress": True}}
    return cfg


def _build(ws: Path, cfg, primary=PRIMARY, aux=AUX, dataset_id="ds-bench", synthetic=True):
    (ws / "in").mkdir(exist_ok=True)
    primary.to_csv(ws / "in/btc.csv", index=False)
    aux.to_csv(ws / "in/usdtd.csv", index=False)
    cli = DataCLI(cfg, ws, out=lambda _: None)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{ws / 'in/btc.csv'}", "--from", "2023-01-01"]) == 0
    assert cli.run(["download", "--symbol", "USDT.D", "--timeframe", "4h", "--source", f"file:{ws / 'in/usdtd.csv'}", "--from", "2023-01-01"]) == 0
    args = ["export", "--dataset", dataset_id, "--overwrite"] + (["--synthetic"] if synthetic else [])
    assert cli.run(args) == 0
    return ws / "data/history/datasets" / dataset_id


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("bench")
    cfg = _bench_cfg()
    _build(ws, cfg)
    return ws, cfg


def run(ws, cfg, bid="b", **kw):
    kw.setdefault("dataset_directory", DS)
    kw.setdefault("now", NOW)
    kw.setdefault("commit", "deadbeef")
    return BenchmarkRunner(copy.deepcopy(cfg), ws, benchmark_id=bid, **kw).run()


@pytest.fixture(scope="module")
def result(workspace):
    ws, cfg = workspace
    return run(ws, cfg, "base")


# ------------------------------------------------------------------ config
def test_benchmark_config_defaults_and_from_yaml():
    b = BenchmarkConfig.from_config({})
    assert b.results_directory == "data/benchmarks/" and b.min_trades_for_statistics == 30 and not b.fail_on_warnings
    cfg = load_config()
    assert "benchmark" in cfg
    b = BenchmarkConfig.from_config(cfg)
    assert b.results_directory == cfg["benchmark"]["results_directory"]
    assert b.dataset_directory is None


def test_config_snapshot_is_immutable_artifact_with_stable_hash():
    cfg = load_config()
    snap = config_snapshot(cfg)
    h = snapshot_hash(snap)
    assert h == snapshot_hash(config_snapshot(load_config()))
    snap["risk"]["starting_balance"] = 1  # mutate the snapshot -> original untouched, hash changes
    assert cfg["risk"]["starting_balance"] != 1 and snapshot_hash(snap) != h
    for k in ("strategy", "risk", "execution", "indicators", "usdtd", "market", "backtesting"):
        assert k in snap


# ------------------------------------------------------------------ identity
def test_manifest_records_full_identity(workspace, result):
    ws, cfg = workspace
    assert not result.aborted
    m = result.manifest
    ds = load_manifest(ws / DS)
    assert m["dataset"]["dataset_id"] == "ds-bench" and m["dataset"]["dataset_sha256"] == ds["dataset_sha256"]
    assert m["dataset"]["primary"]["sha256"] == ds["primary"]["sha256"]
    assert m["dataset"]["auxiliary"][0]["sha256"] == ds["auxiliary"][0]["sha256"]
    assert m["strategy"]["name"] == "smc" and len(m["configuration"]["snapshot_sha256"]) == 64
    assert m["starting_equity"] == cfg["risk"]["starting_balance"]
    assert m["symbol"] == "BTC/USDT" and m["timeframe"] == "15m"
    assert m["benchmark_date_range"] == {"start": ds["primary"]["first_open"], "end": ds["primary"]["last_open"]}
    assert m["fee_rate_pct"] == cfg["execution"]["paper_fee_pct"] and m["slippage_pct"] == cfg["execution"]["slippage_pct"]
    assert m["repository_commit"] == "deadbeef" and m["generated_utc"] == "2025-01-01T00:00:00Z"
    assert m["validation_status"] == "ok" and m["disclaimer"] == DISCLAIMER
    assert m["safety"] == {"config_yaml_modified": False, "strategy_modified": False, "risk_modified": False,
                           "improvement_enabled": False, "proposals_applied": 0, "network_used": False}


def test_synthetic_dataset_is_labelled_not_real(workspace, result):
    ws, _ = workspace
    assert result.manifest["synthetic"] is True and result.manifest["label"] == LABEL_SYNTHETIC
    report = (ws / "data/benchmarks/base/report.md").read_text()
    assert "SYNTHETIC" in report.splitlines()[0] and "NOT real market data" in report
    assert not report.startswith("# " + LABEL_REAL)


def test_unlabelled_export_is_reported_as_real_baseline(tmp_path):
    """A dataset exported without --synthetic (the normal `data download`/`export` path) gets the REAL label."""
    cfg = _bench_cfg()
    _build(tmp_path, cfg, synthetic=False)
    res = run(tmp_path, cfg, "real")
    assert not res.aborted and res.manifest["synthetic"] is False and res.manifest["label"] == LABEL_REAL
    rep = (tmp_path / "data/benchmarks/real/report.md").read_text()
    assert rep.startswith("# " + LABEL_REAL) and "does not guarantee future performance" in rep
    assert "recommend" not in rep.lower().replace("not a recommendation", "")


# ------------------------------------------------------------------ metrics
def test_metrics_cover_required_fields_and_match_engine(workspace, result):
    ws, cfg = workspace
    x, bt = result.metrics, result.backtest
    required = ["starting_equity", "ending_equity", "net_pnl", "return_pct", "total_trades", "winning_trades",
                "losing_trades", "win_rate_pct", "expectancy", "average_r", "median_r", "profit_factor", "max_drawdown_pct",
                "max_consecutive_losses", "average_trade_duration_bars", "fees", "slippage", "risk_rejections",
                "exit_reasons", "signal_funnel"]
    assert all(k in x for k in required)
    assert x["total_trades"] == bt.metrics["trades"] >= 30 and x["statistics_available"]
    assert x["net_pnl"] == pytest.approx(bt.metrics["net_profit"]) and x["ending_equity"] == bt.metrics["ending_equity"]
    rs = sorted(t.r_multiple for t in bt.journal.trades)
    assert x["median_r"] == pytest.approx(pd.Series(rs).median())
    assert x["average_r"] == pytest.approx(bt.metrics["average_r_multiple"])
    # max consecutive losses by hand
    worst = cur = 0
    for t in bt.journal.trades:
        cur = cur + 1 if t.net_pnl < 0 else 0
        worst = max(worst, cur)
    assert x["max_consecutive_losses"] == worst
    assert x["average_trade_duration_bars"] == pytest.approx(sum(t.bars_held for t in bt.journal.trades) / len(bt.journal.trades))
    assert x["fees"]["fee_rate_pct"] == cfg["execution"]["paper_fee_pct"] and x["fees"]["total_fees"] == bt.metrics["total_fees"]
    assert x["slippage"]["slippage_pct"] == cfg["execution"]["slippage_pct"] and x["slippage"]["on_targets"] is False


def test_signal_funnel_distinguishes_signal_approved_executed_closed(workspace, result):
    f, bt, x = result.metrics["signal_funnel"], result.backtest, result.metrics
    d = x["strategy_diagnostics"]
    assert f["buy_signals"] == d["buy_signals"]
    assert f["risk_rejected_buys"] == len(bt.journal.rejections)
    assert f["risk_approved_buys"] == f["buy_signals"] - f["risk_rejected_buys"]
    assert f["executed_buys"] == len(bt.journal.trades)
    assert f["closed_trades"] + f["force_closed_end_of_data"] == f["executed_buys"]
    assert f["closed_trades"] == sum(1 for t in bt.journal.trades if t.exit_reason != EXIT_END)


def test_funnel_counts_risk_rejections_when_risk_engine_rejects(workspace):
    ws, cfg = workspace
    cfg = copy.deepcopy(cfg)
    cfg["risk"]["max_consecutive_losses"] = 2          # loss-streak lock -> rejections
    res = run(ws, cfg, "rej")
    f = res.metrics["signal_funnel"]
    assert f["risk_rejected_buys"] > 0 and res.metrics["risk_rejections"]
    assert f["risk_approved_buys"] == f["buy_signals"] - f["risk_rejected_buys"] == f["executed_buys"]


def test_insufficient_trades_reports_unavailable_not_invented(workspace):
    ws, cfg = workspace
    cfg = copy.deepcopy(cfg)
    cfg["benchmark"] = {"min_trades_for_statistics": 1000}
    res = run(ws, cfg, "few")
    x = res.metrics
    assert not x["statistics_available"] and "unavailable_reason" in x
    for k in ("win_rate_pct", "expectancy", "average_r", "median_r", "profit_factor"):
        assert x[k] is None
    assert x["total_trades"] > 0 and x["net_pnl"] is not None      # counts and P&L still reported
    rep = (ws / "data/benchmarks/few/report.md").read_text()
    assert "| win rate | unavailable |" in rep and "Ratio statistics unavailable" in rep


def test_zero_trade_run_has_no_nan_and_is_unavailable(tmp_path):
    cfg = _bench_cfg()
    cfg["usdtd"]["enabled"] = False
    cfg["strategy"]["filters"]["ema_trend"]["enabled"] = True     # sample-style config -> no trades on this fixture window
    _build(tmp_path, cfg, primary=PRIMARY.iloc[:400].reset_index(drop=True))
    res = run(tmp_path, cfg, "zero")
    assert not res.aborted
    x = res.metrics
    assert x["total_trades"] == 0 and x["median_r"] is None and x["average_trade_duration_bars"] is None
    assert "NaN" not in json.dumps(x) and x["profit_factor"] is None
    assert x["signal_funnel"]["executed_buys"] == 0


# ------------------------------------------------------------------ determinism
def test_report_and_metrics_are_deterministic(workspace, result):
    ws, cfg = workspace
    again = run(ws, cfg, "again")
    assert again.metrics == result.metrics
    a = (ws / "data/benchmarks/base/report.md").read_text().replace("base", "X")
    b = (ws / "data/benchmarks/again/report.md").read_text().replace("again", "X")
    assert a == b
    ma, mb = dict(result.manifest), dict(again.manifest)
    ma.pop("benchmark_id"), mb.pop("benchmark_id")
    assert ma == mb
    # generation timestamp is metadata only: a different clock changes nothing but the stamp
    later = run(ws, cfg, "later", now=dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc))
    assert later.metrics == result.metrics
    assert (ws / "data/benchmarks/later/trades.csv").read_bytes() == (ws / "data/benchmarks/base/trades.csv").read_bytes()


def test_artifacts_written_and_trade_journal_matches_engine(workspace, result):
    ws, _ = workspace
    out = ws / "data/benchmarks/base"
    for f in ("manifest.json", "metrics.json", "trades.csv", "rejections.csv", "equity_curve.csv", "validation.json", "report.md"):
        assert (out / f).exists(), f
    trades = pd.read_csv(out / "trades.csv")
    assert len(trades) == result.metrics["total_trades"]
    assert json.load(open(out / "metrics.json")) == result.metrics
    v = json.load(open(out / "validation.json"))
    assert v["status"] == "ok" and v["series"]["primary"]["rows"] == 3000 and "usdtd" in v["series"]["auxiliary"]
    assert v["series"]["auxiliary"]["usdtd"]["alignment"]["first_visible_primary_index"] == 15


# ------------------------------------------------------------------ validation gating / tampering
def test_refuses_unfrozen_directory_and_missing_dataset(tmp_path):
    cfg = _bench_cfg()
    res = run(tmp_path, cfg, "nods", dataset_directory="data/sample/")
    assert res.aborted and "not a frozen dataset" in res.abort_reason
    assert not (tmp_path / "data/benchmarks").exists()          # nothing written for a non-dataset


def test_refuses_tampered_dataset(tmp_path):
    cfg = _bench_cfg()
    d = _build(tmp_path, cfg)
    ok = run(tmp_path, cfg, "ok")
    assert not ok.aborted
    m = load_manifest(d)
    fpath = d / m["primary"]["file"]
    df = pd.read_csv(fpath)
    df.loc[100, "close"] = df.loc[100, "close"] * 1.001         # silent edit, same shape
    df.to_csv(fpath, index=False, compression="gzip")
    res = run(tmp_path, cfg, "tampered")
    assert res.aborted and "hash verification failed" in res.abort_reason
    assert not (tmp_path / "data/benchmarks/tampered/metrics.json").exists()
    with pytest.raises(Exception):
        DatasetMarketData(d)


def _corrupt_frozen(d: Path, mutate):
    """Simulate a dataset frozen with bad content (hashes consistent, so only validation can catch it)."""
    from src.data.dataset import dataset_hash, sha256_file
    m = load_manifest(d)
    fpath = d / m["primary"]["file"]
    df = pd.read_csv(fpath)
    mutate(df)
    df.to_csv(fpath, index=False, compression="gzip")
    m["primary"]["sha256"] = sha256_file(fpath)
    m["dataset_sha256"] = dataset_hash({e["file"]: e["sha256"] for e in [m["primary"]] + m["auxiliary"]})
    (d / "manifest.json").write_text(json.dumps(m, indent=2))


def test_download_refuses_ohlc_violations_upstream(tmp_path):
    cfg = _bench_cfg()
    bad = PRIMARY.copy()
    bad.loc[500, "high"] = bad.loc[500, "low"] - 1.0
    (tmp_path / "in").mkdir()
    bad.to_csv(tmp_path / "in/btc.csv", index=False)
    cli = DataCLI(cfg, tmp_path, out=lambda _: None)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{tmp_path / 'in/btc.csv'}", "--from", "2023-01-01"]) != 0


def test_critical_data_problem_fails_benchmark(tmp_path):
    cfg = _bench_cfg()
    d = _build(tmp_path, cfg)

    def mutate(df):
        df.loc[500, "high"] = df.loc[500, "low"] - 1.0           # OHLC violation (V6, error)
    _corrupt_frozen(d, mutate)
    res = run(tmp_path, cfg, "bad")
    assert res.aborted and "validation failed" in res.abort_reason and "V6" in res.abort_reason
    assert res.validation["status"] == "failed" and not res.metrics
    rep = (tmp_path / "data/benchmarks/bad/report.md").read_text()
    assert rep.startswith("# BENCHMARK ABORTED") and "V6" in rep
    assert not (tmp_path / "data/benchmarks/bad/metrics.json").exists()


def test_gaps_are_surfaced_as_warnings_and_can_be_made_fatal(tmp_path):
    cfg = _bench_cfg()
    gappy = PRIMARY.drop(index=range(1000, 1004)).reset_index(drop=True)    # 4 missing bars, not repaired
    _build(tmp_path, cfg, primary=gappy)
    assert len(pd.read_csv(tmp_path / DS / "BTCUSDT_15m.csv.gz")) == 2996
    res = run(tmp_path, cfg, "gap")
    assert not res.aborted and res.validation["status"] == "warnings"
    assert any("V5" in w for w in res.validation["warnings"])
    assert res.metrics["bars"] == 2996                               # no fill / interpolation
    rep = (tmp_path / "data/benchmarks/gap/report.md").read_text()
    assert "Warnings (not hidden)" in rep and "missing 4" in rep
    cfg2 = copy.deepcopy(cfg)
    cfg2["benchmark"] = {"fail_on_warnings": True}
    res2 = run(tmp_path, cfg2, "gapfatal")
    assert res2.aborted and res2.validation["status"] == "failed"


def test_missing_required_aux_feed_is_critical(tmp_path):
    cfg = _bench_cfg()
    (tmp_path / "in").mkdir()
    PRIMARY.to_csv(tmp_path / "in/btc.csv", index=False)
    cli = DataCLI(cfg, tmp_path, out=lambda _: None)
    assert cli.run(["download", "--symbol", "BTC/USDT", "--timeframe", "15m", "--source", f"file:{tmp_path / 'in/btc.csv'}", "--from", "2023-01-01"]) == 0
    assert cli.run(["export", "--dataset", "ds-bench", "--series", "BTCUSDT_15m", "--synthetic"]) == 0
    res = run(tmp_path, cfg, "noaux")
    assert res.aborted and "usdtd" in res.abort_reason and "missing" in res.abort_reason
    cfg2 = copy.deepcopy(cfg)
    cfg2["usdtd"]["enabled"] = False
    assert not run(tmp_path, cfg2, "noaux-ok").aborted


def test_symbol_timeframe_mismatch_refused(workspace):
    ws, cfg = workspace
    cfg = copy.deepcopy(cfg)
    cfg["market"]["timeframe"] = "1h"
    res = run(ws, cfg, "mism")
    assert res.aborted and "config expects BTC/USDT 1h" in res.abort_reason


def test_dry_run_validates_but_runs_nothing_and_writes_nothing(tmp_path):
    cfg = _bench_cfg()
    _build(tmp_path, cfg)
    res = run(tmp_path, cfg, "dry", dry_run=True)
    assert not res.aborted and res.manifest["dry_run"] is True and res.manifest["metrics_available"] is False
    assert res.validation["status"] == "ok" and res.metrics == {} and res.backtest is None
    assert not (tmp_path / "data/benchmarks").exists()


# ------------------------------------------------------------------ point-in-time / lookahead
class _Spy:
    """Wraps the real SMCStrategy, recording what it could see on every bar (no behaviour change)."""

    def __init__(self, inner, primary):
        self.inner, self.primary, self.seen = inner, primary, []

    def on_candle(self, ctx: BacktestContext):
        self.seen.append((ctx.index, ctx.candle.timestamp, [b.detected_index for b in ctx.smc.bos_events],
                          [c.detected_index for c in ctx.smc.choch_events], ctx.regime, ctx.indicators.copy()))
        return self.inner.on_candle(ctx)

    @property
    def diag(self):
        return self.inner.diag


def test_benchmark_strategy_only_sees_closed_bars_and_closed_usdtd(workspace, monkeypatch):
    """SMC events (1), indicators (4) and USDT.D regime (3) available to the strategy never reference bar > i."""
    ws, cfg = workspace
    import src.benchmark.runner as runner_mod
    spies = []
    original = SMCStrategy.from_config

    def factory(c):
        s = _Spy(original(c), PRIMARY)
        spies.append(s)
        return s
    monkeypatch.setattr(runner_mod.SMCStrategy, "from_config", staticmethod(factory))
    res = run(ws, cfg, "spy")
    assert not res.aborted and spies
    seen = spies[0].seen
    assert len(seen) == 3000
    aux_ts = list(pd.to_datetime(AUX["timestamp"], utc=True))
    for i, ts, bos, choch, regime, ind in seen:
        assert all(b <= i for b in bos) and all(c <= i for c in choch)
        bar_close = ts + pd.Timedelta(minutes=15)
        if regime is not None and regime.source_index >= 0:
            assert aux_ts[regime.source_index] + pd.Timedelta(hours=4) <= bar_close
            if regime.source_index + 1 < len(aux_ts):
                assert aux_ts[regime.source_index + 1] + pd.Timedelta(hours=4) > bar_close
    # indicators: the row the strategy saw at bar i equals the row computed from bars [0..i] only
    from src.indicators import IndicatorEngine
    from src.indicators.engine import IndicatorConfig
    eng = IndicatorEngine(IndicatorConfig.from_config(cfg))
    for i in (250, 1500, 2999):
        prefix = eng.compute(PRIMARY.iloc[:i + 1].reset_index(drop=True)).iloc[-1]
        full_row = seen[i][5]
        for col in prefix.index:
            if col in full_row.index and pd.notna(prefix[col]):
                assert full_row[col] == pytest.approx(prefix[col], rel=1e-9), (i, col)


def test_benchmark_prefix_equals_full_run_and_future_shock_leaves_past_unchanged(tmp_path):
    """(2) fills at N+1 open, (5) stops/targets, (6) report content: nothing before the cut depends on data after it."""
    cfg = _bench_cfg()
    _build(tmp_path, cfg)
    full = run(tmp_path, cfg, "full")
    k = 1800
    shocked = PRIMARY.copy()
    shocked.loc[k:, ["open", "high", "low", "close"]] *= 3.0
    _build(tmp_path, cfg, primary=shocked, dataset_id="ds-shock")
    other = run(tmp_path, cfg, "shock", dataset_directory="data/history/datasets/ds-shock/")
    assert not full.aborted and not other.aborted
    fb, ob = full.backtest, other.backtest
    pd.testing.assert_frame_equal(fb.equity_curve.iloc[:k], ob.equity_curve.iloc[:k])
    a = [t.to_dict() for t in fb.journal.trades if t.exit_index < k]
    b = [t.to_dict() for t in ob.journal.trades if t.exit_index < k]
    assert a == b and len(a) >= 5
    for t in a:
        assert t["entry_index"] == t["signal_index"] + 1                      # N signal -> N+1 open fill
        assert t["entry_price"] == pytest.approx(PRIMARY["open"].iloc[t["entry_index"]] * (1 + cfg["execution"]["slippage_pct"] / 100))
        if t["exit_reason"] == "stop_loss":
            assert t["exit_price"] <= t["stop_loss"] * (1 - cfg["execution"]["slippage_pct"] / 100) + 1e-9
        if t["exit_reason"] == "take_profit":
            assert t["exit_price"] == pytest.approx(t["take_profit"])           # no target slippage by default
    assert not fb.equity_curve.iloc[k:].equals(ob.equity_curve.iloc[k:])
    # the report's identity section is prefix-independent, and metrics are only computed after the run
    assert full.metrics != other.metrics


def test_benchmark_stops_and_targets_are_the_engines_not_recomputed(workspace, result):
    """(5) trades.csv is the engine journal verbatim: same stop/target/exit fields, no benchmark-side recomputation."""
    ws, _ = workspace
    csv = pd.read_csv(ws / "data/benchmarks/base/trades.csv")
    eng = result.backtest.journal.to_frame()
    for col in ("stop_loss", "take_profit", "entry_price", "exit_price", "exit_reason", "signal_index", "entry_index", "exit_index"):
        assert list(csv[col].fillna(-1)) == pytest.approx(list(eng[col].fillna(-1))) if col != "exit_reason" else list(csv[col]) == list(eng[col])


# ------------------------------------------------------------------ backtest <-> paper consistency
def test_paper_trader_replay_of_benchmark_dataset_matches_benchmark(workspace, result):
    """(7 + replay equivalence) same frozen dataset through PaperTrader.process_candle == BacktestEngine run,
    for signals, timing, fills, stops, targets, exits, timestamps, equity. Only persistence metadata differs."""
    ws, cfg = workspace
    n = 1500
    cfg = copy.deepcopy(cfg)
    cfg["data"]["directory"] = DS
    cfg["backtesting"]["close_open_position_at_end"] = False        # a live trader never force-closes
    provider = DatasetMarketData(ws / DS)
    frame = provider.get_ohlcv("BTC/USDT", "15m")                    # both sides read the SAME frozen file
    trader = PaperTrader(cfg, SMCStrategy.from_config(cfg), state_dir=ws / "data/paper-bench", data_root=ws, reset=True)
    trader.run_replay(provider, limit=n)
    short = BacktestEngine.from_config(cfg, SMCStrategy.from_config(cfg), data_root=ws).run(frame.iloc[:n].reset_index(drop=True))
    bt_curve = short.equity_curve[["timestamp", "close", "cash", "position", "equity"]]
    pd.testing.assert_frame_equal(bt_curve, trader.equity_curve())
    closed = [t for t in short.journal.trades if t.exit_reason != EXIT_END]
    assert len(closed) >= 10 and len(trader.journal.trades) == len(closed)
    for a, b in zip(closed, trader.journal.trades):
        assert a.to_dict() == b.to_dict()
    assert short.journal.rejections_frame().equals(trader.journal.rejections_frame())
    # the benchmark's own journal (full run) has these same trades as its prefix
    bench = [t.to_dict() for t in result.backtest.journal.trades if t.exit_index < n]
    assert bench == [t.to_dict() for t in closed]
    st = trader.status()
    assert st["position"] is None or st["position"]["trade_id"] == len(closed) + 1
    assert st["portfolio"]["equity"] == pytest.approx(short.equity_curve["equity"].iloc[-1])
    assert st["risk"]["consecutive_losses"] >= 0                           # persistence metadata exists, unused above
    trader.close()


# ------------------------------------------------------------------ read-only guarantees
def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def test_benchmark_does_not_mutate_config_strategy_or_risk(workspace):
    ws, cfg = workspace
    before_yaml = CONFIG_PATH.read_bytes()
    before_src = _tree_hash(ROOT / "src")
    frozen = copy.deepcopy(cfg)
    strat_probe = SMCStrategy.from_config(cfg)
    res = run(ws, cfg, "ro")
    assert not res.aborted
    assert cfg == frozen                                                   # in-memory config untouched
    assert CONFIG_PATH.read_bytes() == before_yaml and _tree_hash(ROOT / "src") == before_src
    assert SMCStrategy.from_config(cfg).cfg == strat_probe.cfg              # strategy config identical
    snap = config_snapshot(cfg)
    assert res.manifest["configuration"]["snapshot"] == snap
    assert res.manifest["configuration"]["snapshot"]["risk"] == frozen["risk"]
    assert not (ws / "config").exists()                                    # no proposed/alternate config written
    assert not (ws / "data/improvement").exists()                          # no improvement run
    assert not any(p.name.startswith("config.proposed") for p in ws.rglob("*.yaml"))
    assert res.manifest["safety"]["proposals_applied"] == 0 and res.manifest["safety"]["improvement_enabled"] is False


def test_benchmark_never_imports_improvement_or_network_code():
    import ast
    for p in (ROOT / "src/benchmark").glob("*.py"):
        tree = ast.parse(p.read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.startswith(("src.improvement", "src.data.fetch", "ccxt", "requests", "urllib")) for m in mods), (p, mods)


def test_benchmark_output_dir_is_gitignored():
    import subprocess
    out = subprocess.run(["git", "check-ignore", "-q", "data/benchmarks/x/report.md"], cwd=ROOT)
    assert out.returncode == 0


def test_config_yaml_still_loads_and_other_modes_unaffected():
    cfg = load_config()
    assert cfg["improvement"]["enabled"] is False and cfg["mode"] in ("paper", "backtest")
    assert yaml.safe_load(CONFIG_PATH.read_text())["benchmark"]["dataset_directory"] is None


# ------------------------------------------------------------------ CLI
def test_cli_benchmark_dry_run_and_full_run(tmp_path, monkeypatch, capsys):
    import src.main as main_mod
    cfg = _bench_cfg()
    _build(tmp_path, cfg)
    monkeypatch.setattr(main_mod, "ROOT", tmp_path)
    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: copy.deepcopy(cfg))
    monkeypatch.setattr(sys, "argv", ["main.py", "benchmark", "--dataset", "ds-bench", "--id", "cli", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        main_mod.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "READ-ONLY" in out and not (tmp_path / "data/benchmarks").exists()
    monkeypatch.setattr(sys, "argv", ["main.py", "benchmark", "--dataset", "data/history/datasets/ds-bench/", "--id", "cli2"])
    with pytest.raises(SystemExit) as e:
        main_mod.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "signals" in out and (tmp_path / "data/benchmarks/cli2/report.md").exists()
    assert "SYNTHETIC" in out


def test_cli_benchmark_fails_on_invalid_data(tmp_path, monkeypatch, capsys):
    import src.main as main_mod
    cfg = _bench_cfg()
    d = _build(tmp_path, cfg)

    def mutate(df):
        df.loc[10, "low"] = df.loc[10, "high"] + 5
    _corrupt_frozen(d, mutate)
    monkeypatch.setattr(main_mod, "ROOT", tmp_path)
    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: copy.deepcopy(cfg))
    monkeypatch.setattr(sys, "argv", ["main.py", "benchmark", "--dataset", "ds-bench", "--id", "bad"])
    with pytest.raises(SystemExit) as e:
        main_mod.main()
    assert e.value.code == 1
    assert "ABORTED" in capsys.readouterr().out
    assert not (tmp_path / "data/benchmarks/bad/metrics.json").exists()
