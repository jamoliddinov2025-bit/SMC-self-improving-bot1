"""End-to-end runner, report files, proposal show/apply safety, enabled=false refusal, repeatability, and the
"nothing else changes" guarantees (config.yaml, src/, paper state)."""

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from improvement_data import improvement_cfg, synthetic_frame  # noqa: E402
from src.data import CSVMarketData  # noqa: E402
from src.execution.state_store import config_hash  # noqa: E402
import src.improvement.apply as proposals  # noqa: E402
from src.improvement.report import SYNTHETIC_BANNER  # noqa: E402
from src.improvement.runner import ImprovementDisabled, ImprovementRunner  # noqa: E402
from src.main import CONFIG_PATH, load_config, run_improve  # noqa: E402

DF = synthetic_frame()
RELAXED = {"improvement.constraints.oos_is_ratio_min": -10, "improvement.max_parameter_change_pct": 25, "improvement.constraints.require_positive_all_folds": False,
           "improvement.constraints.min_improvement_pct": 1,
           "improvement.parameters": {"strategy.filters.ema_extension.enabled": {},
                                      "strategy.filters.ema_extension.atr_multiple": {},
                                      "strategy.stops.buffer_atr": {}}}


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def relaxed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("imp")
    cfg = improvement_cfg(**RELAXED)
    res = ImprovementRunner(cfg, data_root=root, run_id="r1", data_label="synthetic", synthetic=True).run(DF)
    return root, cfg, res


# ------------------------------------------------------------------ guards
def test_disabled_refuses_and_writes_nothing(tmp_path):
    cfg = improvement_cfg(**{"improvement.enabled": False})
    with pytest.raises(ImprovementDisabled):
        ImprovementRunner(cfg, data_root=tmp_path, run_id="x").run(DF)
    assert not (tmp_path / "data").exists()


def test_insufficient_data_aborts_clearly_without_candidates_or_proposals(tmp_path):
    cfg = improvement_cfg()   # sample thresholds are too high for the 200-bar sample
    df = CSVMarketData(ROOT / "data/sample").get_ohlcv("BTC/USDT", "15m")
    res = ImprovementRunner(cfg, data_root=tmp_path, run_id="abort", data_label="data/sample/BTCUSDT_15m.csv",
                            synthetic=True).run(df)
    assert res.aborted and "insufficient data" in res.abort_reason and "200 bars" in res.abort_reason
    assert res.ranking == [] and res.proposals == []
    out = tmp_path / "data/improvement/abort"
    assert sorted(p.name for p in out.iterdir()) == ["report.md", "summary.json"]
    text = (out / "report.md").read_text()
    assert "ABORTED" in text and "SYNTHETIC DATA" in text and "No change has been applied" in text
    assert json.load(open(out / "summary.json"))["aborted"] is True


def test_sample_cli_run_aborts_on_200_bars(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["improvement"]["enabled"] = True
    cfg["improvement"]["results_directory"] = str(tmp_path / "imp")
    res = run_improve(cfg, run_id="cli")
    assert res.aborted and "insufficient data" in res.abort_reason
    assert (tmp_path / "imp" / "cli" / "report.md").exists()


def test_skipped_parameters_reported_in_summary_and_report(tmp_path):
    cfg = improvement_cfg(**{"improvement.data.min_trades_per_fold": 500})   # aborts quickly, header still written
    ImprovementRunner(cfg, data_root=tmp_path, run_id="sk", synthetic=True).run(DF)
    summary = json.load(open(tmp_path / "data/improvement/sk/summary.json"))
    assert "strategy.entry.reentry_cooldown_bars" in summary["skipped_parameters"]


def test_baseline_guard_uses_min_trades_per_fold(tmp_path):
    cfg = improvement_cfg(**{"improvement.data.min_trades_per_fold": 500})
    res = ImprovementRunner(cfg, data_root=tmp_path, run_id="g", synthetic=True).run(DF)
    assert res.aborted and "per fold" in res.abort_reason


# ------------------------------------------------------------ full pipeline
def test_pipeline_produces_ranking_proposals_and_files(relaxed_run):
    root, cfg, res = relaxed_run
    assert not res.aborted
    out = root / "data/improvement/r1"
    for name in ("report.md", "ranking.csv", "folds.csv", "summary.json"):
        assert (out / name).exists()
    ranking = pd.read_csv(out / "ranking.csv")
    assert (ranking["candidate_id"] == "baseline").sum() == 1
    assert ranking["rank"].tolist() == list(range(1, len(ranking) + 1))
    assert res.summary["survivors"] >= 1 and res.proposals
    assert res.summary["search"]["method"] == "coordinate_single_parameter"
    # every evaluated candidate value is a declared grid value
    from src.improvement.space import ParameterSpace
    sp = ParameterSpace(cfg)
    for p in res.proposals:
        for path, v in p["parameter_overlay"].items():
            assert v in sp.specs[path].grid()
    folds = pd.read_csv(out / "folds.csv")
    assert set(folds["role"]) == {"is", "oos", "holdout"}
    assert folds[folds["role"] == "holdout"]["candidate_id"].nunique() <= cfg["improvement"]["top_n_for_holdout"] + 1
    assert sorted(p.name for p in (out / "proposals").iterdir()) == [f"P-{i}.yaml" for i in range(1, len(res.proposals) + 1)]


def test_holdout_is_sealed_from_search(relaxed_run):
    """Ranking/pass-fail must be identical when the holdout rows are shocked: only holdout metrics may change."""
    root, cfg, res = relaxed_run
    shocked = DF.copy()
    h = res.plan.holdout
    for col in ("open", "high", "low", "close"):
        shocked.loc[h.start:, col] *= 1.0 + 0.3 * ((shocked.index[h.start:] % 7) / 7.0)
    res2 = ImprovementRunner(cfg, data_root=root / "shock", run_id="r2", synthetic=True).run(shocked)
    keys = ["candidate_id", "oos_score_median", "oos_score_min", "passed", "rank", "reasons"]
    a = [{k: r[k] for k in keys} for r in res.ranking]
    b = [{k: r[k] for k in keys} for r in res2.ranking]
    assert a == b
    assert [p["parameter_overlay"] for p in res.proposals] == [p["parameter_overlay"] for p in res2.proposals]


def test_baseline_goes_through_same_pipeline(relaxed_run):
    root, cfg, res = relaxed_run
    folds = pd.read_csv(root / "data/improvement/r1/folds.csv")
    base = folds[folds["candidate_id"] == "baseline"]
    assert sorted(base[base["role"] != "holdout"]["slice"].tolist()) == sorted(
        [f"is_{k}" for k in range(1, 5)] + [f"oos_{k}" for k in range(1, 5)])
    assert (base["warmup_trades_discarded"] >= 0).all()
    assert next(r for r in res.ranking if r["candidate_id"] == "baseline")["verdict"] == "baseline"


def test_proposal_content(relaxed_run):
    root, cfg, res = relaxed_run
    p = yaml.safe_load((root / "data/improvement/r1/proposals/P-1.yaml").read_text())
    for key in ("proposal_id", "status", "baseline_config_hash", "baseline_values", "parameter_overlay", "evidence",
                "fold_metrics", "holdout_result", "reason"):
        assert key in p, key
    assert p["proposal_id"] == "P-1" and p["baseline_config_hash"] == config_hash(cfg)
    assert p["status"] in ("recommended_pending_human_review", "not_recommended_failed_holdout")
    assert len(p["fold_metrics"]) == 4 and {"is", "oos", "oos_score"} <= set(p["fold_metrics"][0])
    assert p["holdout_result"]["verdict"] in ("pass", "fail") and "metrics" in p["holdout_result"]
    assert all(p["evidence"]["constraint_checks"].values())
    assert len(p["parameter_overlay"]) == 1


def test_report_has_synthetic_banner_and_no_change_footer(relaxed_run):
    root, cfg, res = relaxed_run
    text = (root / "data/improvement/r1/report.md").read_text()
    assert SYNTHETIC_BANNER.strip() in text
    assert "No change has been applied" in text and "## Ranked candidates" in text and "## Baseline" in text


def test_real_data_label_has_no_synthetic_banner(tmp_path):
    cfg = improvement_cfg(**RELAXED, **{"improvement.data.min_trades_per_fold": 500})   # abort quickly
    res = ImprovementRunner(cfg, data_root=tmp_path, run_id="nb", data_label="data/history/BTCUSDT_15m.csv").run(DF)
    assert "SYNTHETIC" not in (tmp_path / "data/improvement/nb/report.md").read_text()


def test_dry_run_writes_nothing(tmp_path):
    cfg = improvement_cfg(**RELAXED, **{"improvement.data.min_trades_per_fold": 500})
    res = ImprovementRunner(cfg, data_root=tmp_path, run_id="dry", dry_run=True, synthetic=True).run(DF)
    assert res.aborted and not (tmp_path / "data").exists()


def test_max_candidates_limits_search(tmp_path):
    cfg = improvement_cfg(**RELAXED)
    res = ImprovementRunner(cfg, data_root=tmp_path, run_id="mc", max_candidates=2, synthetic=True, dry_run=True).run(DF)
    assert res.summary["search"]["candidates"] == 2 and len(res.ranking) == 3


# ------------------------------------------------------- repeatability
def test_repeated_runs_are_byte_identical(tmp_path):
    cfg = improvement_cfg(**RELAXED)
    outs = []
    for i in range(2):
        r = ImprovementRunner(cfg, data_root=tmp_path / str(i), run_id="rep", synthetic=True).run(DF)
        d = tmp_path / str(i) / "data/improvement/rep"
        outs.append({
            "ranking": (d / "ranking.csv").read_bytes(), "folds": (d / "folds.csv").read_bytes(),
            "proposals": {p.name: {k: v for k, v in yaml.safe_load(p.read_text()).items() if k != "created_utc"}
                          for p in (d / "proposals").iterdir()},
            "report": "\n".join(l for l in (d / "report.md").read_text().splitlines()),
            "summary": {k: v for k, v in json.load(open(d / "summary.json")).items() if k != "created_utc"},
        })
    assert outs[0] == outs[1]


# --------------------------------------------------------- safety guarantees
def test_run_leaves_config_yaml_src_and_paper_state_untouched(tmp_path):
    cfg_before = CONFIG_PATH.read_bytes()
    src_before = _tree_hash(ROOT / "src")
    paper_dir = tmp_path / "data/paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "state.json").write_text('{"schema_version": 1, "sentinel": true}')
    state_before = (paper_dir / "state.json").read_bytes()
    cfg = improvement_cfg(**RELAXED)
    cfg["paper"]["state_directory"] = "data/paper/"
    ImprovementRunner(cfg, data_root=tmp_path, run_id="safe", synthetic=True).run(DF)
    assert CONFIG_PATH.read_bytes() == cfg_before
    assert _tree_hash(ROOT / "src") == src_before
    assert (paper_dir / "state.json").read_bytes() == state_before
    assert sorted(p.name for p in paper_dir.iterdir()) == ["state.json"]
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert all(w.startswith("data/improvement/safe/") or w == "data/paper/state.json" for w in written)


def test_improvement_never_imports_paper_trader():
    import ast
    for f in (ROOT / "src/improvement").glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any("paper_trader" in n or "PaperTrader" in n for n in names), (f.name, names)


# ------------------------------------------------------- proposal show/apply
@pytest.fixture
def proposal_env(relaxed_run, tmp_path):
    """Copy the run into a scratch tree with its own config.yaml so apply() can be exercised safely."""
    root, cfg, res = relaxed_run
    results = tmp_path / "improvement"
    shutil.copytree(root / "data/improvement", results)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return results, cfg_path, res


def _files(d: Path):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def test_proposal_show_writes_nothing(proposal_env):
    results, cfg_path, res = proposal_env
    before = _files(results.parent)
    hashes = {p: Path(results.parent / p).read_bytes() for p in before}
    text = proposals.show(results, cfg_path, "P-1")
    assert "P-1" in text and "Nothing has been written" in text and "Diff vs current config.yaml" in text
    assert _files(results.parent) == before
    assert all(Path(results.parent / p).read_bytes() == b for p, b in hashes.items())


def test_proposal_apply_requires_exact_confirmation(proposal_env):
    results, cfg_path, res = proposal_env
    before = _files(results.parent)
    for confirm in (None, "", "P-2", "p-1", "yes"):
        with pytest.raises(proposals.ProposalError):
            proposals.apply(results, cfg_path, "P-1", confirm)
    assert _files(results.parent) == before


def test_proposal_apply_writes_only_proposed_file_and_never_config_yaml(proposal_env):
    results, cfg_path, res = proposal_env
    cfg_bytes = cfg_path.read_bytes()
    before = set(_files(results.parent))
    target = proposals.apply(results, cfg_path, "P-1", "P-1")
    assert target == cfg_path.parent / "config.proposed.P-1.yaml" and target.exists()
    assert cfg_path.read_bytes() == cfg_bytes
    new = set(_files(results.parent)) - before
    assert new == {"config/config.proposed.P-1.yaml"}
    proposed = yaml.safe_load(target.read_text())
    (path, value), = res.proposals[0]["parameter_overlay"].items()
    d = proposed
    for part in path.split("."):
        d = d[part]
    assert d == value
    assert "NOT active" in target.read_text().splitlines()[2]
    p = yaml.safe_load((results / "r1/proposals/P-1.yaml").read_text())
    assert p["status"] == "applied_to_proposed_file"


def test_proposal_apply_refuses_stale_config_or_unknown_id(proposal_env):
    results, cfg_path, res = proposal_env
    with pytest.raises(proposals.ProposalError):
        proposals.apply(results, cfg_path, "P-99", "P-99")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["risk"]["risk_per_trade_pct"] = 0.5
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with pytest.raises(proposals.ProposalError, match="hash mismatch"):
        proposals.apply(results, cfg_path, "P-1", "P-1")
    assert not (cfg_path.parent / "config.proposed.P-1.yaml").exists()
