"""Evaluator (warm-up, baseline consistency, cache), validator constraints, scoring/ranking determinism."""

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from improvement_data import improvement_cfg, synthetic_frame  # noqa: E402
from src.backtesting import BacktestEngine  # noqa: E402
from src.improvement.candidates import Candidate  # noqa: E402
from src.improvement.evaluator import Evaluator, warmup_bars_for  # noqa: E402
from src.improvement.scoring import ScoringConfig, aggregate, fold_score, rank  # noqa: E402
from src.improvement.splitter import DataSplitter, Slice, SplitConfig  # noqa: E402
from src.improvement.validator import ConstraintConfig, FoldMetrics, validate, validate_holdout  # noqa: E402
from src.strategy.smc_strategy import SMCStrategy  # noqa: E402

DF = synthetic_frame(3000)


# ------------------------------------------------------------------ evaluator
def test_warmup_covers_indicators_and_setup_age():
    cfg = improvement_cfg()
    assert warmup_bars_for(cfg) >= 199 + 60   # EMA200 warm-up + setup_max_age_bars


def test_full_frame_slice_equals_plain_backtest():
    """Slice [0, n) has no warm-up, so the evaluator must reproduce a plain BacktestEngine run."""
    cfg = improvement_cfg()
    ev = Evaluator(cfg, DF)
    r = ev.evaluate(Candidate({}), Slice("all", 0, len(DF), "is"))
    c2 = copy.deepcopy(cfg)
    c2["backtesting"]["close_open_position_at_end"] = True
    bt = BacktestEngine.from_config(c2, SMCStrategy.from_config(c2)).run(DF)
    assert r.metrics["trades"] == bt.metrics["trades"] >= 5
    assert r.metrics["net_profit"] == pytest.approx(bt.metrics["net_profit"])
    assert r.metrics["max_drawdown_pct"] == pytest.approx(bt.metrics["max_drawdown_pct"])
    assert r.warmup_trades_discarded == 0


def test_warmup_trades_are_not_counted_and_slice_starts_at_rebased_equity():
    cfg = improvement_cfg()
    ev = Evaluator(cfg, DF)
    s = Slice("mid", 1500, 2400, "oos")
    r = ev.evaluate(Candidate({}), s)
    start_ts = DF["timestamp"].iloc[s.start]
    assert (pd.to_datetime(r.trades["entry_timestamp"], utc=True) >= start_ts).all()
    assert r.warmup_bars == ev.warmup
    # equity is re-based: return is measured from the equity held at the slice start, not from 10 000
    assert r.metrics["starting_equity"] > 0
    assert r.metrics["bars"] == s.bars
    # a run that is fed the exact same window must produce the same in-slice trades as the warm-up run
    # (point-in-time: history before the slice only shapes structure, it cannot leak the future)
    ev_short = Evaluator(cfg, DF.iloc[: s.end].reset_index(drop=True))
    r2 = ev_short.evaluate(Candidate({}), Slice("mid2", 1500, 2400, "oos"))
    pd.testing.assert_frame_equal(r.trades, r2.trades)


def test_evaluator_cache_is_keyed_by_candidate_and_slice_and_deterministic():
    cfg = improvement_cfg()
    ev = Evaluator(cfg, DF)
    s = Slice("x", 1000, 1800, "oos")
    a = ev.evaluate(Candidate({}), s)
    n = ev.backtests_run
    b = ev.evaluate(Candidate({}), s)
    assert a is b and ev.backtests_run == n
    c = ev.evaluate(Candidate({"strategy.stops.buffer_atr": 0.275}), s)
    assert ev.backtests_run == n + 1 and c is not a
    # a fresh evaluator reproduces identical metrics
    d = Evaluator(cfg, DF).evaluate(Candidate({}), s)
    assert d.metrics == a.metrics and d.trades.equals(a.trades)


def test_evaluator_never_mutates_config():
    cfg = improvement_cfg()
    snap = copy.deepcopy(cfg)
    Evaluator(cfg, DF).evaluate(Candidate({"strategy.stops.buffer_atr": 0.275}), Slice("x", 1000, 1500, "oos"))
    assert cfg == snap


# ------------------------------------------------------------------ validator
def _m(trades=30, r=0.5, net=100.0, dd=3.0):
    return {"trades": trades, "average_r_multiple": r, "net_profit": net, "max_drawdown_pct": dd, "expectancy": net / max(trades, 1)}


def _fm(oos, is_=None, sc=ScoringConfig()):
    is_ = is_ or [_m() for _ in oos]
    scores = [fold_score(m, sc) for m in oos]
    return FoldMetrics(is_, oos, scores, aggregate(scores))


CFG = ConstraintConfig(min_trades_before_change=100, min_trades_per_fold=8, oos_is_ratio_min=0.5,
                       min_improvement_pct=5.0, max_drawdown_pct_limit=10.0)
BASE = _fm([_m(r=0.4)] * 4)


def test_validator_passes_a_clearly_better_candidate():
    good = _fm([_m(r=0.6)] * 4)
    v = validate(good, BASE, CFG, neighbour_medians=[good.agg["median"] * 0.9, good.agg["median"] * 0.95])
    assert v.passed and v.reasons == [] and all(v.checks.values())


@pytest.mark.parametrize("oos, is_, neigh, code", [
    ([_m(trades=20, r=0.6)] * 4, None, None, "C1"),                         # 80 total < 100
    ([_m(r=0.6)] * 3 + [_m(trades=5, r=0.6)], None, None, "C2"),
    ([_m(r=0.6)] * 3 + [_m(r=0.6, net=-1.0)], None, None, "C3"),
    ([_m(r=0.6)] * 3 + [_m(r=0.1)], None, None, "C4"),                      # min below baseline min
    ([_m(r=0.6)] * 4, [_m(r=2.0)] * 4, None, "C5"),                          # OOS/IS = 0.3
    ([_m(r=0.6)] * 3 + [_m(r=0.6, dd=12.0)], None, None, "C6"),
    ([_m(r=0.41)] * 4, None, None, "C7"),                                     # +2.5% only
    ([_m(r=0.6)] * 4, None, [BASE.agg["median"] - 1, BASE.agg["median"] - 1], "C8"),
])
def test_each_constraint_rejects(oos, is_, neigh, code):
    v = validate(_fm(oos, is_), BASE, CFG, neighbour_medians=neigh)
    assert not v.passed
    assert any(r.startswith(code) for r in v.reasons), v.reasons
    assert v.checks[next(k for k in v.checks if k.startswith(code))] is False


def test_neighbourhood_single_worse_neighbour_is_tolerated():
    good = _fm([_m(r=0.6)] * 4)
    v = validate(good, BASE, CFG, neighbour_medians=[BASE.agg["median"] - 1, good.agg["median"]])
    assert v.checks["C8_neighbourhood"]


def test_require_positive_all_folds_false_allows_baseline_matching():
    cfg = ConstraintConfig(min_trades_before_change=10, min_trades_per_fold=1, require_positive_all_folds=False,
                           min_improvement_pct=0.0, max_drawdown_pct_limit=50)
    base = _fm([_m(r=0.4, net=-5.0)] * 4)
    cand = _fm([_m(r=0.6, net=-5.0)] * 4)
    assert validate(cand, base, cfg).checks["C3_positive_folds"]


def test_holdout_validation():
    fm = _fm([_m(r=0.6)] * 4)
    assert validate_holdout(_m(r=0.5, net=50), fm, CFG).passed
    assert not validate_holdout(_m(r=0.5, net=-1), fm, CFG).passed          # H1
    assert not validate_holdout(_m(r=0.1, net=10), fm, CFG).passed          # H2: 0.1 < 0.5 * 0.6


def test_constraint_config_defaults_use_risk_max_drawdown():
    cfg = improvement_cfg()
    c = ConstraintConfig.from_config(cfg)
    assert c.max_drawdown_pct_limit == cfg["risk"]["max_drawdown_pct"]
    cfg["improvement"]["constraints"]["max_drawdown_pct_limit"] = 4.0
    assert ConstraintConfig.from_config(cfg).max_drawdown_pct_limit == 4.0


# ------------------------------------------------------------ scoring / rank
def test_fold_score_formula_and_aggregate():
    sc = ScoringConfig(dd_penalty=0.05)
    assert fold_score(_m(trades=16, r=0.5, dd=2.0), sc) == pytest.approx(0.5 * 4 - 0.1)
    assert fold_score(_m(trades=0), sc) == 0.0
    assert aggregate([3.0, 1.0, 2.0]) == {"median": 2.0, "min": 1.0, "mean": 2.0}


def test_rank_is_deterministic_and_prefers_pass_then_score_then_simplicity():
    rows = [
        {"candidate_id": "b", "passed": True, "oos_score_median": 1.0, "params_changed": 1, "rel_change_pct": 5.0},
        {"candidate_id": "a", "passed": True, "oos_score_median": 1.0, "params_changed": 1, "rel_change_pct": 2.0},
        {"candidate_id": "c", "passed": False, "oos_score_median": 9.0, "params_changed": 1, "rel_change_pct": 0.0},
        {"candidate_id": "d", "passed": True, "oos_score_median": 2.0, "params_changed": 2, "rel_change_pct": 0.0},
    ]
    r1 = rank([dict(x) for x in rows])
    r2 = rank([dict(x) for x in reversed(rows)])
    assert [x["candidate_id"] for x in r1] == ["d", "a", "b", "c"] == [x["candidate_id"] for x in r2]
    assert [x["rank"] for x in r1] == [1, 2, 3, 4]
