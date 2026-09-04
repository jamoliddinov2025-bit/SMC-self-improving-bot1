"""ParameterSpace: whitelist, bounds/steps, change-% cap, max changed params, invariants; candidate determinism;
DataSplitter: sealed holdout, anchored folds, guards."""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from improvement_data import improvement_cfg  # noqa: E402
from src.improvement.candidates import Candidate, CandidateGenerator, SingleParameterStage  # noqa: E402
from src.improvement.space import WHITELIST, ParameterSpace, SpaceError, apply_overlay, get_path  # noqa: E402
from src.improvement.splitter import DataSplitter, SplitConfig  # noqa: E402
from src.main import load_config  # noqa: E402


# ------------------------------------------------------------------ whitelist
def test_whitelist_contains_no_risk_fee_market_or_usdtd_keys():
    for k in WHITELIST:
        assert k.startswith("strategy."), k
        assert not any(k.startswith(p) for p in ("risk.", "execution.", "market.", "usdtd.", "indicators."))
    assert "risk.min_risk_reward" not in WHITELIST


@pytest.mark.parametrize("bad", ["risk.risk_per_trade_pct", "risk.min_risk_reward", "risk.max_drawdown_pct",
                                 "execution.paper_fee_pct", "execution.slippage_pct", "risk.starting_balance",
                                 "market.symbol", "market.timeframe", "usdtd.slope_threshold_pct",
                                 "strategy.structure.pivot_strength", "strategy.entry.nonexistent"])
def test_non_whitelisted_parameter_rejected(bad):
    cfg = improvement_cfg(**{"improvement.parameters": {bad: {}}})
    with pytest.raises(SpaceError):
        ParameterSpace(cfg)


def test_overlay_with_frozen_key_is_inadmissible():
    sp = ParameterSpace(improvement_cfg())
    assert any("not tunable" in p for p in sp.validate_overlay({"risk.risk_per_trade_pct": 2.0}))
    assert any("not tunable" in p for p in sp.validate_overlay({"execution.paper_fee_pct": 0.0}))


# ------------------------------------------------------------- bounds / steps
def test_bounds_and_steps_from_whitelist_and_grid():
    sp = ParameterSpace(improvement_cfg())
    spec = sp.specs["strategy.stops.buffer_atr"]
    assert (spec.min, spec.max, spec.step) == (0.1, 0.5, 0.05)
    assert spec.grid() == pytest.approx([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    assert sp.specs["strategy.entry.setup_max_age_bars"].grid() == list(range(20, 121, 10))
    assert sp.specs["strategy.targets.mode"].grid() == ["structure", "fixed_rr", "hybrid"]


def test_config_may_narrow_but_not_widen_bounds():
    narrowed = improvement_cfg(**{"improvement.parameters": {"strategy.stops.buffer_atr": {"min": 0.15, "max": 0.35}}})
    sp = ParameterSpace(narrowed)
    assert list(sp.specs) == ["strategy.stops.buffer_atr"]
    assert (sp.specs["strategy.stops.buffer_atr"].min, sp.specs["strategy.stops.buffer_atr"].max) == (0.15, 0.35)
    with pytest.raises(SpaceError):
        ParameterSpace(improvement_cfg(**{"improvement.parameters": {"strategy.stops.buffer_atr": {"max": 0.9}}}))


def test_out_of_bounds_value_rejected():
    sp = ParameterSpace(improvement_cfg(**{"improvement.max_parameter_change_pct": 1000}))
    assert sp.validate_overlay({"strategy.stops.buffer_atr": 0.9})
    assert sp.validate_overlay({"strategy.targets.mode": "martingale"})
    assert sp.validate_overlay({"strategy.stops.buffer_atr": 0.3}) == []


# ------------------------------------------------------- change percentage
def test_change_pct_cap_enforced_in_overlay_validation():
    sp = ParameterSpace(improvement_cfg())      # baseline buffer_atr 0.25, cap 10%
    assert sp.validate_overlay({"strategy.stops.buffer_atr": 0.3})        # +20% -> rejected
    assert sp.validate_overlay({"strategy.stops.buffer_atr": 0.27}) == [] # +8% ok (manual overlay, on/off grid)


def test_candidates_are_grid_values_only_and_within_cap():
    """(a) no off-grid cap-probe values, (b) grid neighbours beyond the cap are skipped."""
    sp = ParameterSpace(improvement_cfg())
    for path, spec in sp.specs.items():
        grid = spec.grid()
        for v in sp.admissible_values(path):
            assert v in grid, (path, v)
            pct = sp.change_pct(path, v)
            assert pct is None or pct <= sp.max_change_pct + 1e-9, (path, v, pct)
    # buffer_atr 0.25 step 0.05: both neighbours are +/-20% -> skipped, and 0.225/0.275 are never invented
    assert sp.admissible_values("strategy.stops.buffer_atr") == []
    assert sp.admissible_values("strategy.exits.max_bars_in_trade") == []       # 72/120 are +/-25%
    assert sp.admissible_values("strategy.entry.reentry_cooldown_bars") == []   # int: 0 or 6 vs 3
    for c in CandidateGenerator(sp).generate():
        for path, v in c.overlay.items():
            assert v in sp.specs[path].grid(), (path, v)
            assert v not in (0.225, 0.275, 87, 105)


def test_parameters_can_have_zero_legal_candidates_and_are_reported():
    """(c) zero legal candidates is legitimate and transparent."""
    sp = ParameterSpace(improvement_cfg())
    skipped = sp.skipped_parameters()
    assert "strategy.entry.reentry_cooldown_bars" in skipped
    assert "exceeds max_parameter_change_pct" in skipped["strategy.entry.reentry_cooldown_bars"]
    assert "strategy.targets.mode" not in skipped and "strategy.filters.volume.enabled" not in skipped
    assert sp.neighbours("strategy.stops.buffer_atr", 0.25) == []
    # once the cap allows the grid step the same grid neighbours become candidates and leave the skipped list
    wide = ParameterSpace(improvement_cfg(**{"improvement.max_parameter_change_pct": 25}))
    assert wide.admissible_values("strategy.stops.buffer_atr") == pytest.approx([0.2, 0.3])
    assert wide.admissible_values("strategy.exits.max_bars_in_trade") == [72, 120]
    assert "strategy.stops.buffer_atr" not in wide.skipped_parameters()
    assert wide.neighbours("strategy.stops.buffer_atr", 0.3) == [0.25]


def test_change_pct_cap_is_configurable():
    sp = ParameterSpace(improvement_cfg(**{"improvement.max_parameter_change_pct": 25}))
    assert sp.validate_overlay({"strategy.stops.buffer_atr": 0.3}) == []
    assert 0.3 in sp.admissible_values("strategy.stops.buffer_atr")


# ---------------------------------------------------------- max params changed
def test_max_two_changed_parameters_per_proposal():
    sp = ParameterSpace(improvement_cfg())
    two = {"strategy.stops.buffer_atr": 0.27, "strategy.filters.volume.enabled": False}
    three = {**two, "strategy.targets.mode": "hybrid"}
    assert sp.validate_overlay(two) == []
    assert any("max_params_changed_per_proposal" in p for p in sp.validate_overlay(three))
    # unchanged values do not count
    assert sp.validate_overlay({**two, "strategy.targets.mode": "structure"}) == []


# ----------------------------------------------------------------- invariants
def test_invariants_min_stop_le_max_stop_and_fixed_rr_ge_min_rr():
    cfg = improvement_cfg(**{"improvement.max_parameter_change_pct": 1000})
    sp = ParameterSpace(cfg)
    assert sp.validate_overlay({"strategy.stops.min_stop_atr": 1.0, "strategy.stops.max_stop_atr": 2.0}) == []
    # baseline max_stop_atr 3.0: raising min_stop_atr above it must fail via the invariant, not just bounds
    cfg2 = copy.deepcopy(cfg)
    cfg2["strategy"]["stops"]["max_stop_atr"] = 2.0
    sp2 = ParameterSpace(cfg2)
    cfg2b = copy.deepcopy(cfg2)
    cfg2b["strategy"]["stops"]["min_stop_atr"] = 0.9
    # min 0.9 with max 2.0 is fine; an overlay pushing max below min is caught by the invariant
    sp3 = ParameterSpace(cfg2b)
    sp3.specs["strategy.stops.max_stop_atr"]  # exists
    msgs = sp2.validate_overlay({"strategy.stops.min_stop_atr": 1.0, "strategy.stops.max_stop_atr": 2.0})
    assert msgs == []
    fake = dict(sp.baseline)
    fake.update({"strategy.stops.min_stop_atr": 2.5, "strategy.stops.max_stop_atr": 2.0})
    assert any("min_stop_atr" in m for m in sp.invariants(fake))
    # fixed_rr below risk.min_risk_reward (2.0) is never admissible
    assert any("min_risk_reward" in p for p in sp.validate_overlay({"strategy.targets.fixed_rr": 1.5}))
    assert 1.5 not in sp.admissible_values("strategy.targets.fixed_rr")
    assert sp.validate_overlay({"strategy.targets.fixed_rr": 2.5}) == []


def test_apply_overlay_does_not_mutate_input():
    cfg = improvement_cfg()
    out = apply_overlay(cfg, {"strategy.stops.buffer_atr": 0.27})
    assert get_path(out, "strategy.stops.buffer_atr") == 0.27
    assert get_path(cfg, "strategy.stops.buffer_atr") == 0.25


# ------------------------------------------------------- candidate generation
def test_candidate_generation_is_deterministic_single_parameter_and_admissible():
    """(d) byte-identical candidate lists across independent generations."""
    cfg = improvement_cfg()
    a = CandidateGenerator(ParameterSpace(cfg)).generate()
    b = CandidateGenerator(ParameterSpace(copy.deepcopy(cfg))).generate()
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.overlay for c in a] == [c.overlay for c in b]
    assert repr([c.overlay for c in a]) == repr([c.overlay for c in b])
    assert a[0].is_baseline and a[0].id == "baseline"
    sp = ParameterSpace(cfg)
    for c in a[1:]:
        assert len(c.overlay) == 1 and c.stage == "single"
        assert sp.validate_overlay(c.overlay) == []
    assert len({c.id for c in a}) == len(a)
    # whitelist order then ascending values
    paths = [next(iter(c.overlay)) for c in a[1:]]
    order = list(sp.specs)
    assert paths == sorted(paths, key=order.index)


def test_max_candidates_truncates_deterministically():
    sp = ParameterSpace(improvement_cfg(**{"improvement.max_parameter_change_pct": 25}))
    full = CandidateGenerator(sp).generate()
    part = CandidateGenerator(sp, max_candidates=5).generate()
    assert len(full) > 6
    assert len(part) == 6 and [c.id for c in part] == [c.id for c in full[:6]]


def test_candidate_id_stable_and_label():
    c = Candidate({"strategy.stops.buffer_atr": 0.27})
    assert c.id == Candidate({"strategy.stops.buffer_atr": 0.27}).id and len(c.id) == 12
    assert c.label() == "stops.buffer_atr=0.27"


# ------------------------------------------------------------------ splitter
def test_holdout_is_final_20pct_and_sealed_from_folds():
    plan = DataSplitter(SplitConfig()).plan(6000)
    assert (plan.holdout.start, plan.holdout.end) == (4800, 6000)
    assert plan.development.end == plan.holdout.start
    for f in plan.folds:
        assert f.test.end <= plan.holdout.start and f.train.end <= plan.holdout.start


def test_anchored_fold_boundaries():
    plan = DataSplitter(SplitConfig()).plan(6000)
    assert [(f.train.start, f.train.end, f.test.start, f.test.end) for f in plan.folds] == [
        (0, 960, 960, 1920), (0, 1920, 1920, 2880), (0, 2880, 2880, 3840), (0, 3840, 3840, 4800)]
    assert plan.folds[-1].test.end == plan.development.end
    d = plan.to_dict()
    assert d["holdout"]["bars"] == 1200 and d["folds"][0]["test"]["bars"] == 960


def test_rolling_folds_option_and_guards():
    plan = DataSplitter(SplitConfig(anchored=False)).plan(6000)
    assert plan.folds[1].train.start > 0
    with pytest.raises(ValueError):
        DataSplitter(SplitConfig(folds=10)).plan(30)
    with pytest.raises(ValueError):
        SplitConfig.from_config({"improvement": {"data": {"holdout_pct": 60}}})


def test_split_config_defaults_from_yaml():
    c = SplitConfig.from_config(load_config())
    assert (c.holdout_pct, c.folds, c.oos_pct_per_fold, c.anchored) == (20.0, 4, 20.0, True)
