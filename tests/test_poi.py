"""POI discovery, mitigation, selection, deeper-POI search and confluence score."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from smc_scenarios import contexts, pullback_scenario  # noqa: E402
from src.strategy.poi import (  # noqa: E402
    FVG,
    ORDER_BLOCK,
    POI,
    Setup,
    build_setup,
    confluence_score,
    deeper_pois,
    mitigate,
    select_poi,
)

PRIO = [ORDER_BLOCK, FVG]


def setup_at(bar=10, atr=8.0, max_mult=3.0, vol=None):
    ctx = contexts(pullback_scenario())[bar][2]
    bos = ctx.smc.bos_events[-1]
    return build_setup(ctx.smc, bos, atr, max_mult, PRIO, vol), ctx


def test_order_block_from_trigger_bos():
    setup, _ = setup_at()
    obs = [p for p in setup.pois if p.kind == ORDER_BLOCK]
    assert len(obs) == 1 and (obs[0].low, obs[0].high) == (98.0, 102.0)
    assert obs[0].created_index == setup.trigger_index == 7
    assert setup.impulse_start_index == 5


def test_fvg_from_impulse_only():
    setup, _ = setup_at()
    fvgs = [p for p in setup.pois if p.kind == FVG]
    assert len(fvgs) == 0   # the FVG [102,108] is detected at bar 8 > bos.detected_index 7 -> not part of the impulse
    # extend the impulse: a bos detected later would include it. Emulate with a fake BOS-like window:
    ctx = contexts(pullback_scenario())[10][2]
    bos = ctx.smc.bos_events[-1]
    fake = type(bos)(**{**bos.__dict__, "detected_index": 8})
    s2 = build_setup(ctx.smc, fake, 8.0, 3.0, PRIO, None)
    assert [p.kind for p in s2.pois] == [FVG]  # OB is tied to the real bos object, FVG [102,108] now inside window
    assert (s2.pois[0].low, s2.pois[0].high) == (102.0, 108.0)


def test_priority_order_and_dedup():
    setup, _ = setup_at()
    setup.pois.append(POI(ORDER_BLOCK, 98.0, 102.0, 7, None))  # duplicate zone
    s = Setup(setup.trigger_bos, 7, 5, setup.pois)
    assert select_poi(s, low=100.0, close=101.0, priority=PRIO).kind == ORDER_BLOCK
    fvg_first = select_poi(Setup(setup.trigger_bos, 7, 5, [POI(FVG, 99, 103, 8, None)] + setup.pois),
                           100.0, 101.0, [FVG, ORDER_BLOCK])
    assert fvg_first.kind == FVG


def test_zone_width_sanity():
    setup, _ = setup_at(atr=1.0, max_mult=3.0)   # OB height 4 > 3 ATR -> dropped
    assert setup.pois == []


def test_mitigation_by_close_not_wick():
    setup, _ = setup_at()
    poi = setup.pois[0]
    mitigate(setup, close=98.5)       # closed inside zone -> still valid
    assert not poi.mitigated
    mitigate(setup, close=97.9)       # closed below zone.low -> mitigated
    assert poi.mitigated and setup.valid_pois() == []


def test_contains_touch_semantics():
    p = POI(ORDER_BLOCK, 98.0, 102.0, 7, None)
    assert p.contains_touch(low=101.0, close=104.0)     # wick into zone, close above
    assert p.contains_touch(low=97.0, close=99.0)       # through zone, closed inside
    assert not p.contains_touch(low=103.0, close=105.0) # never touched
    assert not p.contains_touch(low=95.0, close=97.0)   # closed below


def test_deeper_pois_search_all_kinds():
    ref = POI(ORDER_BLOCK, 100.0, 104.0, 7, None)
    pois = [ref, POI(FVG, 96.0, 99.0, 8, None), POI(ORDER_BLOCK, 90.0, 94.0, 9, None),
            POI(FVG, 102.0, 106.0, 8, None), POI(FVG, 80.0, 85.0, 8, None, mitigated=True)]
    s = Setup(None, 7, 5, pois)
    deeper = deeper_pois(s, ref)
    assert {(p.kind, p.low) for p in deeper} == {(FVG, 96.0), (ORDER_BLOCK, 90.0)}


def test_overlaps_level():
    p = POI(FVG, 96.0, 99.0, 8, None)
    assert p.overlaps_level(95.8, tolerance=0.25) and not p.overlaps_level(95.0, tolerance=0.25)


def test_confluence_score():
    s = Setup(None, 7, 5, [], bos_volume_ratio=1.5)
    ob = POI(ORDER_BLOCK, 98, 102, 7, None)
    fvg = POI(FVG, 99, 103, 8, None, overlaps_ob=True)
    plain_fvg = POI(FVG, 99, 103, 8, None)
    assert confluence_score(ob, s, has_sweep=False, bos_vol_ratio_min=1.2) == 2      # OB + BOS volume
    assert confluence_score(ob, s, has_sweep=True, bos_vol_ratio_min=1.2) == 3
    assert confluence_score(fvg, s, has_sweep=False, bos_vol_ratio_min=1.2) == 2     # overlap + volume
    assert confluence_score(plain_fvg, Setup(None, 7, 5, []), False, 1.2) == 0
    assert confluence_score(ob, Setup(None, 7, 5, [], bos_volume_ratio=1.0), False, 1.2) == 1


def test_bos_volume_ratio_recorded():
    setup, _ = setup_at(vol=1.5)
    assert setup.bos_volume_ratio == 1.5
