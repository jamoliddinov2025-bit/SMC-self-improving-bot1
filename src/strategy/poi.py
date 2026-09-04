"""Points of interest (POI) derived from the SMC engine's output.

A POI is a price zone where the strategy is willing to buy a pullback:
  * ORDER_BLOCK : the bullish OrderBlock created by the trigger BOS  (zone = [ob.low, ob.high])
  * FVG         : a bullish FairValueGap formed during the impulse that produced
                  the trigger BOS, i.e. detected_index in (impulse_start, bos.detected_index]

Mitigation: a POI is invalid once any candle CLOSES below zone.low. Wicks do not mitigate.
Confluence score (0-4): OB present (+1), FVG overlaps the OB (+1), bullish sweep on the
trigger candle (+1), trigger BOS candle volume ratio >= threshold (+1).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.strategy.smc_types import BULLISH, BOSEvent, FairValueGap, OrderBlock, SMCResult

ORDER_BLOCK = "order_block"
FVG = "fvg"


@dataclass
class POI:
    kind: str
    low: float
    high: float
    created_index: int            # detected_index of the underlying event
    timestamp: object
    mitigated: bool = False
    skipped: bool = False         # skipped by the risk-off "wait for the dip" rule
    entries: int = 0
    overlaps_ob: bool = False

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    def contains_touch(self, low: float, close: float) -> bool:
        """Candle traded into the zone and did not close below it."""
        return low <= self.high and close >= self.low

    def overlaps_level(self, level: float, tolerance: float) -> bool:
        return self.low - tolerance <= level <= self.high + tolerance


@dataclass
class Setup:
    trigger_bos: BOSEvent
    trigger_index: int                     # == bos.detected_index
    impulse_start_index: int               # bar index where the impulse is deemed to start
    pois: List[POI] = field(default_factory=list)
    bos_volume_ratio: Optional[float] = None
    entries: int = 0

    def valid_pois(self) -> List[POI]:
        return [p for p in self.pois if not p.mitigated]


def _impulse_start(smc: SMCResult, bos: BOSEvent) -> int:
    """Start of the impulse = the most recent confirmed swing low BEFORE the break candle."""
    best = -1
    for s in smc.swing_lows:
        if s.index < bos.detected_index and s.detected_index <= bos.detected_index and s.index > best:
            best = s.index
    return best


def build_setup(smc: SMCResult, bos: BOSEvent, atr: float, poi_max_atr_multiple: float,
                priority: List[str], bos_volume_ratio: Optional[float]) -> Setup:
    """Collect the POIs that belong to `bos`. Point-in-time: only uses events already in `smc`."""
    start = _impulse_start(smc, bos)
    setup = Setup(bos, bos.detected_index, start, [], bos_volume_ratio)
    max_h = poi_max_atr_multiple * atr if atr and atr > 0 else float("inf")

    obs: List[OrderBlock] = [ob for ob in smc.order_blocks if ob.bos is bos and ob.direction == BULLISH]
    ob_pois = [POI(ORDER_BLOCK, ob.low, ob.high, ob.detected_index, ob.timestamp) for ob in obs
               if ob.high > ob.low and (ob.high - ob.low) <= max_h]
    fvgs: List[FairValueGap] = [g for g in smc.fair_value_gaps
                                if g.direction == BULLISH and start < g.detected_index <= bos.detected_index]
    fvg_pois = [POI(FVG, g.lower, g.upper, g.detected_index, g.timestamp) for g in fvgs
                if g.upper > g.lower and (g.upper - g.lower) <= max_h]
    for fp in fvg_pois:
        fp.overlaps_ob = any(fp.low <= op.high and fp.high >= op.low for op in ob_pois)

    by_kind = {ORDER_BLOCK: ob_pois, FVG: fvg_pois}
    for kind in priority:
        setup.pois.extend(by_kind.get(kind, []))
    for kind in by_kind:
        if kind not in priority:
            setup.pois.extend(by_kind[kind])
    # de-duplicate identical zones
    seen, uniq = set(), []
    for p in setup.pois:
        key = (p.kind, round(p.low, 10), round(p.high, 10))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    setup.pois = uniq
    return setup


def mitigate(setup: Setup, close: float) -> None:
    for p in setup.pois:
        if not p.mitigated and close < p.low:
            p.mitigated = True


def select_poi(setup: Setup, low: float, close: float, priority: List[str]) -> Optional[POI]:
    """The highest-priority valid, non-skipped POI that the current candle touched."""
    rank = {k: i for i, k in enumerate(priority)}
    candidates = [p for p in setup.valid_pois() if not p.skipped and p.contains_touch(low, close)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (rank.get(p.kind, len(rank)), -p.high))
    return candidates[0]


def deeper_pois(setup: Setup, reference: POI) -> List[POI]:
    """All valid POIs strictly below `reference` (zone high below the reference's low), any kind."""
    return [p for p in setup.valid_pois() if p is not reference and p.high < reference.low]


def confluence_score(poi: POI, setup: Setup, has_sweep: bool, bos_vol_ratio_min: float) -> int:
    score = 0
    if poi.kind == ORDER_BLOCK:
        score += 1
    if poi.kind == FVG and poi.overlaps_ob:
        score += 1
    if has_sweep:
        score += 1
    if setup.bos_volume_ratio is not None and setup.bos_volume_ratio >= bos_vol_ratio_min:
        score += 1
    return score
