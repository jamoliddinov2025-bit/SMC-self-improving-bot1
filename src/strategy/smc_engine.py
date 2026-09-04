"""Deterministic SMC (Smart Money Concepts) market-structure engine.

This module ANALYSES price. It emits no buy/sell signals.

Processing model
----------------
The engine is incremental: `update(candle)` is called once per candle in
chronological order and only ever looks at candles already received.
`analyze(df)` is a convenience that replays a whole DataFrame through `update`.
Because of this, an event's `detected_index` is exactly the candle on which it
became knowable, and nothing received later can alter an already emitted event.

Definitions (exact)
-------------------
Pivot strength N (config `strategy.structure.pivot_strength`, default 3).

Swing High at i : high[i] >  high[j] for all j in [i-N, i) and (i, i+N]
Swing Low  at i : low[i]  <  low[j]  for all j in [i-N, i) and (i, i+N]
  * Strict inequalities. Requires the N candles AFTER i, therefore a swing at i
    is CONFIRMED (and first usable) when candle i+N has been processed:
    detected_index = i + N.

Structure labels: each new confirmed swing high is HH if its price > previous
confirmed swing high, else LH. Each new confirmed swing low is HL if its price
> previous confirmed swing low, else LL. The very first swing of each kind has
no label. (Equal prices count as LH / LL.)

BOS (Break of Structure), break condition = candle CLOSE:
  bullish BOS at candle c : close[c] > price of an unbroken confirmed swing high
  bearish BOS at candle c : close[c] < price of an unbroken confirmed swing low
  * Only swings with detected_index <= c are eligible (they must be confirmed).
  * The swing is marked broken and can never trigger again (repeated-break
    prevention). If several unbroken swing highs are broken by one close, one
    event is emitted per swing, all on the same candle.
  * Checked for every candle, including the confirming candle itself.

Structural direction: starts NEUTRAL; becomes BULLISH after a bullish BOS and
BEARISH after a bearish BOS. It changes only through BOS events.

CHoCH (Change of Character): a BOS whose direction is opposite to the current
structural direction.
  NEUTRAL -> any BOS      : BOS, not CHoCH (first direction is established)
  BULLISH -> bullish BOS  : BOS (continuation)
  BULLISH -> bearish BOS  : CHoCH (structure becomes BEARISH)
  BEARISH -> bearish BOS  : BOS (continuation)
  BEARISH -> bullish BOS  : CHoCH (structure becomes BULLISH)
Every CHoCH is also recorded in `bos_events` with `is_choch=True`.

Liquidity sweep of a confirmed, unbroken swing level, at candle c:
  bullish sweep : low[c]  < swing_low.price  and close[c] >= swing_low.price
  bearish sweep : high[c] > swing_high.price and close[c] <= swing_high.price
  * A candle that closes through the level is a BOS, not a sweep (exclusive).
  * A level may be swept more than once (each is recorded); a swept level stays
    eligible for a later BOS.

Fair Value Gap (three candles i-1, i, i+1), event timestamp = candle i:
  bullish FVG : high[i-1] < low[i+1]  -> lower = high[i-1], upper = low[i+1]
  bearish FVG : low[i-1]  > high[i+1] -> lower = high[i+1], upper = low[i-1]
  detected_index = i + 1. Optional filter `min_gap_pct`: gap size / close[i]
  * 100 must be >= min_gap_pct (default 0 = no filter).

Order Block (first version, deterministic):
  bullish OB : on a bullish BOS at candle c, the most recent candle k < c with
               close[k] < open[k] (bearish candle). Exactly one per BOS event.
  bearish OB : on a bearish BOS at candle c, the most recent candle k < c with
               close[k] > open[k] (bullish candle).
  If no such candle exists, no OB is produced. Dojis (close == open) are
  neither bullish nor bearish and are skipped.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import pandas as pd

from src.strategy.smc_types import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    BOSEvent,
    FairValueGap,
    LiquiditySweep,
    OrderBlock,
    SMCResult,
    Swing,
)


@dataclass(frozen=True)
class SMCConfig:
    pivot_strength: int = 3
    break_on: str = "close"
    sweep_enabled: bool = True
    order_blocks_enabled: bool = True
    fvg_enabled: bool = True
    fvg_min_gap_pct: float = 0.0

    def __post_init__(self):
        if self.pivot_strength < 1:
            raise ValueError("pivot_strength must be >= 1")
        if self.break_on != "close":
            raise ValueError("Only break_on='close' is supported")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SMCConfig":
        s = config["strategy"]
        return cls(
            pivot_strength=int(s["structure"].get("pivot_strength", 3)),
            break_on=s["structure"].get("break_on", "close"),
            sweep_enabled=bool(s.get("liquidity", {}).get("sweep_enabled", True)),
            order_blocks_enabled=bool(s.get("order_blocks", {}).get("enabled", True)),
            fvg_enabled=bool(s.get("fair_value_gaps", {}).get("enabled", True)),
            fvg_min_gap_pct=float(s.get("fair_value_gaps", {}).get("min_gap_pct", 0.0)),
        )


class SMCEngine:
    """Incremental SMC analyser. Feed candles in order via `update`."""

    def __init__(self, config: Optional[SMCConfig] = None):
        self.config = config or SMCConfig()
        self.result = SMCResult()
        # raw history (only what has been fed so far)
        self._ts: List[Any] = []
        self._o: List[float] = []
        self._h: List[float] = []
        self._l: List[float] = []
        self._c: List[float] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SMCEngine":
        return cls(SMCConfig.from_config(config))

    # ------------------------------------------------------------------ API
    @property
    def n(self) -> int:
        return len(self._c)

    @property
    def structure(self) -> str:
        return self.result.structure

    def update(self, timestamp: Any, open_: float, high: float, low: float, close: float) -> SMCResult:
        """Process the next candle. Returns the cumulative result."""
        self._ts.append(timestamp)
        self._o.append(float(open_))
        self._h.append(float(high))
        self._l.append(float(low))
        self._c.append(float(close))
        i = self.n - 1

        # 1) confirm any pivot that becomes knowable with this candle
        self._confirm_pivots(i)
        # 2) sweeps and breaks against confirmed, unbroken levels (this candle)
        if self.config.sweep_enabled:
            self._detect_sweeps(i)
        self._detect_bos(i)
        # 3) three-candle FVG whose third candle is this one
        if self.config.fvg_enabled:
            self._detect_fvg(i)
        return self.result

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """Replay a full OHLCV frame (columns timestamp, open, high, low, close)."""
        for row in df.itertuples(index=False):
            self.update(row.timestamp, row.open, row.high, row.low, row.close)
        return self.result

    # -------------------------------------------------------------- pivots
    def _confirm_pivots(self, i: int) -> None:
        N = self.config.pivot_strength
        p = i - N                      # candidate pivot index confirmed by candle i
        if p - N < 0:
            return
        left = range(p - N, p)
        right = range(p + 1, i + 1)

        hp = self._h[p]
        if all(hp > self._h[j] for j in left) and all(hp > self._h[j] for j in right):
            prev = self.result.swing_highs[-1] if self.result.swing_highs else None
            label = None if prev is None else ("HH" if hp > prev.price else "LH")
            self.result.swing_highs.append(
                Swing("high", p, self._ts[p], hp, i, self._ts[i], label)
            )

        lp = self._l[p]
        if all(lp < self._l[j] for j in left) and all(lp < self._l[j] for j in right):
            prev = self.result.swing_lows[-1] if self.result.swing_lows else None
            label = None if prev is None else ("HL" if lp > prev.price else "LL")
            self.result.swing_lows.append(
                Swing("low", p, self._ts[p], lp, i, self._ts[i], label)
            )

    # -------------------------------------------------------------- sweeps
    def _detect_sweeps(self, i: int) -> None:
        hi, lo, cl = self._h[i], self._l[i], self._c[i]
        for s in self.result.swing_lows:
            if s.broken or s.detected_index > i or s.index >= i:
                continue
            if lo < s.price <= cl:
                s.sweeps += 1
                self.result.liquidity_sweeps.append(
                    LiquiditySweep(BULLISH, s.price, s.timestamp, self._ts[i], lo, cl, i)
                )
        for s in self.result.swing_highs:
            if s.broken or s.detected_index > i or s.index >= i:
                continue
            if hi > s.price >= cl:
                s.sweeps += 1
                self.result.liquidity_sweeps.append(
                    LiquiditySweep(BEARISH, s.price, s.timestamp, self._ts[i], hi, cl, i)
                )

    # ----------------------------------------------------------------- BOS
    def _detect_bos(self, i: int) -> None:
        cl = self._c[i]
        for s in self.result.swing_highs:
            if s.broken or s.detected_index > i:
                continue
            if cl > s.price:
                self._emit_bos(BULLISH, s, i)
        for s in self.result.swing_lows:
            if s.broken or s.detected_index > i:
                continue
            if cl < s.price:
                self._emit_bos(BEARISH, s, i)

    def _emit_bos(self, direction: str, swing: Swing, i: int) -> None:
        swing.broken = True
        before = self.result.structure
        is_choch = (before == BULLISH and direction == BEARISH) or (before == BEARISH and direction == BULLISH)
        self.result.structure = direction
        event = BOSEvent(
            direction=direction,
            timestamp=self._ts[i],
            broken_swing_timestamp=swing.timestamp,
            broken_swing_price=swing.price,
            break_candle_timestamp=self._ts[i],
            break_candle_close=self._c[i],
            detected_index=i,
            is_choch=is_choch,
            structure_before=before,
            structure_after=direction,
        )
        self.result.bos_events.append(event)
        if self.config.order_blocks_enabled:
            self._find_order_block(event, i)

    # -------------------------------------------------------- order blocks
    def _find_order_block(self, bos: BOSEvent, i: int) -> None:
        want_bearish_candle = bos.direction == BULLISH
        for k in range(i - 1, -1, -1):
            o, c = self._o[k], self._c[k]
            if (want_bearish_candle and c < o) or (not want_bearish_candle and c > o):
                self.result.order_blocks.append(
                    OrderBlock(bos.direction, self._ts[k], o, self._h[k], self._l[k], c, bos, i)
                )
                return

    # ----------------------------------------------------------------- FVG
    def _detect_fvg(self, i: int) -> None:
        if i < 2:
            return
        m = i - 1
        first_h, first_l = self._h[i - 2], self._l[i - 2]
        third_h, third_l = self._h[i], self._l[i]
        mid_close = self._c[m]

        def big_enough(size: float) -> bool:
            if self.config.fvg_min_gap_pct <= 0:
                return True
            return mid_close > 0 and size / mid_close * 100.0 >= self.config.fvg_min_gap_pct

        if first_h < third_l and big_enough(third_l - first_h):
            self.result.fair_value_gaps.append(FairValueGap(BULLISH, self._ts[m], third_l, first_h, i))
        elif first_l > third_h and big_enough(first_l - third_h):
            self.result.fair_value_gaps.append(FairValueGap(BEARISH, self._ts[m], first_l, third_h, i))

    # ------------------------------------------------------------- helpers
    def events_known_at(self, index: int) -> SMCResult:
        """Snapshot of everything knowable at candle `index` (inclusive)."""
        r = self.result
        bos = [e for e in r.bos_events if e.detected_index <= index]
        sweeps = [e for e in r.liquidity_sweeps if e.detected_index <= index]
        broken = {(e.broken_swing_timestamp, e.broken_swing_price) for e in bos}

        def snapshot(s: Swing) -> Swing:
            # Swing.broken / Swing.sweeps mutate over time; rebuild them as of `index`.
            key = (s.timestamp, s.price)
            return replace(
                s,
                broken=key in broken,
                sweeps=sum(1 for e in sweeps if (e.swept_swing_timestamp, e.swept_level) == key),
            )

        return SMCResult(
            swing_highs=[snapshot(s) for s in r.swing_highs if s.detected_index <= index],
            swing_lows=[snapshot(s) for s in r.swing_lows if s.detected_index <= index],
            bos_events=bos,
            liquidity_sweeps=sweeps,
            fair_value_gaps=[e for e in r.fair_value_gaps if e.detected_index <= index],
            order_blocks=[e for e in r.order_blocks if e.detected_index <= index],
            structure=_structure_at(r.bos_events, index),
        )


def _structure_at(bos_events: List[BOSEvent], index: int) -> str:
    for e in reversed(bos_events):
        if e.detected_index <= index:
            return e.structure_after
    return NEUTRAL
