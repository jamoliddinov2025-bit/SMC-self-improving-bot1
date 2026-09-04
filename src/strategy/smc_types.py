"""Data types emitted by the SMC engine.

Every event carries `detected_index`: the 0-based candle index at which the
event became knowable. Consumers must never use an event before that index.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"


@dataclass
class Swing:
    kind: str                 # "high" | "low"
    index: int                # pivot candle index
    timestamp: Any
    price: float              # high for swing highs, low for swing lows
    detected_index: int       # index + pivot_strength
    detected_timestamp: Any
    label: Optional[str] = None   # HH / LH for highs, HL / LL for lows, None for the first swing
    broken: bool = False          # consumed by a BOS
    sweeps: int = 0               # number of times this level has been swept


@dataclass(frozen=True)
class BOSEvent:
    direction: str            # bullish | bearish
    timestamp: Any            # == break candle timestamp
    broken_swing_timestamp: Any
    broken_swing_price: float
    break_candle_timestamp: Any
    break_candle_close: float
    detected_index: int
    is_choch: bool
    structure_before: str
    structure_after: str


@dataclass(frozen=True)
class LiquiditySweep:
    direction: str            # bullish (swept a low) | bearish (swept a high)
    swept_level: float
    swept_swing_timestamp: Any
    timestamp: Any            # sweep candle timestamp
    wick_extreme: float       # candle low (bullish) / candle high (bearish)
    close: float
    detected_index: int


@dataclass(frozen=True)
class FairValueGap:
    direction: str
    timestamp: Any            # middle candle timestamp
    upper: float
    lower: float
    detected_index: int       # middle index + 1 (third candle)

    @property
    def size(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class OrderBlock:
    direction: str
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    bos: BOSEvent
    detected_index: int       # == bos.detected_index


@dataclass
class SMCResult:
    swing_highs: List[Swing] = field(default_factory=list)
    swing_lows: List[Swing] = field(default_factory=list)
    bos_events: List[BOSEvent] = field(default_factory=list)
    liquidity_sweeps: List[LiquiditySweep] = field(default_factory=list)
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    structure: str = NEUTRAL

    @property
    def choch_events(self) -> List[BOSEvent]:
        return [e for e in self.bos_events if e.is_choch]
