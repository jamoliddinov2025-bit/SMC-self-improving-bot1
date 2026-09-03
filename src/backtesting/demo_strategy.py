"""FixedIntervalTestStrategy - a deterministic TEST FIXTURE.

It is NOT a trading strategy and has no edge: every `interval_bars` bars while
flat it buys with an ATR-based stop and a fixed risk:reward target. Its only
purpose is to exercise the backtest engine's fill, risk and journal plumbing.
The real SMC strategy will be a separate class in a later step.
"""

import math
from typing import Any, Dict, Optional

from src.backtesting.strategy import BUY, BacktestContext, Signal


class FixedIntervalTestStrategy:
    def __init__(self, interval_bars: int = 25, stop_atr_multiplier: float = 1.5,
                 risk_reward: float = 2.0, atr_column: str = "atr_14"):
        if interval_bars < 1:
            raise ValueError("interval_bars must be >= 1")
        self.interval_bars = interval_bars
        self.stop_atr_multiplier = stop_atr_multiplier
        self.risk_reward = risk_reward
        self.atr_column = atr_column

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FixedIntervalTestStrategy":
        d = config.get("backtesting", {}).get("demo_strategy", {})
        atr_col = f"atr_{config['indicators']['atr']['period']}"
        return cls(int(d.get("interval_bars", 25)), float(d.get("stop_atr_multiplier", 1.5)),
                   float(d.get("risk_reward", 2.0)), atr_col)

    def on_candle(self, ctx: BacktestContext) -> Optional[Signal]:
        if ctx.has_position or ctx.index == 0 or ctx.index % self.interval_bars != 0:
            return None
        atr = ctx.indicators.get(self.atr_column)
        if atr is None or not math.isfinite(atr) or atr <= 0:
            return None  # still in indicator warm-up
        close = ctx.candle.close
        stop = close - self.stop_atr_multiplier * atr
        if stop <= 0:
            return None
        target = close + self.risk_reward * (close - stop)
        return Signal(BUY, stop_loss=stop, take_profit=target, reason=f"fixture bar {ctx.index}")
