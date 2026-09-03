"""SMC market-structure analysis. No trading signals are produced here."""

from src.strategy.smc_engine import SMCConfig, SMCEngine
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

__all__ = [
    "SMCConfig", "SMCEngine", "SMCResult", "Swing", "BOSEvent", "LiquiditySweep",
    "FairValueGap", "OrderBlock", "BULLISH", "BEARISH", "NEUTRAL",
]
