"""SMC market-structure analysis (SMCEngine) and the SMC trading strategy (SMCStrategy)."""

from src.strategy.regime import Regime, RegimeConfig, RegimeState, USDTDRegimeDetector
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
    "SMCStrategy", "SMCStrategyConfig", "Regime", "RegimeConfig", "RegimeState", "USDTDRegimeDetector",
]


def __getattr__(name):  # lazy: SMCStrategy depends on src.backtesting, which imports this package
    if name in ("SMCStrategy", "SMCStrategyConfig"):
        from src.strategy import smc_strategy
        return getattr(smc_strategy, name)
    raise AttributeError(name)
