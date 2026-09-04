"""Strategy plug-in interface for the backtester.

A strategy receives a read-only `BacktestContext` for the current candle and
may return a `Signal`. The context intentionally exposes NO DataFrame and no
future candles: it carries only bar i's indicator row, the SMC result as known
at bar i, and the account snapshot. That is the anti-lookahead boundary.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

import pandas as pd

from src.data.base import Candle
from src.strategy.smc_types import SMCResult

if TYPE_CHECKING:  # avoid import cycles at runtime
    from src.strategy.regime import RegimeState

BUY = "buy"
EXIT = "exit"


@dataclass(frozen=True)
class Signal:
    side: str                          # BUY | EXIT
    stop_loss: Optional[float] = None  # required for BUY
    take_profit: Optional[float] = None
    reason: str = ""

    def __post_init__(self):
        if self.side not in (BUY, EXIT):
            raise ValueError(f"unknown signal side {self.side!r}")
        if self.side == BUY and (self.stop_loss is None or self.stop_loss <= 0):
            raise ValueError("BUY signal requires a positive stop_loss")


@dataclass(frozen=True)
class BacktestContext:
    index: int                    # 0-based bar index within the backtest
    candle: Candle                # bar i (fully closed)
    indicators: pd.Series         # indicator row for bar i only (may contain NaN in warm-up)
    smc: SMCResult                # live SMC state; every event has detected_index <= index
    has_position: bool
    equity: float                 # mark-to-market at bar i close
    risk: Dict[str, Any]          # RiskState.snapshot()
    regime: Optional["RegimeState"] = None   # USDT.D regime as of the last CLOSED aux candle (None if unused)
    aux: Dict[str, Any] = field(default_factory=dict)  # every named auxiliary feed's point-in-time state
                                                       # (e.g. aux["usdtd"] is the same object as `regime`)


class Strategy(Protocol):
    def on_candle(self, ctx: BacktestContext) -> Optional[Signal]: ...
