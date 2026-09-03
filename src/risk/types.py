"""Risk-engine data types and decision codes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED_INVALID_STOP = "REJECTED_INVALID_STOP"
    REJECTED_DAILY_LOSS_LIMIT = "REJECTED_DAILY_LOSS_LIMIT"
    REJECTED_DRAWDOWN_LIMIT = "REJECTED_DRAWDOWN_LIMIT"
    REJECTED_CONSECUTIVE_LOSS_LIMIT = "REJECTED_CONSECUTIVE_LOSS_LIMIT"
    REJECTED_POSITION_LIMIT = "REJECTED_POSITION_LIMIT"
    REJECTED_INVALID_EQUITY = "REJECTED_INVALID_EQUITY"
    REJECTED_INVALID_INPUT = "REJECTED_INVALID_INPUT"


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 5
    max_open_positions: int = 1

    def __post_init__(self):
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade_pct must be in (0, 100]")
        for name in ("max_daily_loss_pct", "max_drawdown_pct"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RiskConfig":
        r = config["risk"]
        return cls(
            risk_per_trade_pct=float(r["risk_per_trade_pct"]),
            max_daily_loss_pct=float(r["max_daily_loss_pct"]),
            max_drawdown_pct=float(r["max_drawdown_pct"]),
            max_consecutive_losses=int(r["max_consecutive_losses"]),
            max_open_positions=int(r["max_open_positions"]),
        )


@dataclass(frozen=True)
class PositionSize:
    valid: bool
    risk_amount: float = 0.0        # quote currency at risk if the stop is hit
    position_size: float = 0.0      # base-asset quantity
    stop_distance: float = 0.0      # quote currency per unit
    effective_risk_pct: float = 0.0 # risk_amount / equity * 100
    reason: Optional[str] = None    # why invalid


@dataclass(frozen=True)
class TradeProposal:
    account_equity: float
    entry_price: float
    stop_loss_price: float
    current_drawdown_pct: float = 0.0     # >= 0
    current_daily_loss_pct: float = 0.0   # >= 0, loss expressed as a positive number
    consecutive_losses: int = 0
    open_positions: int = 0


@dataclass(frozen=True)
class RiskAssessment:
    decision: RiskDecision
    approved: bool
    reason: str
    sizing: Optional[PositionSize] = None
    checks: Dict[str, Any] = field(default_factory=dict)
