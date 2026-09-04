"""Trade validator: applies kill switches and sizing to a `TradeProposal`.

Check order (first failure wins):
  1. input sanity           -> REJECTED_INVALID_INPUT
  2. equity > 0             -> REJECTED_INVALID_EQUITY
  3. open positions < max   -> REJECTED_POSITION_LIMIT
  4. daily loss < limit     -> REJECTED_DAILY_LOSS_LIMIT
  5. drawdown < limit       -> REJECTED_DRAWDOWN_LIMIT
  6. loss streak < limit    -> REJECTED_CONSECUTIVE_LOSS_LIMIT
  7. stop below entry       -> REJECTED_INVALID_STOP
  8. sizing valid           -> APPROVED (else REJECTED_INVALID_INPUT)

Limits are inclusive: reaching the limit (>=) locks trading.
"""

import math
from typing import Any, Dict, Optional

from src.risk.position_sizing import calculate_position_size
from src.risk.state import RiskState
from src.risk.types import RiskAssessment, RiskConfig, RiskDecision, TradeProposal


class TradeValidator:
    def __init__(self, config: RiskConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TradeValidator":
        return cls(RiskConfig.from_config(config))

    def validate(self, p: TradeProposal) -> RiskAssessment:
        c = self.config
        checks = {
            "open_positions": f"{p.open_positions}/{c.max_open_positions}",
            "daily_loss_pct": f"{p.current_daily_loss_pct:.2f}/{c.max_daily_loss_pct:.2f}",
            "drawdown_pct": f"{p.current_drawdown_pct:.2f}/{c.max_drawdown_pct:.2f}",
            "consecutive_losses": f"{p.consecutive_losses}/{c.max_consecutive_losses}",
        }

        def reject(decision: RiskDecision, reason: str) -> RiskAssessment:
            return RiskAssessment(decision, False, reason, None, checks)

        # 1. input sanity
        numeric = {
            "account_equity": p.account_equity, "entry_price": p.entry_price,
            "stop_loss_price": p.stop_loss_price, "current_drawdown_pct": p.current_drawdown_pct,
            "current_daily_loss_pct": p.current_daily_loss_pct,
        }
        for name, v in numeric.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return reject(RiskDecision.REJECTED_INVALID_INPUT, f"{name} must be a finite number")
        for name, v in (("consecutive_losses", p.consecutive_losses), ("open_positions", p.open_positions)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return reject(RiskDecision.REJECTED_INVALID_INPUT, f"{name} must be a non-negative integer")
        if p.current_drawdown_pct < 0 or p.current_daily_loss_pct < 0:
            return reject(RiskDecision.REJECTED_INVALID_INPUT, "drawdown and daily loss must be >= 0")
        if p.entry_price <= 0 or p.stop_loss_price <= 0:
            return reject(RiskDecision.REJECTED_INVALID_INPUT, "prices must be > 0")

        # 2. equity
        if p.account_equity <= 0:
            return reject(RiskDecision.REJECTED_INVALID_EQUITY, "account_equity must be > 0")

        # 3-6. kill switches
        if p.open_positions >= c.max_open_positions:
            return reject(RiskDecision.REJECTED_POSITION_LIMIT,
                          f"open positions {p.open_positions} >= limit {c.max_open_positions}")
        if p.current_daily_loss_pct >= c.max_daily_loss_pct:
            return reject(RiskDecision.REJECTED_DAILY_LOSS_LIMIT,
                          f"daily loss {p.current_daily_loss_pct:.2f}% >= limit {c.max_daily_loss_pct:.2f}%")
        if p.current_drawdown_pct >= c.max_drawdown_pct:
            return reject(RiskDecision.REJECTED_DRAWDOWN_LIMIT,
                          f"drawdown {p.current_drawdown_pct:.2f}% >= limit {c.max_drawdown_pct:.2f}%")
        # NOTE (deadlock): this lock only clears when the streak is reset. A win
        # cannot happen while locked, so the streak must be cleared externally via
        # RiskState.reset_loss_streak() - the documented hook for a future cooldown
        # unlock. Not implemented yet; see src/risk/state.py and README.
        if p.consecutive_losses >= c.max_consecutive_losses:
            return reject(RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT,
                          f"consecutive losses {p.consecutive_losses} >= limit {c.max_consecutive_losses}")

        # 7. stop
        if p.stop_loss_price >= p.entry_price:
            return reject(RiskDecision.REJECTED_INVALID_STOP,
                          f"stop {p.stop_loss_price} must be below entry {p.entry_price} (long spot only)")

        # 8. sizing
        sizing = calculate_position_size(p.account_equity, p.entry_price, p.stop_loss_price, c.risk_per_trade_pct)
        if not sizing.valid or sizing.position_size <= 0:
            return reject(RiskDecision.REJECTED_INVALID_INPUT, sizing.reason or "position size is zero")
        return RiskAssessment(RiskDecision.APPROVED, True,
                              f"risk {sizing.risk_amount:.2f} ({sizing.effective_risk_pct:.2f}%), size {sizing.position_size:.6f}",
                              sizing, checks)

    def validate_with_state(self, state: RiskState, entry_price: float, stop_loss_price: float,
                            account_equity: Optional[float] = None) -> RiskAssessment:
        """Convenience: build the proposal from a `RiskState`."""
        return self.validate(TradeProposal(
            account_equity=state.equity if account_equity is None else account_equity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            current_drawdown_pct=state.drawdown_pct,
            current_daily_loss_pct=state.daily_loss_pct,
            consecutive_losses=state.consecutive_losses,
            open_positions=state.open_positions,
        ))
