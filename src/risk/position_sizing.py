"""Fixed-fractional position sizing.

    risk_amount   = account_equity * risk_per_trade_pct / 100
    stop_distance = entry_price - stop_loss_price      (long / spot only)
    position_size = risk_amount / stop_distance

Returns an invalid `PositionSize` (never raises) for bad inputs so callers can
surface the reason.
"""

import math

from src.risk.types import PositionSize


def _finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: float,
    max_position_pct: float = 100.0,
) -> PositionSize:
    """Size a long spot position.

    `max_position_pct` caps notional at a percentage of equity (spot cannot
    spend more cash than it has; default 100%). When the cap binds, the
    effective risk % is lower than requested.
    """
    for name, v in (("account_equity", account_equity), ("entry_price", entry_price),
                    ("stop_loss_price", stop_loss_price), ("risk_per_trade_pct", risk_per_trade_pct)):
        if not _finite_number(v):
            return PositionSize(False, reason=f"{name} must be a finite number")
    if account_equity <= 0:
        return PositionSize(False, reason="account_equity must be > 0")
    if entry_price <= 0 or stop_loss_price <= 0:
        return PositionSize(False, reason="prices must be > 0")
    if not 0 < risk_per_trade_pct <= 100:
        return PositionSize(False, reason="risk_per_trade_pct must be in (0, 100]")

    stop_distance = entry_price - stop_loss_price
    if stop_distance <= 0:
        return PositionSize(
            False, stop_distance=stop_distance,
            reason="stop_loss_price must be below entry_price (long spot only)",
        )

    risk_amount = account_equity * risk_per_trade_pct / 100.0
    position_size = risk_amount / stop_distance

    max_notional = account_equity * max_position_pct / 100.0
    if position_size * entry_price > max_notional:
        position_size = max_notional / entry_price
        risk_amount = position_size * stop_distance

    return PositionSize(
        valid=True,
        risk_amount=risk_amount,
        position_size=max(position_size, 0.0),
        stop_distance=stop_distance,
        effective_risk_pct=risk_amount / account_equity * 100.0,
    )
