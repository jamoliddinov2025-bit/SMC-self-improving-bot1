"""Deterministic risk-state tracker: equity, peak, drawdown, loss streak, daily PnL.

Pure bookkeeping - the caller tells it what happened (`record_trade`,
`update_equity`, `new_day`). It never talks to a broker or exchange.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RiskState:
    starting_equity: float
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    day_start_equity: float = field(init=False)
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    current_day: Optional[Any] = None   # any hashable day key (e.g. datetime.date)
    last_streak_reset_reason: Optional[str] = None  # set by reset_loss_streak()

    def __post_init__(self):
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")
        self.equity = float(self.starting_equity)
        self.peak_equity = self.equity
        self.day_start_equity = self.equity

    # ------------------------------------------------------------ metrics
    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)

    @property
    def daily_pnl_pct(self) -> float:
        return self.daily_pnl / self.day_start_equity * 100.0 if self.day_start_equity else 0.0

    @property
    def daily_loss_pct(self) -> float:
        """Today's loss as a positive percentage (0 when the day is flat or positive)."""
        return max(0.0, -self.daily_pnl_pct)

    # ------------------------------------------------------------ updates
    def update_equity(self, equity: float) -> None:
        """Mark-to-market equity update (does not touch PnL counters)."""
        if equity < 0:
            raise ValueError("equity cannot be negative")
        self.equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.equity)

    def record_trade(self, realized_pnl: float, day: Optional[Any] = None) -> None:
        """Register a closed trade's net PnL (fees included)."""
        if day is not None:
            self.new_day(day)
        self.equity += realized_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl += realized_pnl
        if realized_pnl < 0:
            self.consecutive_losses += 1
        elif realized_pnl > 0:
            self.consecutive_losses = 0
        # a zero-PnL trade leaves the streak unchanged

    def new_day(self, day: Any) -> bool:
        """Roll the daily counters if `day` differs from the current day. Returns True if rolled."""
        if self.current_day == day:
            return False
        self.current_day = day
        self.daily_pnl = 0.0
        self.day_start_equity = self.equity
        return True

    def reset_loss_streak(self, reason: str = "manual") -> int:
        """Clear the consecutive-loss counter and return the value it had.

        KNOWN LIMITATION / EXTENSION POINT (loss-streak deadlock)
        ---------------------------------------------------------
        Once `consecutive_losses >= max_consecutive_losses` the validator
        rejects every new trade. Because the streak is only reset by a
        *winning* trade, and no trade can be opened while locked, the lock
        would never clear on its own. This method is the single, explicit
        place where an external unlock may happen.

        Current behaviour : nothing calls this automatically. The lock stays
                            until the operator (or a later component) calls
                            `reset_loss_streak()`. `new_day()` does NOT clear it.
        Planned extension : a cooldown-based unlock (e.g. after N bars or a
                            fixed time since the last loss) implemented in a
                            later step, which will call this method with
                            reason="cooldown". No cooldown logic exists yet.
        """
        previous = self.consecutive_losses
        self.consecutive_losses = 0
        self.last_streak_reset_reason = reason
        return previous

    def open_position(self) -> None:
        self.open_positions += 1

    def close_position(self) -> None:
        if self.open_positions <= 0:
            raise ValueError("no open position to close")
        self.open_positions -= 1

    # ------------------------------------------------------------ persistence
    def to_snapshot(self) -> Dict[str, Any]:
        """Full JSON-serialisable state (superset of `snapshot()`), for restart recovery."""
        return {
            "starting_equity": self.starting_equity, "equity": self.equity, "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity, "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses, "open_positions": self.open_positions,
            "current_day": self.current_day.isoformat() if hasattr(self.current_day, "isoformat") else self.current_day,
            "last_streak_reset_reason": self.last_streak_reset_reason,
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "RiskState":
        import datetime as _dt
        rs = cls(float(snap["starting_equity"]))
        rs.equity = float(snap["equity"])
        rs.peak_equity = float(snap["peak_equity"])
        rs.day_start_equity = float(snap["day_start_equity"])
        rs.daily_pnl = float(snap["daily_pnl"])
        rs.consecutive_losses = int(snap["consecutive_losses"])
        rs.open_positions = int(snap["open_positions"])
        day = snap.get("current_day")
        if isinstance(day, str):
            try:
                day = _dt.date.fromisoformat(day)
            except ValueError:
                pass
        rs.current_day = day
        rs.last_streak_reset_reason = snap.get("last_streak_reset_reason")
        if rs.consecutive_losses < 0 or rs.open_positions < 0 or rs.equity < 0:
            raise ValueError("corrupt risk snapshot")
        return rs

    def snapshot(self) -> Dict[str, float]:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.drawdown_pct,
            "daily_pnl": self.daily_pnl,
            "daily_loss_pct": self.daily_loss_pct,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": self.open_positions,
        }
