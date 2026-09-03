"""Simulated spot broker. No exchange connection - all state is in memory.

Conventions
-----------
- `cash` is the quote-currency balance (e.g. USDT).
- `position` is the base-asset quantity held (e.g. BTC). Spot only: never negative.
- Fees are charged in quote currency on the notional value of every fill.
- `avg_entry_price` is the fee-inclusive average cost of the current position.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


class InsufficientFunds(ValueError):
    """Raised when a buy would cost more cash than is available."""


class InsufficientPosition(ValueError):
    """Raised when a sell asks for more asset than is held."""


@dataclass(frozen=True)
class Trade:
    timestamp: Any
    side: str            # "buy" | "sell"
    symbol: str
    price: float
    quantity: float
    notional: float      # price * quantity
    fee: float           # quote currency
    realized_pnl: float  # 0.0 for buys; net of fees for sells
    cash_after: float
    position_after: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaperBroker:
    def __init__(self, starting_cash: float, fee_pct: float = 0.0, symbol: str = "BTC/USDT"):
        if starting_cash < 0:
            raise ValueError("starting_cash must be >= 0")
        if fee_pct < 0:
            raise ValueError("fee_pct must be >= 0")
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.fee_rate = float(fee_pct) / 100.0
        self.symbol = symbol
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.trades: List[Trade] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PaperBroker":
        return cls(
            starting_cash=config["risk"]["starting_balance"],
            fee_pct=config["execution"]["paper_fee_pct"],
            symbol=config["market"]["symbol"],
        )

    # ----------------------------------------------------------------- orders
    def buy(self, price: float, quantity: float, timestamp: Optional[Any] = None) -> Trade:
        _validate_order(price, quantity)
        notional = price * quantity
        fee = notional * self.fee_rate
        total_cost = notional + fee
        if total_cost > self.cash + 1e-9:
            raise InsufficientFunds(
                f"Buy needs {total_cost:.2f} (incl. fee {fee:.2f}) but only {self.cash:.2f} cash available"
            )

        # Fee-inclusive weighted average entry price.
        prev_cost = self.avg_entry_price * self.position
        self.position += quantity
        self.avg_entry_price = (prev_cost + total_cost) / self.position
        self.cash -= total_cost

        return self._record(timestamp, "buy", price, quantity, notional, fee, 0.0)

    def sell(self, price: float, quantity: float, timestamp: Optional[Any] = None) -> Trade:
        _validate_order(price, quantity)
        if quantity > self.position + 1e-9:
            raise InsufficientPosition(
                f"Sell asks for {quantity} but only {self.position} held (spot: no shorting)"
            )
        quantity = min(quantity, self.position)  # absorb float dust

        notional = price * quantity
        fee = notional * self.fee_rate
        proceeds = notional - fee
        pnl = proceeds - self.avg_entry_price * quantity

        self.cash += proceeds
        self.position -= quantity
        self.realized_pnl += pnl
        if self.position <= 1e-12:
            self.position = 0.0
            self.avg_entry_price = 0.0

        return self._record(timestamp, "sell", price, quantity, notional, fee, pnl)

    # -------------------------------------------------------------- portfolio
    def position_value(self, price: float) -> float:
        return self.position * price

    def equity(self, price: float) -> float:
        """Cash plus mark-to-market value of the position (fees on a future exit not deducted)."""
        return self.cash + self.position_value(price)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_entry_price) * self.position if self.position else 0.0

    def total_fees(self) -> float:
        return sum(t.fee for t in self.trades)

    def portfolio(self, price: float) -> Dict[str, float]:
        equity = self.equity(price)
        return {
            "cash": self.cash,
            "position": self.position,
            "avg_entry_price": self.avg_entry_price,
            "position_value": self.position_value(price),
            "equity": equity,
            "unrealized_pnl": self.unrealized_pnl(price),
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees(),
            "return_pct": (equity / self.starting_cash - 1.0) * 100.0 if self.starting_cash else 0.0,
        }

    def trade_history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.trades]

    # ---------------------------------------------------------------- helpers
    def _record(self, timestamp, side, price, quantity, notional, fee, pnl) -> Trade:
        trade = Trade(
            timestamp=timestamp,
            side=side,
            symbol=self.symbol,
            price=float(price),
            quantity=float(quantity),
            notional=float(notional),
            fee=float(fee),
            realized_pnl=float(pnl),
            cash_after=self.cash,
            position_after=self.position,
        )
        self.trades.append(trade)
        return trade


def _validate_order(price: float, quantity: float) -> None:
    if price <= 0:
        raise ValueError("price must be > 0")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
