"""Trade journal: completed round trips plus every rejected proposal."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

EXIT_STOP = "stop_loss"
EXIT_TARGET = "take_profit"
EXIT_SIGNAL = "signal"
EXIT_END = "end_of_data"


@dataclass(frozen=True)
class TradeRecord:
    trade_id: int
    signal_index: int
    signal_timestamp: Any
    entry_index: int
    entry_timestamp: Any
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: Optional[float]
    exit_index: int
    exit_timestamp: Any
    exit_price: float
    exit_reason: str
    entry_fee: float
    exit_fee: float
    gross_pnl: float          # (exit - entry) * qty, before fees
    net_pnl: float            # PaperBroker realized_pnl (fees included)
    r_multiple: float         # net_pnl / risk_amount
    risk_amount: float        # sized risk in quote currency (from validator)
    bars_held: int
    entry_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedProposal:
    index: int
    timestamp: Any
    decision: str
    reason: str
    entry_price: float
    stop_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeJournal:
    trades: List[TradeRecord] = field(default_factory=list)
    rejections: List[RejectedProposal] = field(default_factory=list)

    def add_trade(self, t: TradeRecord) -> None:
        self.trades.append(t)

    def add_rejection(self, r: RejectedProposal) -> None:
        self.rejections.append(r)

    @property
    def total_fees(self) -> float:
        return sum(t.entry_fee + t.exit_fee for t in self.trades)

    def rejection_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.rejections:
            out[r.decision] = out.get(r.decision, 0) + 1
        return out

    def to_frame(self) -> pd.DataFrame:
        cols = list(TradeRecord.__dataclass_fields__)
        return pd.DataFrame([t.to_dict() for t in self.trades], columns=cols)

    def rejections_frame(self) -> pd.DataFrame:
        cols = list(RejectedProposal.__dataclass_fields__)
        return pd.DataFrame([r.to_dict() for r in self.rejections], columns=cols)
