"""Performance metrics computed from a TradeJournal and an equity curve."""

import math
from typing import Any, Dict, Tuple

import pandas as pd

from src.backtesting.journal import TradeJournal


def max_drawdown(equity: pd.Series) -> Tuple[float, int]:
    """Return (max drawdown %, longest drawdown duration in bars)."""
    if equity.empty:
        return 0.0, 0
    peak = equity.cummax()
    dd = (peak - equity) / peak * 100.0
    max_dd = float(dd.max()) if len(dd) else 0.0
    longest = current = 0
    for in_dd in (dd > 0).tolist():
        current = current + 1 if in_dd else 0
        longest = max(longest, current)
    return max_dd, longest


def compute_metrics(journal: TradeJournal, equity_curve: pd.DataFrame, starting_equity: float,
                    risk_per_trade_pct: float) -> Dict[str, Any]:
    trades = journal.trades
    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    breakeven = len(pnls) - len(wins) - len(losses)
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    ending_equity = float(equity_curve["equity"].iloc[-1]) if len(equity_curve) else float(starting_equity)
    max_dd, dd_bars = max_drawdown(equity_curve["equity"]) if len(equity_curve) else (0.0, 0)

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    n = len(pnls)
    exit_reasons: Dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "starting_equity": float(starting_equity),
        "ending_equity": ending_equity,
        "total_return_pct": (ending_equity / starting_equity - 1.0) * 100.0 if starting_equity else 0.0,
        "net_profit": ending_equity - starting_equity,
        "trades": n,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "breakeven_trades": breakeven,
        "win_rate_pct": len(wins) / n * 100.0 if n else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "average_win": gross_profit / len(wins) if wins else 0.0,
        "average_loss": -gross_loss / len(losses) if losses else 0.0,
        "expectancy": sum(pnls) / n if n else 0.0,
        "average_r_multiple": sum(t.r_multiple for t in trades) / n if n else 0.0,
        "total_fees": journal.total_fees,
        "max_drawdown_pct": max_dd,
        "max_drawdown_bars": dd_bars,
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "avg_realized_risk_pct": (sum(t.risk_amount for t in trades) / n / starting_equity * 100.0) if n else 0.0,
        "bars": int(len(equity_curve)),
        "exit_reasons": exit_reasons,
        "risk_rejections": journal.rejection_counts(),
    }
