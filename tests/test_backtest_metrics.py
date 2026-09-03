"""Metrics tests on hand-built journals and equity curves."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting import EXIT_STOP, EXIT_TARGET, TradeJournal, TradeRecord, compute_metrics, max_drawdown  # noqa: E402
from src.backtesting.journal import RejectedProposal  # noqa: E402


def rec(i, net, entry_fee=1.0, exit_fee=1.0, risk=100.0, reason=EXIT_TARGET):
    gross = net + entry_fee + exit_fee
    return TradeRecord(i, i, None, i + 1, None, 100.0, 1.0, 95.0, 110.0, i + 2, None, 100.0 + gross, reason,
                       entry_fee, exit_fee, gross, net, net / risk, risk, 1, "test")


def curve(values):
    return pd.DataFrame({"equity": values})


def test_max_drawdown_hand_calculated():
    dd, bars = max_drawdown(pd.Series([100, 110, 99, 104.5, 121, 120]))
    assert dd == pytest.approx(10.0)          # 110 -> 99
    assert bars == 2                           # 99, 104.5 below the 110 peak
    assert max_drawdown(pd.Series([100, 101, 102])) == (0.0, 0)
    assert max_drawdown(pd.Series([], dtype=float)) == (0.0, 0)


def test_metrics_hand_calculated():
    j = TradeJournal()
    j.add_trade(rec(1, +200.0))
    j.add_trade(rec(2, -50.0, reason=EXIT_STOP))
    j.add_trade(rec(3, +100.0))
    j.add_trade(rec(4, -150.0, reason=EXIT_STOP))
    j.add_rejection(RejectedProposal(9, None, "REJECTED_POSITION_LIMIT", "x", 100.0, 95.0))
    m = compute_metrics(j, curve([10_000, 10_200, 10_150, 10_250, 10_100]), 10_000, 1.0)
    assert m["trades"] == 4 and m["winning_trades"] == 2 and m["losing_trades"] == 2 and m["breakeven_trades"] == 0
    assert m["win_rate_pct"] == 50.0
    assert m["gross_profit"] == 300.0 and m["gross_loss"] == 200.0
    assert m["profit_factor"] == pytest.approx(1.5)
    assert m["average_win"] == 150.0 and m["average_loss"] == -100.0
    assert m["expectancy"] == 25.0 and m["average_r_multiple"] == pytest.approx(0.25)
    assert m["total_fees"] == 8.0
    assert m["ending_equity"] == 10_100 and m["total_return_pct"] == pytest.approx(1.0) and m["net_profit"] == 100
    assert m["max_drawdown_pct"] == pytest.approx((10_250 - 10_100) / 10_250 * 100)
    assert m["risk_per_trade_pct"] == 1.0 and m["avg_realized_risk_pct"] == pytest.approx(1.0)
    assert m["exit_reasons"] == {EXIT_TARGET: 2, EXIT_STOP: 2}
    assert m["risk_rejections"] == {"REJECTED_POSITION_LIMIT": 1}


def test_profit_factor_edge_cases():
    j = TradeJournal(); j.add_trade(rec(1, 10.0))
    assert math.isinf(compute_metrics(j, curve([1, 1]), 1_000, 1.0)["profit_factor"])
    j = TradeJournal(); j.add_trade(rec(1, -10.0))
    assert compute_metrics(j, curve([1, 1]), 1_000, 1.0)["profit_factor"] == 0.0
    j = TradeJournal(); j.add_trade(rec(1, 0.0))
    m = compute_metrics(j, curve([1, 1]), 1_000, 1.0)
    assert m["breakeven_trades"] == 1 and m["win_rate_pct"] == 0.0 and m["profit_factor"] == 0.0


def test_empty_journal_no_division_errors():
    m = compute_metrics(TradeJournal(), curve([]), 5_000, 2.0)
    assert m["trades"] == 0 and m["ending_equity"] == 5_000 and m["total_return_pct"] == 0.0
    assert m["average_win"] == m["average_loss"] == m["expectancy"] == 0.0
    assert m["max_drawdown_pct"] == 0.0 and m["risk_per_trade_pct"] == 2.0


def test_journal_frames_have_stable_columns():
    j = TradeJournal()
    assert list(j.to_frame().columns)[:3] == ["trade_id", "signal_index", "signal_timestamp"]
    assert list(j.rejections_frame().columns) == ["index", "timestamp", "decision", "reason", "entry_price", "stop_loss"]
