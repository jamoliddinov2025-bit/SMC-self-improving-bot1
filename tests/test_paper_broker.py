"""Unit tests for the PaperBroker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution import InsufficientFunds, InsufficientPosition, PaperBroker  # noqa: E402
from src.main import load_config  # noqa: E402


@pytest.fixture
def broker():
    return PaperBroker(starting_cash=1000.0, fee_pct=0.1, symbol="BTC/USDT")


@pytest.fixture
def free_broker():
    return PaperBroker(starting_cash=1000.0, fee_pct=0.0)


# ------------------------------------------------------------------ buying
def test_buy_reduces_cash_and_increases_position(free_broker):
    free_broker.buy(price=100.0, quantity=2.0)
    assert free_broker.cash == pytest.approx(800.0)
    assert free_broker.position == pytest.approx(2.0)
    assert free_broker.avg_entry_price == pytest.approx(100.0)


def test_buy_average_entry_price_is_weighted(free_broker):
    free_broker.buy(100.0, 1.0)
    free_broker.buy(200.0, 1.0)
    assert free_broker.avg_entry_price == pytest.approx(150.0)
    assert free_broker.position == pytest.approx(2.0)


def test_buy_exact_full_cash_allowed(free_broker):
    free_broker.buy(100.0, 10.0)
    assert free_broker.cash == pytest.approx(0.0)


def test_buy_rejects_bad_inputs(free_broker):
    with pytest.raises(ValueError):
        free_broker.buy(0.0, 1.0)
    with pytest.raises(ValueError):
        free_broker.buy(100.0, -1.0)


# ----------------------------------------------------------------- selling
def test_sell_increases_cash_and_reduces_position(free_broker):
    free_broker.buy(100.0, 2.0)
    free_broker.sell(110.0, 1.0)
    assert free_broker.cash == pytest.approx(910.0)
    assert free_broker.position == pytest.approx(1.0)
    assert free_broker.avg_entry_price == pytest.approx(100.0)  # unchanged by partial sell


def test_sell_full_position_resets_entry_price(free_broker):
    free_broker.buy(100.0, 2.0)
    free_broker.sell(120.0, 2.0)
    assert free_broker.position == 0.0
    assert free_broker.avg_entry_price == 0.0
    assert free_broker.realized_pnl == pytest.approx(40.0)


def test_sell_realized_pnl_loss(free_broker):
    free_broker.buy(100.0, 1.0)
    trade = free_broker.sell(90.0, 1.0)
    assert trade.realized_pnl == pytest.approx(-10.0)
    assert free_broker.cash == pytest.approx(990.0)


# -------------------------------------------------------------------- fees
def test_buy_fee_charged_in_quote(broker):
    trade = broker.buy(100.0, 1.0)
    assert trade.fee == pytest.approx(0.1)
    assert broker.cash == pytest.approx(1000.0 - 100.1)
    # entry price includes the fee
    assert broker.avg_entry_price == pytest.approx(100.1)


def test_sell_fee_and_round_trip_pnl(broker):
    broker.buy(100.0, 1.0)
    trade = broker.sell(100.0, 1.0)
    assert trade.fee == pytest.approx(0.1)
    # flat price round trip loses exactly both fees
    assert broker.cash == pytest.approx(1000.0 - 0.2)
    assert trade.realized_pnl == pytest.approx(-0.2)
    assert broker.total_fees() == pytest.approx(0.2)


def test_fee_rate_from_config():
    cfg = load_config()
    b = PaperBroker.from_config(cfg)
    assert b.fee_rate == pytest.approx(cfg["execution"]["paper_fee_pct"] / 100)
    assert b.cash == cfg["risk"]["starting_balance"]
    assert b.symbol == cfg["market"]["symbol"]


def test_negative_fee_rejected():
    with pytest.raises(ValueError):
        PaperBroker(1000.0, fee_pct=-1)


# ------------------------------------------------------- insufficient cash
def test_insufficient_cash_raises_and_leaves_state_untouched(free_broker):
    with pytest.raises(InsufficientFunds):
        free_broker.buy(100.0, 10.01)
    assert free_broker.cash == 1000.0
    assert free_broker.position == 0.0
    assert free_broker.trades == []


def test_fee_counts_towards_insufficient_cash(broker):
    # 10 units @100 = 1000 notional, +0.1% fee -> 1001 > 1000 cash
    with pytest.raises(InsufficientFunds):
        broker.buy(100.0, 10.0)


# --------------------------------------------------- insufficient position
def test_sell_without_position_raises(free_broker):
    with pytest.raises(InsufficientPosition):
        free_broker.sell(100.0, 1.0)
    assert free_broker.trades == []


def test_sell_more_than_held_raises(free_broker):
    free_broker.buy(100.0, 1.0)
    with pytest.raises(InsufficientPosition):
        free_broker.sell(100.0, 1.5)
    assert free_broker.position == pytest.approx(1.0)


def test_no_shorting_ever():
    b = PaperBroker(1_000_000.0)
    with pytest.raises(InsufficientPosition):
        b.sell(100.0, 0.0001)


# ------------------------------------------------------- equity calculation
def test_equity_with_no_position_equals_cash(free_broker):
    assert free_broker.equity(123.0) == pytest.approx(1000.0)


def test_equity_marks_position_to_market(free_broker):
    free_broker.buy(100.0, 2.0)
    assert free_broker.equity(100.0) == pytest.approx(1000.0)
    assert free_broker.equity(150.0) == pytest.approx(1100.0)
    assert free_broker.equity(50.0) == pytest.approx(900.0)
    assert free_broker.unrealized_pnl(150.0) == pytest.approx(100.0)


def test_portfolio_summary(broker):
    broker.buy(100.0, 1.0)
    p = broker.portfolio(110.0)
    assert p["cash"] == pytest.approx(899.9)
    assert p["position"] == pytest.approx(1.0)
    assert p["position_value"] == pytest.approx(110.0)
    assert p["equity"] == pytest.approx(1009.9)
    assert p["return_pct"] == pytest.approx(0.99)
    assert p["total_fees"] == pytest.approx(0.1)


# ----------------------------------------------------------- trade history
def test_trade_history_records_every_fill(free_broker):
    free_broker.buy(100.0, 1.0, timestamp="t1")
    free_broker.sell(110.0, 0.5, timestamp="t2")
    hist = free_broker.trade_history()
    assert len(hist) == 2
    assert hist[0]["side"] == "buy" and hist[0]["timestamp"] == "t1"
    assert hist[1]["side"] == "sell" and hist[1]["timestamp"] == "t2"
    assert hist[1]["realized_pnl"] == pytest.approx(5.0)
    assert hist[1]["cash_after"] == pytest.approx(955.0)
    assert hist[1]["position_after"] == pytest.approx(0.5)
    assert set(hist[0]) == {"timestamp", "side", "symbol", "price", "quantity", "notional",
                            "fee", "realized_pnl", "cash_after", "position_after"}


def test_failed_orders_not_in_history(free_broker):
    with pytest.raises(InsufficientFunds):
        free_broker.buy(100.0, 100.0)
    with pytest.raises(InsufficientPosition):
        free_broker.sell(100.0, 1.0)
    assert free_broker.trade_history() == []
