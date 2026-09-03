"""Tests for the risk engine: sizing, validator kill switches, risk state."""

import datetime as dt
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import load_config  # noqa: E402
from src.risk import (  # noqa: E402
    RiskConfig,
    RiskDecision,
    RiskState,
    TradeProposal,
    TradeValidator,
    calculate_position_size,
)

CFG = RiskConfig(risk_per_trade_pct=1.0, max_daily_loss_pct=3.0, max_drawdown_pct=10.0,
                 max_consecutive_losses=5, max_open_positions=1)


def proposal(**kw):
    base = dict(account_equity=10_000.0, entry_price=100.0, stop_loss_price=98.0)
    base.update(kw)
    return TradeProposal(**base)


@pytest.fixture
def validator():
    return TradeValidator(CFG)


# ---------------------------------------------------------- position sizing
def test_position_size_basic():
    s = calculate_position_size(10_000, 100, 98, 1.0)
    assert s.valid
    assert s.risk_amount == pytest.approx(100.0)
    assert s.stop_distance == pytest.approx(2.0)
    assert s.position_size == pytest.approx(50.0)
    assert s.effective_risk_pct == pytest.approx(1.0)


def test_position_size_scales_with_risk_pct_and_equity():
    a = calculate_position_size(10_000, 100, 98, 1.0).position_size
    b = calculate_position_size(10_000, 100, 98, 2.0).position_size
    c = calculate_position_size(20_000, 100, 98, 1.0).position_size
    assert b == pytest.approx(2 * a) and c == pytest.approx(2 * a)


def test_position_size_btc_like_numbers():
    s = calculate_position_size(10_000, 42_000, 41_370, 1.0)  # 1.5% stop
    assert s.risk_amount == pytest.approx(100.0)
    assert s.position_size == pytest.approx(100 / 630)
    assert s.position_size * s.stop_distance == pytest.approx(100.0)


def test_position_size_capped_by_available_cash():
    # 1% risk with a 0.1% stop would need 10x equity -> capped at 100% notional
    s = calculate_position_size(10_000, 100, 99.9, 1.0)
    assert s.valid
    assert s.position_size * 100 == pytest.approx(10_000)
    assert s.effective_risk_pct == pytest.approx(0.1)
    assert s.risk_amount == pytest.approx(10.0)


def test_zero_stop_distance_invalid():
    s = calculate_position_size(10_000, 100, 100, 1.0)
    assert not s.valid and s.position_size == 0.0 and "below entry" in s.reason


def test_negative_stop_distance_invalid():
    s = calculate_position_size(10_000, 100, 105, 1.0)
    assert not s.valid and s.stop_distance == pytest.approx(-5.0)


def test_invalid_equity():
    assert not calculate_position_size(0, 100, 98, 1.0).valid
    assert not calculate_position_size(-1, 100, 98, 1.0).valid


def test_invalid_inputs_never_raise_and_never_negative():
    for args in [(math.nan, 100, 98, 1.0), (10_000, math.inf, 98, 1.0), (10_000, 100, 98, 0.0),
                 (10_000, 100, 98, 150.0), (10_000, -100, 98, 1.0), (10_000, 100, -1, 1.0),
                 ("1000", 100, 98, 1.0), (True, 100, 98, 1.0)]:
        s = calculate_position_size(*args)
        assert not s.valid and s.position_size >= 0.0


# ----------------------------------------------------------------- validator
def test_approved_trade(validator):
    a = validator.validate(proposal())
    assert a.decision is RiskDecision.APPROVED and a.approved
    assert a.sizing.position_size == pytest.approx(50.0)
    assert a.checks["open_positions"] == "0/1"


def test_invalid_stop_rejected(validator):
    for stop in (100.0, 101.0):
        a = validator.validate(proposal(stop_loss_price=stop))
        assert a.decision is RiskDecision.REJECTED_INVALID_STOP and not a.approved and a.sizing is None


def test_invalid_equity_rejected(validator):
    for eq in (0.0, -5.0):
        assert validator.validate(proposal(account_equity=eq)).decision is RiskDecision.REJECTED_INVALID_EQUITY


def test_daily_loss_lock(validator):
    assert validator.validate(proposal(current_daily_loss_pct=2.99)).approved
    assert validator.validate(proposal(current_daily_loss_pct=3.0)).decision is RiskDecision.REJECTED_DAILY_LOSS_LIMIT
    assert validator.validate(proposal(current_daily_loss_pct=7.5)).decision is RiskDecision.REJECTED_DAILY_LOSS_LIMIT


def test_drawdown_lock(validator):
    assert validator.validate(proposal(current_drawdown_pct=9.99)).approved
    assert validator.validate(proposal(current_drawdown_pct=10.0)).decision is RiskDecision.REJECTED_DRAWDOWN_LIMIT


def test_consecutive_loss_lock(validator):
    assert validator.validate(proposal(consecutive_losses=4)).approved
    assert validator.validate(proposal(consecutive_losses=5)).decision is RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT


def test_position_limit_lock(validator):
    assert validator.validate(proposal(open_positions=1)).decision is RiskDecision.REJECTED_POSITION_LIMIT
    v2 = TradeValidator(RiskConfig(max_open_positions=2))
    assert v2.validate(proposal(open_positions=1)).approved
    assert v2.validate(proposal(open_positions=2)).decision is RiskDecision.REJECTED_POSITION_LIMIT


def test_invalid_inputs_rejected(validator):
    cases = [
        dict(entry_price=math.nan), dict(stop_loss_price=math.inf), dict(entry_price=0.0),
        dict(stop_loss_price=-1.0), dict(current_drawdown_pct=-1.0), dict(current_daily_loss_pct=-0.5),
        dict(consecutive_losses=-1), dict(open_positions=-1), dict(consecutive_losses=1.5),
        dict(account_equity="10000"), dict(open_positions=True),
    ]
    for kw in cases:
        a = validator.validate(proposal(**kw))
        assert a.decision is RiskDecision.REJECTED_INVALID_INPUT, kw


def test_check_precedence_position_limit_before_daily_loss(validator):
    a = validator.validate(proposal(open_positions=1, current_daily_loss_pct=5.0, consecutive_losses=9))
    assert a.decision is RiskDecision.REJECTED_POSITION_LIMIT


def test_check_precedence_kill_switch_before_invalid_stop(validator):
    a = validator.validate(proposal(current_drawdown_pct=50.0, stop_loss_price=200.0))
    assert a.decision is RiskDecision.REJECTED_DRAWDOWN_LIMIT


def test_decision_is_enum_and_str():
    assert isinstance(RiskDecision.APPROVED, str)
    assert RiskDecision("REJECTED_POSITION_LIMIT") is RiskDecision.REJECTED_POSITION_LIMIT
    assert len(RiskDecision) == 8


# -------------------------------------------------------------- risk state
def test_state_initial_values():
    s = RiskState(10_000)
    assert s.equity == s.peak_equity == s.day_start_equity == 10_000
    assert s.drawdown_pct == 0.0 and s.daily_loss_pct == 0.0 and s.consecutive_losses == 0
    with pytest.raises(ValueError):
        RiskState(0)


def test_drawdown_calculation():
    s = RiskState(10_000)
    s.record_trade(-500)
    assert s.equity == 9_500 and s.peak_equity == 10_000
    assert s.drawdown_pct == pytest.approx(5.0)
    s.record_trade(-450)
    assert s.drawdown_pct == pytest.approx(9.5)


def test_equity_peak_updates_and_drawdown_from_new_peak():
    s = RiskState(10_000)
    s.record_trade(+2_000)
    assert s.peak_equity == 12_000 and s.drawdown_pct == 0.0
    s.record_trade(-1_200)
    assert s.equity == 10_800 and s.drawdown_pct == pytest.approx(10.0)
    s.update_equity(12_500)  # mark-to-market above peak
    assert s.peak_equity == 12_500 and s.drawdown_pct == 0.0
    s.update_equity(11_250)
    assert s.drawdown_pct == pytest.approx(10.0)


def test_update_equity_rejects_negative():
    with pytest.raises(ValueError):
        RiskState(100).update_equity(-1)


def test_loss_streak_tracking():
    s = RiskState(10_000)
    s.record_trade(-10); s.record_trade(-10); s.record_trade(-10)
    assert s.consecutive_losses == 3
    s.record_trade(0.0)
    assert s.consecutive_losses == 3  # breakeven leaves the streak untouched
    s.record_trade(+5)
    assert s.consecutive_losses == 0
    s.record_trade(-1)
    assert s.consecutive_losses == 1


def test_daily_pnl_and_loss_pct():
    s = RiskState(10_000)
    s.record_trade(-150)
    s.record_trade(-150)
    assert s.daily_pnl == -300 and s.daily_loss_pct == pytest.approx(3.0)
    s.record_trade(+400)
    assert s.daily_pnl == 100 and s.daily_pnl_pct == pytest.approx(1.0) and s.daily_loss_pct == 0.0


def test_daily_reset_logic():
    d1, d2 = dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    s = RiskState(10_000)
    s.record_trade(-300, day=d1)
    assert s.daily_loss_pct == pytest.approx(3.0) and s.current_day == d1
    assert s.new_day(d1) is False                  # same day -> no reset
    assert s.daily_pnl == -300
    assert s.new_day(d2) is True                   # new day -> reset
    assert s.daily_pnl == 0.0 and s.daily_loss_pct == 0.0
    assert s.day_start_equity == 9_700             # today's % is relative to today's start
    # drawdown, peak and loss streak survive the day roll
    assert s.drawdown_pct == pytest.approx(3.0) and s.consecutive_losses == 1
    s.record_trade(-97, day=d2)
    assert s.daily_loss_pct == pytest.approx(1.0)


def test_record_trade_with_day_rolls_automatically():
    s = RiskState(10_000)
    s.record_trade(-100, day=dt.date(2024, 1, 1))
    s.record_trade(-100, day=dt.date(2024, 1, 2))
    assert s.daily_pnl == -100 and s.equity == 9_800


def test_open_close_position_counter():
    s = RiskState(10_000)
    s.open_position()
    assert s.open_positions == 1
    s.close_position()
    assert s.open_positions == 0
    with pytest.raises(ValueError):
        s.close_position()


def test_snapshot_keys():
    keys = set(RiskState(1_000).snapshot())
    assert keys == {"equity", "peak_equity", "drawdown_pct", "daily_pnl", "daily_loss_pct",
                    "consecutive_losses", "open_positions"}


# ------------------------------------------------------- validator + state
def test_validate_with_state_end_to_end(validator):
    s = RiskState(10_000)
    assert validator.validate_with_state(s, 100, 98).approved
    s.open_position()
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_POSITION_LIMIT
    s.close_position()
    for _ in range(5):
        s.record_trade(-50, day=dt.date(2024, 1, 1))  # 2.5% daily loss, 5 losses in a row
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT
    s.record_trade(+1, day=dt.date(2024, 1, 1))       # streak broken, still 2.49% down today
    a = validator.validate_with_state(s, 100, 98)
    assert a.approved and a.sizing.risk_amount == pytest.approx(s.equity * 0.01)
    s.record_trade(-60, day=dt.date(2024, 1, 1))      # now >= 3% daily loss
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_DAILY_LOSS_LIMIT
    s.new_day(dt.date(2024, 1, 2))
    assert validator.validate_with_state(s, 100, 98).approved


def test_drawdown_lock_via_state(validator):
    s = RiskState(10_000)
    s.update_equity(9_000)
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_DRAWDOWN_LIMIT


# ----------------------------------------------------------- configuration
def test_config_loading():
    cfg = load_config()
    rc = RiskConfig.from_config(cfg)
    assert rc.risk_per_trade_pct == 1.0
    assert rc.max_daily_loss_pct == 3.0
    assert rc.max_drawdown_pct == 10.0
    assert rc.max_consecutive_losses == 5
    assert rc.max_open_positions == 1
    v = TradeValidator.from_config(cfg)
    assert v.validate(proposal(account_equity=cfg["risk"]["starting_balance"])).approved


def test_config_validation():
    for kw in (dict(risk_per_trade_pct=0), dict(risk_per_trade_pct=101), dict(max_daily_loss_pct=0),
               dict(max_drawdown_pct=-1), dict(max_consecutive_losses=0), dict(max_open_positions=0)):
        with pytest.raises(ValueError):
            RiskConfig(**kw)


# ------------------------------------------- loss-streak deadlock (documented)
def test_loss_streak_lock_is_not_a_permanent_deadlock(validator):
    """5 losses -> locked. Nothing automatic clears it (documented), but the
    explicit `reset_loss_streak` hook does."""
    s = RiskState(10_000)
    for _ in range(5):
        s.record_trade(-10, day=dt.date(2024, 1, 1))
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT

    # a new day does NOT clear the streak (current behaviour, kept on purpose)
    s.new_day(dt.date(2024, 1, 2))
    assert s.consecutive_losses == 5
    assert validator.validate_with_state(s, 100, 98).decision is RiskDecision.REJECTED_CONSECUTIVE_LOSS_LIMIT

    # the documented extension point clears it
    assert s.reset_loss_streak(reason="cooldown") == 5
    assert s.consecutive_losses == 0 and s.last_streak_reset_reason == "cooldown"
    assert validator.validate_with_state(s, 100, 98).approved


def test_reset_loss_streak_does_not_touch_other_state():
    s = RiskState(10_000)
    s.record_trade(-500)
    s.reset_loss_streak()
    assert s.equity == 9_500 and s.drawdown_pct == pytest.approx(5.0) and s.daily_pnl == -500
    assert s.last_streak_reset_reason == "manual"
