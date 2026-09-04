"""StateStore atomic JSON + snapshot round-trips for PaperBroker, RiskState and SMCStrategy."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from smc_scenarios import contexts, pullback_scenario  # noqa: E402
from src.execution.paper_broker import PaperBroker  # noqa: E402
from src.execution.state_store import SCHEMA_VERSION, StateStore, config_hash  # noqa: E402
from src.main import load_config  # noqa: E402
from src.risk.state import RiskState  # noqa: E402
from src.strategy.smc_strategy import ARMED, IN_TRADE, SMCStrategy, SMCStrategyConfig  # noqa: E402
from tests.test_smc_strategy import BASE  # noqa: E402


def test_store_roundtrip_and_atomic_replace(tmp_path):
    st = StateStore(tmp_path / "sub" / "state.json")
    assert st.load() is None and not st.exists()
    st.save({"a": 1, "b": [1, 2]})
    assert st.exists()
    assert st.load() == {"a": 1, "b": [1, 2], "schema_version": SCHEMA_VERSION}
    st.save({"a": 2})
    assert st.load()["a"] == 2
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["state.json"]  # no temp files left behind


def test_store_rejects_wrong_schema(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError):
        StateStore(p).load()


def test_config_hash_only_covers_trading_sections():
    cfg = load_config()
    h = config_hash(cfg)
    other = json.loads(json.dumps(cfg))
    other["logging"]["level"] = "DEBUG"
    other["paper"] = {"warmup_bars": 5}
    assert config_hash(other) == h
    other["risk"]["risk_per_trade_pct"] = 2.0
    assert config_hash(other) != h


def test_broker_snapshot_roundtrip():
    b = PaperBroker(10_000, 0.1, "BTC/USDT")
    b.buy(100.0, 10.0, timestamp="t0")
    b.sell(110.0, 4.0, timestamp="t1")
    r = PaperBroker.from_snapshot(json.loads(json.dumps(b.to_snapshot())))
    assert r.cash == pytest.approx(b.cash) and r.position == pytest.approx(b.position)
    assert r.avg_entry_price == pytest.approx(b.avg_entry_price) and r.fee_rate == b.fee_rate
    assert r.realized_pnl == pytest.approx(b.realized_pnl) and r.total_fees() == pytest.approx(b.total_fees())
    assert r.equity(105.0) == pytest.approx(b.equity(105.0))
    r.sell(120.0, r.position)   # restored broker keeps working and keeps counting fees
    assert r.total_fees() > b.total_fees()


def test_risk_state_snapshot_roundtrip_keeps_locks_and_day():
    rs = RiskState(10_000)
    rs.new_day(dt.date(2024, 1, 5))
    rs.record_trade(-300.0)
    rs.record_trade(-200.0)
    rs.update_equity(9_400.0)
    rs.open_position()
    r = RiskState.from_snapshot(json.loads(json.dumps(rs.to_snapshot())))
    assert r.snapshot() == rs.snapshot()
    assert r.current_day == dt.date(2024, 1, 5) and r.day_start_equity == rs.day_start_equity
    assert r.consecutive_losses == 2 and r.open_positions == 1
    assert r.daily_loss_pct == pytest.approx(rs.daily_loss_pct) and r.drawdown_pct == pytest.approx(rs.drawdown_pct)
    assert r.new_day(dt.date(2024, 1, 5)) is False   # same day -> no roll after restart


def _armed_strategy_and_ctx(upto):
    s = SMCStrategy(SMCStrategyConfig(**BASE))
    rows = pullback_scenario()[: upto + 1]
    out = contexts(rows, s)
    return s, out[upto][2].smc


def test_strategy_snapshot_roundtrip_armed_setup():
    s, smc = _armed_strategy_and_ctx(10)
    assert s.state == ARMED and s.setup is not None
    snap = json.loads(json.dumps(s.to_snapshot()))
    r = SMCStrategy(SMCStrategyConfig(**BASE))
    warnings = r.restore_snapshot(snap, smc)
    assert warnings == []
    assert r.state == ARMED and r.setup.trigger_bos is s.setup.trigger_bos   # relinked by key
    assert [(p.kind, p.low, p.high, p.mitigated, p.skipped) for p in r.setup.pois] == \
           [(p.kind, p.low, p.high, p.mitigated, p.skipped) for p in s.setup.pois]
    assert r._handled_bos is s._handled_bos and r._seen_bos == len(smc.bos_events)


def test_strategy_snapshot_roundtrip_in_trade_and_index_shift():
    s, smc = _armed_strategy_and_ctx(11)
    assert s.state == IN_TRADE and s._pending_entry is not None
    snap = json.loads(json.dumps(s.to_snapshot()))
    r = SMCStrategy(SMCStrategyConfig(**BASE))
    r.restore_snapshot(snap, smc, index_shift=0)
    assert r.state == IN_TRADE and r._pending_entry.stop == s._pending_entry.stop
    assert r.setup.entries == 1 and r.diag.buy_signals == 1
    # a shift moves every bar index consistently
    r2 = SMCStrategy(SMCStrategyConfig(**BASE))
    r2.restore_snapshot(snap, smc, index_shift=-5)
    assert r2._pending_entry.entry_index == s._pending_entry.entry_index - 5
    assert r2.setup.trigger_index == s.setup.trigger_index - 5
    assert r2.setup.pois[0].created_index == s.setup.pois[0].created_index - 5


def test_strategy_restore_with_missing_bos_drops_setup_safely():
    s, smc = _armed_strategy_and_ctx(10)
    snap = json.loads(json.dumps(s.to_snapshot()))
    from src.strategy.smc_types import SMCResult
    r = SMCStrategy(SMCStrategyConfig(**BASE))
    warnings = r.restore_snapshot(snap, SMCResult())
    assert r.setup is None and r.state == "IDLE" and len(warnings) == 2
