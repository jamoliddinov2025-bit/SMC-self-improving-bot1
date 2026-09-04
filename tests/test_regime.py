"""USDT.D regime detector tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import load_config  # noqa: E402
from src.strategy.regime import Regime, RegimeConfig, USDTDRegimeDetector  # noqa: E402

FAST = RegimeConfig(ema_fast=3, ema_slow=5, slope_lookback=2, roc_lookback=3,
                    slope_threshold_pct=0.1, roc_threshold_pct=0.5, confirm_bars=2)


def feed(det, closes):
    out = []
    for k, c in enumerate(closes):
        out.append(det.update(f"t{k}", c))
    return out


def test_warmup_is_unknown():
    det = USDTDRegimeDetector(FAST)
    states = feed(det, [4.0] * 4)
    assert all(s.regime is Regime.UNKNOWN for s in states)
    assert states[-1].close == 4.0 and states[-1].source_index == 3


def test_flat_series_is_neutral():
    det = USDTDRegimeDetector(FAST)
    states = feed(det, [4.0] * 15)
    assert states[-1].regime is Regime.NEUTRAL and states[-1].raw_regime is Regime.NEUTRAL


def test_rising_series_detected():
    det = USDTDRegimeDetector(FAST)
    states = feed(det, [4.0 * (1.02 ** k) for k in range(15)])
    assert states[-1].regime is Regime.RISING and states[-1].is_risk_off
    assert states[-1].slope_pct > 0 and states[-1].roc_pct > 0 and states[-1].close > states[-1].ema_slow


def test_falling_series_detected():
    det = USDTDRegimeDetector(FAST)
    states = feed(det, [4.0 * (0.98 ** k) for k in range(15)])
    assert states[-1].regime is Regime.FALLING and not states[-1].is_risk_off


def test_thresholds_boundary_neutral_when_move_too_small():
    strict = RegimeConfig(ema_fast=3, ema_slow=5, slope_lookback=2, roc_lookback=3,
                          slope_threshold_pct=5.0, roc_threshold_pct=20.0, confirm_bars=1)
    det = USDTDRegimeDetector(strict)
    states = feed(det, [4.0 * (1.02 ** k) for k in range(15)])
    assert states[-1].raw_regime is Regime.NEUTRAL


def test_hysteresis_requires_confirm_bars():
    det = USDTDRegimeDetector(FAST)
    closes = [4.0] * 12                                   # NEUTRAL established
    states = feed(det, closes)
    assert states[-1].regime is Regime.NEUTRAL
    # now a strong rise: raw flips to RISING immediately, adopted only after 2 agreeing bars
    s1 = det.update("r1", 4.6)
    s2 = det.update("r2", 5.2)
    s3 = det.update("r3", 5.9)
    raw = [s1.raw_regime, s2.raw_regime, s3.raw_regime]
    first_raw_rising = raw.index(Regime.RISING)
    adopted = [s1.regime, s2.regime, s3.regime]
    assert adopted[first_raw_rising] is Regime.NEUTRAL      # not yet
    assert Regime.RISING in adopted[first_raw_rising + 1:]  # adopted one bar later
    assert det.state.bars_in_regime >= 1


def test_confirm_bars_one_adopts_immediately():
    cfg = RegimeConfig(ema_fast=3, ema_slow=5, slope_lookback=2, roc_lookback=3,
                       slope_threshold_pct=0.1, roc_threshold_pct=0.5, confirm_bars=1)
    det = USDTDRegimeDetector(cfg)
    states = feed(det, [4.0] * 12 + [4.6, 5.2, 5.9])
    assert all(s.regime is s.raw_regime for s in states[-3:])


def test_incremental_is_prefix_invariant():
    closes = list(4 + np.sin(np.linspace(0, 6, 60)) * 0.3)
    full = feed(USDTDRegimeDetector(FAST), closes)
    for cut in (10, 25, 59):
        part = feed(USDTDRegimeDetector(FAST), closes[:cut])
        assert [s.regime for s in part] == [s.regime for s in full[:cut]]
        assert part[-1] == full[cut - 1]


def test_future_values_never_change_past_states():
    closes = list(4 + np.sin(np.linspace(0, 6, 40)) * 0.3)
    a = feed(USDTDRegimeDetector(FAST), closes)
    b = feed(USDTDRegimeDetector(FAST), closes[:20] + [c * 3 for c in closes[20:]])
    assert a[:20] == b[:20]


def test_config_loading_and_validation():
    cfg = load_config()
    rc = RegimeConfig.from_config(cfg)
    assert rc.enabled is True and rc.timeframe == "4h" and rc.ema_fast == 20 and rc.ema_slow == 50
    assert rc.confirm_bars == 2 and rc.slope_threshold_pct == 0.10
    for kw in (dict(ema_fast=50, ema_slow=20), dict(confirm_bars=0), dict(slope_lookback=0),
               dict(slope_threshold_pct=-1)):
        with pytest.raises(ValueError):
            RegimeConfig(**kw)
    assert RegimeConfig.from_config({}).enabled is False


def test_state_fields_are_point_in_time():
    det = USDTDRegimeDetector(FAST)
    s = feed(det, [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7])[-1]
    assert s.source_timestamp == "t7" and s.source_index == 7 and s.close == 4.7
