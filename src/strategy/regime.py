"""USDT.D market-regime detector (incremental, point-in-time).

Hypothesis (NOT assumed to be always true - every effect is configurable):
    USDT dominance rising  -> risk-off  -> be more selective with longs
    USDT dominance falling -> risk-on   -> normal SMC rules
    neutral / unknown      -> normal SMC rules

Regime rules, evaluated on the USDT.D close series only (all backward-looking):
    ema_fast = EMA(close, ema_fast)        ema_slow = EMA(close, ema_slow)
    slope    = (ema_fast[t] - ema_fast[t-slope_lookback]) / ema_fast[t-slope_lookback]
    roc      = (close[t] - close[t-roc_lookback]) / close[t-roc_lookback]

    raw = RISING  if close > ema_slow and slope > +slope_thr and roc > +roc_thr
    raw = FALLING if close < ema_slow and slope < -slope_thr and roc < -roc_thr
    raw = NEUTRAL otherwise
    raw = UNKNOWN while any input is not yet available (warm-up)

Hysteresis: a new raw regime is adopted only after `confirm_bars` consecutive
USDT.D bars agree. UNKNOWN is adopted immediately (fail-open to normal rules).

This module never touches RiskState or TradeValidator. It only emits RegimeState.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Regime(str, Enum):
    RISING = "RISING"      # risk-off
    FALLING = "FALLING"    # risk-on
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"    # no data / warm-up / disabled -> treated as NEUTRAL by the strategy


@dataclass(frozen=True)
class RegimeState:
    regime: Regime
    raw_regime: Regime
    close: Optional[float]
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    slope_pct: Optional[float]
    roc_pct: Optional[float]
    bars_in_regime: int
    source_timestamp: Any            # open timestamp of the USDT.D candle used
    source_index: int                # index of that candle in the USDT.D series (-1 if none)

    @property
    def is_risk_off(self) -> bool:
        return self.regime is Regime.RISING


UNKNOWN_STATE = RegimeState(Regime.UNKNOWN, Regime.UNKNOWN, None, None, None, None, None, 0, None, -1)


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = True
    symbol: str = "USDT.D"
    timeframe: str = "4h"
    ema_fast: int = 20
    ema_slow: int = 50
    slope_lookback: int = 6
    roc_lookback: int = 12
    slope_threshold_pct: float = 0.10
    roc_threshold_pct: float = 0.50
    confirm_bars: int = 2

    def __post_init__(self):
        if self.ema_fast < 1 or self.ema_slow < 1 or self.ema_fast >= self.ema_slow:
            raise ValueError("require 1 <= ema_fast < ema_slow")
        if self.slope_lookback < 1 or self.roc_lookback < 1:
            raise ValueError("lookbacks must be >= 1")
        if self.slope_threshold_pct < 0 or self.roc_threshold_pct < 0:
            raise ValueError("thresholds must be >= 0")
        if self.confirm_bars < 1:
            raise ValueError("confirm_bars must be >= 1")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegimeConfig":
        u = config.get("usdtd", {}) or {}
        return cls(
            enabled=bool(u.get("enabled", False)),
            symbol=str(u.get("symbol", "USDT.D")),
            timeframe=str(u.get("timeframe", "4h")),
            ema_fast=int(u.get("ema_fast", 20)),
            ema_slow=int(u.get("ema_slow", 50)),
            slope_lookback=int(u.get("slope_lookback", 6)),
            roc_lookback=int(u.get("roc_lookback", 12)),
            slope_threshold_pct=float(u.get("slope_threshold_pct", 0.10)),
            roc_threshold_pct=float(u.get("roc_threshold_pct", 0.50)),
            confirm_bars=int(u.get("confirm_bars", 2)),
        )


class _IncEMA:
    """SMA-seeded EMA, identical semantics to src/indicators/ema.py, but incremental."""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self._seed: List[float] = []
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class USDTDRegimeDetector:
    """Feed one USDT.D candle close at a time via `update`. Never sees future candles."""

    def __init__(self, config: RegimeConfig):
        self.config = config
        self._fast = _IncEMA(config.ema_fast)
        self._slow = _IncEMA(config.ema_slow)
        self._closes: List[float] = []
        self._fast_hist: List[float] = []
        self._current = Regime.UNKNOWN
        self._candidate = Regime.UNKNOWN
        self._candidate_count = 0
        self._bars_in_regime = 0
        self._n = 0
        self.state: RegimeState = UNKNOWN_STATE

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "USDTDRegimeDetector":
        return cls(RegimeConfig.from_config(config))

    def update(self, timestamp: Any, close: float) -> RegimeState:
        c = self.config
        close = float(close)
        idx = self._n
        self._n += 1
        self._closes.append(close)
        fast = self._fast.update(close)
        slow = self._slow.update(close)
        if fast is not None:
            self._fast_hist.append(fast)

        slope = roc = None
        if fast is not None and len(self._fast_hist) > c.slope_lookback:
            ref = self._fast_hist[-1 - c.slope_lookback]
            slope = (fast - ref) / ref * 100.0 if ref else None
        if len(self._closes) > c.roc_lookback:
            ref = self._closes[-1 - c.roc_lookback]
            roc = (close - ref) / ref * 100.0 if ref else None

        if slow is None or slope is None or roc is None:
            raw = Regime.UNKNOWN
        elif close > slow and slope > c.slope_threshold_pct and roc > c.roc_threshold_pct:
            raw = Regime.RISING
        elif close < slow and slope < -c.slope_threshold_pct and roc < -c.roc_threshold_pct:
            raw = Regime.FALLING
        else:
            raw = Regime.NEUTRAL

        self._apply_hysteresis(raw)
        self.state = RegimeState(self._current, raw, close, fast, slow, slope, roc,
                                 self._bars_in_regime, timestamp, idx)
        return self.state

    def _apply_hysteresis(self, raw: Regime) -> None:
        c = self.config
        if raw is Regime.UNKNOWN:
            self._current, self._candidate, self._candidate_count, self._bars_in_regime = raw, raw, 0, 0
            return
        if raw is self._current:
            self._candidate, self._candidate_count = raw, 0
            self._bars_in_regime += 1
            return
        if raw is self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = raw, 1
        if self._candidate_count >= c.confirm_bars or self._current is Regime.UNKNOWN:
            self._current = raw
            self._candidate_count = 0
            self._bars_in_regime = 1
        else:
            self._bars_in_regime += 1
