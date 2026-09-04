"""Indicator engine: computes every configured indicator on an OHLCV frame.

Input : DataFrame with columns timestamp, open, high, low, close, volume
Output: the same frame (copy) plus aligned indicator columns:
        ema_<p> for each configured period, atr_<p>, true_range,
        volume_sma_<p>, volume_ratio_<p>.
Warm-up rows contain NaN. No trading decisions are made here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.indicators.atr import atr, true_range
from src.indicators.ema import ema
from src.indicators.volume import volume_ratio, volume_sma

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class IndicatorConfig:
    ema_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    atr_period: int = 14
    volume_ma_period: int = 20

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "IndicatorConfig":
        ind = config["indicators"]
        periods = list(ind["ema"].get("periods", []))
        for alias in ("fast", "slow"):
            if alias in ind["ema"] and ind["ema"][alias] not in periods:
                periods.append(ind["ema"][alias])
        return cls(
            ema_periods=sorted(set(int(p) for p in periods)),
            atr_period=int(ind["atr"]["period"]),
            volume_ma_period=int(ind["volume"]["ma_period"]),
        )

    @property
    def warmup_bars(self) -> int:
        """Number of leading rows that will contain at least one NaN."""
        return max(max(self.ema_periods, default=1), self.atr_period, self.volume_ma_period) - 1


class IndicatorEngine:
    def __init__(self, config: IndicatorConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "IndicatorEngine":
        return cls(IndicatorConfig.from_config(config))

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV frame missing columns: {missing}")

        out = df.copy()
        for p in self.config.ema_periods:
            out[f"ema_{p}"] = ema(out["close"], p)
        out["true_range"] = true_range(out)
        out[f"atr_{self.config.atr_period}"] = atr(out, self.config.atr_period)
        out[f"volume_sma_{self.config.volume_ma_period}"] = volume_sma(out["volume"], self.config.volume_ma_period)
        out[f"volume_ratio_{self.config.volume_ma_period}"] = volume_ratio(out["volume"], self.config.volume_ma_period)
        return out

    @property
    def columns(self) -> List[str]:
        c = self.config
        return (
            [f"ema_{p}" for p in c.ema_periods]
            + ["true_range", f"atr_{c.atr_period}", f"volume_sma_{c.volume_ma_period}", f"volume_ratio_{c.volume_ma_period}"]
        )
