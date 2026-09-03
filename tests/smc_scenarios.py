"""Hand-crafted OHLCV scenarios shared by the Step 7 tests.

All candles: (open, high, low, close, volume). Pivot strength 2 is used so datasets stay small.
Index map for `pullback_scenario()` (pivot_strength=2):

  0-1  flat 100
  2    swing high 110 (SH confirmed at 4)
  3-4  flat 100 (bar 4 confirms SH)
  5    swing low  90 (SL confirmed at 7)   -> impulse start
  6    bearish candle 101->99            -> becomes the bullish ORDER BLOCK
  7    strong bullish 99->112 (hi 113)    -> bullish BOS (close 112 > 110), OB = bar 6
  8    108-116 ... continuation, FVG between bar6.high(102) and bar8.low(108) -> FVG [102,108]
  9-10 drift 114
  11   PULLBACK INTO OB: open 112, low 100 (inside OB [99,102]), close 104 (bullish, upper half)
"""

import pandas as pd

from src.backtesting.strategy import BacktestContext
from src.data.base import Candle
from src.indicators import IndicatorConfig, IndicatorEngine
from src.strategy.smc_engine import SMCConfig, SMCEngine

IND = IndicatorEngine(IndicatorConfig(ema_periods=[3, 5, 8], atr_period=3, volume_ma_period=3))
IND_COLS = dict(atr_col="atr_3", ema_fast_col="ema_3", ema_mid_col="ema_5", ema_slow_col="ema_8",
                vol_ratio_col="volume_ratio_3")


def frame(rows, start="2024-01-01"):
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(rows), freq="15min", tz="UTC"),
        "open": [float(r[0]) for r in rows], "high": [float(r[1]) for r in rows],
        "low": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows],
        "volume": [float(r[4]) if len(r) > 4 else 100.0 for r in rows],
    })


def flat(p, n=1, v=100.0):
    return [(p, p + 1, p - 1, p, v)] * n


def pullback_scenario(rejection_close=104.0, pull_low=100.0, pull_open=112.0, vol=100.0):
    rows = (
        flat(100, 2)
        + [(100, 110, 99, 101)]                 # 2  swing high 110
        + flat(100, 2)                          # 3,4
        + [(100, 101, 90, 100)]                 # 5  swing low 90
        + [(101, 102, 98, 99)]                  # 6  bearish -> order block [98, 102] (low 98, high 102)
        + [(99, 113, 98.5, 112, 150.0)]         # 7  BOS candle (close 112 > 110), volume spike
        + [(112, 116, 108, 114)]                # 8  gap: bar6.high 102 < bar8.low 108 -> FVG [102,108]
        + [(114, 115, 113, 114), (114, 115, 113, 114)]  # 9,10
        + [(pull_open, pull_open + 0.5, pull_low, rejection_close, vol)]  # 11 pullback into OB
    )
    return rows


def contexts(rows, strategy=None, regime_fn=None, positions=None):
    """Replay rows through SMCEngine + indicators exactly like the backtester and call strategy per bar.

    Returns list of (index, signal). `positions` may map index -> has_position to emulate fills.
    `regime_fn(index) -> RegimeState | None` injects a USDT.D regime.
    """
    df = frame(rows)
    ind = IND.compute(df)[IND.columns]
    smc = SMCEngine(SMCConfig(pivot_strength=2))
    out = []
    has_pos = False
    for i, r in enumerate(df.itertuples(index=False)):
        c = Candle(r.timestamp, r.open, r.high, r.low, r.close, r.volume)
        smc.update(c.timestamp, c.open, c.high, c.low, c.close)
        if positions is not None:
            has_pos = positions.get(i, has_pos)
        ctx = BacktestContext(i, c, ind.iloc[i], smc.result, has_pos, 10_000.0, {},
                              regime=regime_fn(i) if regime_fn else None)
        sig = strategy.on_candle(ctx) if strategy else None
        out.append((i, sig, ctx))
    return out
