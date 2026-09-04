"""Deterministic SYNTHETIC long price series for Step 9 plumbing tests.

A seeded random walk with a mild upward drift and periodic impulse/pullback waves so that the
SMCStrategy (EMA-trend filter off) produces enough trades for the walk-forward guard to pass.
This is NOT market data and says nothing about performance.
"""

import copy

import numpy as np
import pandas as pd

from src.main import load_config


def synthetic_frame(n: int = 6000, seed: int = 7, start: str = "2023-01-01", vol: float = 0.0035,
                    wave_amp: float = 0.0015, wave_len: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    for i in range(n):
        drift = wave_amp * np.sin(2 * np.pi * i / wave_len)      # regular swing cycles -> BOS / pullbacks
        shock = rng.normal(0, vol)
        o = price
        c = o * (1 + drift + shock)
        h = max(o, c) + abs(rng.normal(0, 0.0015)) * o
        lo = min(o, c) - abs(rng.normal(0, 0.0015)) * o
        v = 100 + 60 * abs(rng.normal()) + (200 if abs(shock) > 0.006 else 0)
        rows.append((o, h, lo, c, v))
        price = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "timestamp", pd.date_range(start, periods=n, freq="15min", tz="UTC"))
    return df


def improvement_cfg(**over):
    """Sample config with improvement enabled and thresholds sized for the synthetic frame."""
    cfg = copy.deepcopy(load_config())
    cfg["usdtd"]["enabled"] = False
    cfg["strategy"]["filters"]["ema_trend"]["enabled"] = False
    cfg["improvement"]["enabled"] = True
    cfg["improvement"]["min_trades_before_change"] = 20
    cfg["improvement"]["data"]["min_trades_per_fold"] = 2
    for k, v in over.items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg
