"""Data splitting: sealed final holdout + anchored walk-forward folds over the development window.

    |<--------------------- development (1 - holdout_pct) --------------------->|<- holdout ->|
    | IS_1                 | OOS_1 |
    | IS_1 + ...                   | OOS_2 |            (anchored: IS always starts at bar 0)
    | ...                                  | OOS_3 | ...

Each slice is expressed as [start, end) bar indices into the FULL primary frame. Because the
engine needs indicator / structure warm-up, `Evaluator` feeds `warmup_bars` of extra history
before `start` and discards trades whose entry timestamp precedes the slice start. The holdout
is never used by the search; only top-N survivors are evaluated on it once.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class Slice:
    name: str
    start: int          # inclusive bar index
    end: int            # exclusive bar index
    role: str           # "is" | "oos" | "holdout" | "development"

    @property
    def bars(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Fold:
    k: int
    train: Slice
    test: Slice


@dataclass(frozen=True)
class SplitPlan:
    n_bars: int
    development: Slice
    holdout: Slice
    folds: List[Fold]

    def to_dict(self) -> Dict[str, Any]:
        def sd(s: Slice) -> Dict[str, Any]:
            return {**vars(s), "bars": s.bars}
        return {
            "n_bars": self.n_bars,
            "development": sd(self.development), "holdout": sd(self.holdout),
            "folds": [{"k": f.k, "train": sd(f.train), "test": sd(f.test)} for f in self.folds],
        }


@dataclass(frozen=True)
class SplitConfig:
    holdout_pct: float = 20.0
    folds: int = 4
    oos_pct_per_fold: float = 20.0
    anchored: bool = True
    min_trades_per_fold: int = 8

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SplitConfig":
        d = (config.get("improvement", {}) or {}).get("data", {}) or {}
        c = cls(float(d.get("holdout_pct", 20.0)), int(d.get("folds", 4)), float(d.get("oos_pct_per_fold", 20.0)),
                bool(d.get("anchored", True)), int(d.get("min_trades_per_fold", 8)))
        if not (0 < c.holdout_pct < 50) or c.folds < 1 or not (0 < c.oos_pct_per_fold < 50):
            raise ValueError("improvement.data: need 0<holdout_pct<50, folds>=1, 0<oos_pct_per_fold<50")
        return c


class DataSplitter:
    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg

    def plan(self, n_bars: int) -> SplitPlan:
        if n_bars < 10:
            raise ValueError("not enough bars to split")
        dev_end = n_bars - int(round(n_bars * self.cfg.holdout_pct / 100.0))
        development = Slice("development", 0, dev_end, "development")
        holdout = Slice("holdout", dev_end, n_bars, "holdout")
        oos_len = int(round(dev_end * self.cfg.oos_pct_per_fold / 100.0))
        if oos_len < 1:
            raise ValueError("development window too small for the requested OOS fraction")
        # last fold's OOS ends exactly at dev_end; earlier folds step back by oos_len each
        folds: List[Fold] = []
        for k in range(1, self.cfg.folds + 1):
            test_end = dev_end - (self.cfg.folds - k) * oos_len
            test_start = test_end - oos_len
            train_start = 0 if self.cfg.anchored else max(0, test_start - (dev_end - self.cfg.folds * oos_len))
            if test_start <= train_start:
                raise ValueError(f"fold {k}: no in-sample bars left (n_bars={n_bars}); reduce folds/oos_pct")
            folds.append(Fold(k, Slice(f"is_{k}", train_start, test_start, "is"),
                              Slice(f"oos_{k}", test_start, test_end, "oos")))
        plan = SplitPlan(n_bars, development, holdout, folds)
        self._check(plan)
        return plan

    @staticmethod
    def _check(plan: SplitPlan) -> None:
        for f in plan.folds:
            assert f.test.end <= plan.holdout.start, "OOS fold overlaps the sealed holdout"
            assert f.train.end == f.test.start, "IS/OOS must be contiguous"
        for a, b in zip(plan.folds, plan.folds[1:]):
            assert a.test.end == b.test.start, "OOS folds must tile the tail of the development window"

    @staticmethod
    def timestamps(df: pd.DataFrame, s: Slice):
        return df["timestamp"].iloc[s.start], df["timestamp"].iloc[s.end - 1]
