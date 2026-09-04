"""Tunable parameter space for the Controlled Improvement Framework (Step 9).

Only the parameters listed in WHITELIST may ever be searched. Risk-engine limits, fees,
slippage, starting balance, symbol/timeframe and USDT.D thresholds are NOT tunable and
`ParameterSpace` raises if config tries to add them.

A *candidate* is a plain overlay `{config_path: value}` applied to an in-memory copy of
the config (`apply_overlay`). Nothing here writes files.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# config path -> (type, min, max, step, choices)
WHITELIST: Dict[str, Tuple[str, Optional[float], Optional[float], Optional[float], Optional[tuple]]] = {
    "strategy.entry.setup_max_age_bars":            ("int",   20,   120,  10,   None),
    "strategy.entry.poi_max_atr_multiple":          ("float", 1.5,  5.0,  0.5,  None),
    "strategy.entry.rejection_close_position_min":  ("float", 0.3,  0.7,  0.1,  None),
    "strategy.entry.reentry_cooldown_bars":         ("int",   0,    12,   3,    None),
    "strategy.filters.ema_trend.enabled":           ("bool",  None, None, None, (False, True)),
    "strategy.filters.ema_extension.enabled":       ("bool",  None, None, None, (False, True)),
    "strategy.filters.ema_extension.atr_multiple":  ("float", 0.5,  2.0,  0.25, None),
    "strategy.filters.volume.enabled":              ("bool",  None, None, None, (False, True)),
    "strategy.filters.volume.ratio_min":            ("float", 0.5,  1.5,  0.1,  None),
    "strategy.filters.volume.bos_ratio_min":        ("float", 1.0,  2.0,  0.2,  None),
    "strategy.stops.buffer_atr":                    ("float", 0.1,  0.5,  0.05, None),
    "strategy.stops.min_stop_atr":                  ("float", 0.3,  1.0,  0.1,  None),
    "strategy.stops.max_stop_atr":                  ("float", 2.0,  4.0,  0.5,  None),
    "strategy.targets.mode":                        ("enum",  None, None, None, ("structure", "fixed_rr", "hybrid")),
    "strategy.targets.fixed_rr":                    ("float", 1.5,  4.0,  0.5,  None),
    "strategy.exits.max_bars_in_trade":             ("int",   48,   192,  24,   None),
    "strategy.exits.exit_on_bearish_choch":         ("bool",  None, None, None, (False, True)),
}

# never tunable, whatever the config says
FROZEN_PREFIXES = ("risk.", "execution.", "market.", "usdtd.", "data.", "paper.", "backtesting.", "indicators.",
                   "strategy.structure.", "strategy.liquidity.", "strategy.order_blocks.", "strategy.fair_value_gaps.")


def _decimals(step: Optional[float]) -> int:
    if not step:
        return 6
    txt = f"{step:.10f}".rstrip("0")
    return max(2, len(txt.split(".")[1]) if "." in txt else 0)


class SpaceError(ValueError):
    pass


def get_path(config: Dict[str, Any], path: str) -> Any:
    d: Any = config
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            raise SpaceError(f"config path {path!r} not found")
        d = d[p]
    return d


def set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    d = config
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def apply_overlay(config: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    for k, v in overlay.items():
        set_path(out, k, v)
    return out


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    kind: str                      # int | float | bool | enum
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[tuple] = None

    def grid(self) -> List[Any]:
        """Every admissible value in deterministic ascending order."""
        if self.kind in ("bool", "enum"):
            return list(self.choices)
        vals, x, n = [], self.min, 0
        while x <= self.max + 1e-9:
            vals.append(int(round(x)) if self.kind == "int" else round(x, 10))
            n += 1
            x = self.min + n * self.step
        return vals

    def coerce(self, v: Any) -> Any:
        if self.kind == "int":
            return int(round(float(v)))
        if self.kind == "float":
            return round(float(v), 10)
        return v

    def in_bounds(self, v: Any) -> bool:
        if self.kind in ("bool", "enum"):
            return v in self.choices
        return self.min - 1e-9 <= float(v) <= self.max + 1e-9


class ParameterSpace:
    def __init__(self, config: Dict[str, Any]):
        imp = config.get("improvement", {}) or {}
        self.max_change_pct = float(imp.get("max_parameter_change_pct", 10))
        self.max_params_changed = int(imp.get("max_params_changed_per_proposal", 2))
        requested = imp.get("parameters")
        self.specs: Dict[str, ParameterSpec] = {}
        items = WHITELIST.items() if requested is None else [(k, None) for k in requested]
        for path, wl in items:
            if any(path.startswith(pfx) for pfx in FROZEN_PREFIXES) or path not in WHITELIST:
                raise SpaceError(f"parameter {path!r} is not tunable (not in whitelist)")
            kind, lo, hi, step, choices = WHITELIST[path]
            over = (requested or {}).get(path) or {}
            if kind in ("int", "float"):
                lo2, hi2, st2 = float(over.get("min", lo)), float(over.get("max", hi)), float(over.get("step", step))
                if lo2 < lo or hi2 > hi or st2 <= 0:
                    raise SpaceError(f"{path}: bounds must stay within whitelist [{lo}, {hi}] and step > 0")
                lo, hi, step = lo2, hi2, st2
            self.specs[path] = ParameterSpec(path, kind, lo, hi, step, choices)
        self.config = config
        self.baseline: Dict[str, Any] = {p: get_path(config, p) for p in self.specs}
        self.min_rr = float(get_path(config, "risk.min_risk_reward"))   # read-only invariant input

    # ------------------------------------------------------------ checks
    def change_pct(self, path: str, value: Any) -> Optional[float]:
        spec = self.specs[path]
        if spec.kind not in ("int", "float"):
            return None
        base = float(self.baseline[path])
        if base == 0:
            return None if float(value) == 0 else float("inf")
        return abs(float(value) - base) / abs(base) * 100.0

    def validate_overlay(self, overlay: Dict[str, Any]) -> List[str]:
        """Return a list of violation strings (empty = admissible)."""
        problems: List[str] = []
        changed = {k: v for k, v in overlay.items() if k not in self.baseline or v != self.baseline[k]}
        for path, v in changed.items():
            if path not in self.specs:
                problems.append(f"{path}: not tunable")
                continue
            spec = self.specs[path]
            if not spec.in_bounds(v):
                problems.append(f"{path}: {v} outside [{spec.min}, {spec.max}] / choices {spec.choices}")
            pct = self.change_pct(path, v)
            if pct is not None and pct > self.max_change_pct + 1e-9:
                problems.append(f"{path}: change {pct:.1f}% exceeds max_parameter_change_pct {self.max_change_pct}")
        if len(changed) > self.max_params_changed:
            problems.append(f"{len(changed)} parameters changed > max_params_changed_per_proposal {self.max_params_changed}")
        merged = {**self.baseline, **overlay}
        problems += self.invariants(merged)
        return problems

    def invariants(self, values: Dict[str, Any]) -> List[str]:
        out = []
        lo, hi = values.get("strategy.stops.min_stop_atr"), values.get("strategy.stops.max_stop_atr")
        if lo is not None and hi is not None and float(lo) > float(hi):
            out.append(f"invariant min_stop_atr ({lo}) <= max_stop_atr ({hi}) violated")
        rr = values.get("strategy.targets.fixed_rr")
        if rr is not None and float(rr) < self.min_rr:
            out.append(f"invariant fixed_rr ({rr}) >= risk.min_risk_reward ({self.min_rr}) violated")
        return out

    def admissible_values(self, path: str) -> List[Any]:
        """Candidate values for one parameter (baseline excluded), ascending, deterministic.

        Only declared grid values are ever returned: a grid point must lie within bounds, within
        `max_parameter_change_pct` of the current value and satisfy the invariants. A grid neighbour
        that exceeds the change cap is skipped - values are never interpolated off the grid - so a
        parameter can legitimately have zero legal candidates (reported via `skipped_parameters`).
        """
        spec = self.specs[path]
        base = self.baseline[path]
        return [v for v in spec.grid() if v != base and not self.validate_overlay({path: v})]

    def skipped_parameters(self) -> Dict[str, str]:
        """Whitelisted parameters with no legal candidate and the reason (for transparent reporting)."""
        out = {}
        for path, spec in self.specs.items():
            if self.admissible_values(path):
                continue
            base = self.baseline[path]
            others = [v for v in spec.grid() if v != base]
            if not others:
                out[path] = "grid has no value other than the current one"
                continue
            nearest = min(others, key=lambda v: abs(float(v) - float(base))) if spec.kind in ("int", "float") else others[0]
            problems = self.validate_overlay({path: nearest})
            out[path] = f"nearest grid value {nearest!r}: " + ("; ".join(problems) if problems else "not admissible")
        return out

    def neighbours(self, path: str, value: Any) -> List[Any]:
        """Adjacent admissible values (used by the neighbourhood-stability constraint): the candidate
        values / baseline immediately below and above `value` on the parameter's probed axis."""
        spec = self.specs[path]
        if spec.kind not in ("int", "float"):
            return []
        axis = sorted(set(self.admissible_values(path) + [self.baseline[path]]))
        if value not in axis:
            return []
        i = axis.index(value)
        return [axis[j] for j in (i - 1, i + 1) if 0 <= j < len(axis)]

    def describe(self) -> List[Dict[str, Any]]:
        return [{"path": s.path, "kind": s.kind, "min": s.min, "max": s.max, "step": s.step,
                 "choices": list(s.choices) if s.choices else None, "baseline": self.baseline[s.path]}
                for s in self.specs.values()]
