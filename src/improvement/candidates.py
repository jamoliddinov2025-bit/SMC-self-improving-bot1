"""Deterministic candidate generation.

v1: SINGLE-PARAMETER coordinate descent - every admissible grid value of every tunable
parameter, one parameter at a time, in whitelist order then ascending value order.
Extension point: `Stage` objects; a future `PairwiseStage` can combine the top single
parameter winners (bounded by max_params_changed_per_proposal = 2) without changing the
runner. No randomness anywhere.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.improvement.space import ParameterSpace


def candidate_id(overlay: Dict[str, Any]) -> str:
    blob = json.dumps(overlay, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Candidate:
    overlay: Dict[str, Any]           # {config_path: value}; {} == baseline
    stage: str = "baseline"
    id: str = field(default="")

    def __post_init__(self):
        object.__setattr__(self, "id", candidate_id(self.overlay) if self.overlay else "baseline")

    @property
    def is_baseline(self) -> bool:
        return not self.overlay

    def label(self, space: Optional[ParameterSpace] = None) -> str:
        if self.is_baseline:
            return "baseline"
        return ", ".join(f"{k.split('.', 1)[1]}={v}" for k, v in sorted(self.overlay.items()))


class Stage:
    name = "abstract"

    def generate(self, space: ParameterSpace, previous: List[Candidate]) -> Iterable[Candidate]:  # pragma: no cover
        raise NotImplementedError


class SingleParameterStage(Stage):
    """Coordinate descent over one parameter at a time."""
    name = "single"

    def generate(self, space: ParameterSpace, previous: List[Candidate]) -> Iterable[Candidate]:
        for path in space.specs:                       # whitelist order (deterministic)
            for v in space.admissible_values(path):    # ascending grid order
                yield Candidate({path: v}, self.name)


class CandidateGenerator:
    def __init__(self, space: ParameterSpace, stages: Optional[List[Stage]] = None, max_candidates: Optional[int] = None):
        self.space = space
        self.stages = stages or [SingleParameterStage()]
        self.max_candidates = max_candidates

    def generate(self) -> List[Candidate]:
        out: List[Candidate] = [Candidate({})]
        seen = {"baseline"}
        for stage in self.stages:
            for c in stage.generate(self.space, out):
                if c.id in seen:
                    continue
                if self.space.validate_overlay(c.overlay):   # defence in depth: never emit an inadmissible overlay
                    continue
                seen.add(c.id)
                out.append(c)
                if self.max_candidates is not None and len(out) - 1 >= self.max_candidates:
                    return out
        return out
