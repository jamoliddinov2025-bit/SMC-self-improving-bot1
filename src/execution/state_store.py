"""Atomic JSON state file for the paper trader (no database).

`save()` writes to a temporary file in the same directory and `os.replace()`s it over
the target, so a crash mid-write can never leave a truncated state file behind.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

SCHEMA_VERSION = 1

# config sections that change trading behaviour; a different hash refuses a silent resume
_HASHED_SECTIONS = ("market", "strategy", "usdtd", "auxiliary", "indicators", "risk", "execution", "backtesting")


def config_hash(config: Dict[str, Any]) -> str:
    subset = {k: config.get(k) for k in _HASHED_SECTIONS}
    blob = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class StateStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported or corrupt state file {self.path}")
        return state

    def save(self, state: Dict[str, Any]) -> None:
        state = dict(state)
        state["schema_version"] = SCHEMA_VERSION
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()
