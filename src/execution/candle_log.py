"""Append-only CSV logs for the paper trader: one row per processed candle, plus trades and rejections.

Everything is plain CSV under the paper state directory. Files are opened in append mode
and the header is written only when the file is new/empty, so a restart continues the
same files. Rows are flushed immediately after each candle.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

CANDLE_COLUMNS: List[str] = [
    "bar_index", "timestamp", "status", "open", "high", "low", "close", "volume",
    "ema_fast", "ema_mid", "ema_slow", "atr", "volume_ratio",
    "regime", "regime_raw", "aux", "smc_structure", "smc_events", "strategy_state", "setup_trigger_index",
    "poi_kind", "poi_low", "poi_high", "gate_failures",
    "signal_side", "signal_stop", "signal_target", "signal_reason",
    "risk_decision", "risk_reason", "fill_side", "fill_price", "fill_qty", "fill_fee", "exit_reason",
    "position_qty", "entry_price", "stop_loss", "take_profit", "cash", "equity", "unrealized_pnl",
    "realized_pnl_total", "daily_pnl", "drawdown_pct", "consecutive_losses", "note",
]


class CsvAppender:
    def __init__(self, path: Union[str, Path], columns: Iterable[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns, extrasaction="ignore")
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: _fmt(row.get(k)) for k in self.columns})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, dict):
        return ";".join(f"{k}={val}" for k, val in v.items())
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return v
