"""BacktestResult: bundles journal, equity curve, metrics; save / print helpers."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from src.backtesting.journal import TradeJournal


@dataclass
class BacktestResult:
    run_id: str
    symbol: str
    timeframe: str
    journal: TradeJournal
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        def clean(v):
            if isinstance(v, float) and math.isinf(v):
                return "inf"
            return v
        return {
            "run_id": self.run_id, "symbol": self.symbol, "timeframe": self.timeframe,
            "metrics": {k: clean(v) for k, v in self.metrics.items()},
            "config": self.config,
        }

    def save(self, directory: Union[str, Path]) -> Path:
        out = Path(directory) / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        self.journal.to_frame().to_csv(out / "trades.csv", index=False)
        self.journal.rejections_frame().to_csv(out / "rejections.csv", index=False)
        self.equity_curve.to_csv(out / "equity_curve.csv", index=False)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, indent=2, default=str)
        return out

    def format_summary(self) -> str:
        m = self.metrics
        pf = "inf" if math.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}"
        lines = [
            f"Backtest {self.run_id}  {self.symbol} {self.timeframe}  ({m['bars']} bars)",
            f"  starting equity : {m['starting_equity']:.2f}",
            f"  ending equity   : {m['ending_equity']:.2f}",
            f"  total return    : {m['total_return_pct']:+.2f}%",
            f"  trades          : {m['trades']}  (W {m['winning_trades']} / L {m['losing_trades']} / BE {m['breakeven_trades']})",
            f"  win rate        : {m['win_rate_pct']:.1f}%",
            f"  gross profit    : {m['gross_profit']:.2f}",
            f"  gross loss      : {m['gross_loss']:.2f}",
            f"  profit factor   : {pf}",
            f"  average win     : {m['average_win']:.2f}",
            f"  average loss    : {m['average_loss']:.2f}",
            f"  expectancy      : {m['expectancy']:.2f}  (avg R {m['average_r_multiple']:+.2f})",
            f"  fees            : {m['total_fees']:.2f}",
            f"  max drawdown    : {m['max_drawdown_pct']:.2f}%  ({m['max_drawdown_bars']} bars)",
            f"  risk per trade  : {m['risk_per_trade_pct']:.2f}% configured, {m['avg_realized_risk_pct']:.2f}% avg realized",
            f"  exit reasons    : {m['exit_reasons']}",
            f"  risk rejections : {m['risk_rejections']}",
        ]
        return "\n".join(lines)
