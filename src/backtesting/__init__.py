"""Backtesting: point-in-time engine, trade journal, metrics. No live trading."""

from src.backtesting.aux_data import REGIME_FEED, AuxFeed, AuxPoint, AuxReplayer, build_aux_feeds
from src.backtesting.demo_strategy import FixedIntervalTestStrategy
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.journal import EXIT_END, EXIT_SIGNAL, EXIT_STOP, EXIT_TARGET, RejectedProposal, TradeJournal, TradeRecord
from src.backtesting.metrics import compute_metrics, max_drawdown
from src.backtesting.result import BacktestResult
from src.backtesting.strategy import BUY, EXIT, BacktestContext, Signal, Strategy

__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult", "BacktestContext", "Signal", "Strategy",
           "FixedIntervalTestStrategy", "TradeJournal", "TradeRecord", "RejectedProposal", "compute_metrics",
           "max_drawdown", "BUY", "EXIT", "EXIT_STOP", "EXIT_TARGET", "EXIT_SIGNAL", "EXIT_END",
           "AuxFeed", "AuxPoint", "AuxReplayer", "build_aux_feeds", "REGIME_FEED"]
