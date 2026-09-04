"""Paper-trading execution layer: simulated broker and the replay-driven paper-trading loop. No live trading."""

from src.execution.paper_broker import InsufficientFunds, InsufficientPosition, PaperBroker, Trade

__all__ = ["PaperBroker", "Trade", "InsufficientFunds", "InsufficientPosition", "PaperTrader", "PaperConfig",
           "CandleReport", "StateStore", "ConfigMismatch"]


def __getattr__(name):  # lazy: PaperTrader depends on src.backtesting, which imports this package
    if name in ("PaperTrader", "PaperConfig", "CandleReport", "ConfigMismatch"):
        from src.execution import paper_trader
        return getattr(paper_trader, name)
    if name == "StateStore":
        from src.execution.state_store import StateStore
        return StateStore
    raise AttributeError(name)
