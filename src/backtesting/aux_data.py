"""Point-in-time auxiliary data feeds (e.g. USDT.D 4h, BTC.D, TOTAL3, DXY) for the primary bar series.

Alignment rule (the only rule): for primary bar i (a fully CLOSED candle with open
timestamp T_i and close time T_i + primary_tf), an auxiliary candle is visible iff its
own close time (open + aux_tf) is <= the primary bar's close time. A 4h aux candle
opening at 12:00 is therefore first visible to primary bars closing at/after 16:00.

Building blocks
---------------
AuxFeed      : a named auxiliary series (timestamp, close) plus an optional consumer factory.
AuxReplayer  : stateful cursor over one AuxFeed. `advance(primary_close_time)` feeds every
               newly closed aux row to the consumer and returns the consumer's state
               (None until the first aux row has closed). Used identically by the
               BacktestEngine and the PaperTrader, which is what keeps them consistent.
Consumer     : any object with `update(timestamp, close)` and a `.state` attribute
               (USDTDRegimeDetector is one). Feeds without a consumer expose the latest
               closed row as an `AuxPoint`.
build_aux_feeds(config, ...) : the `usdtd:` block defines the regime feed named
               REGIME_FEED ("usdtd"); `auxiliary.feeds` may add further named feeds that
               are exposed on `ctx.aux[name]` but consumed by nothing yet.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from src.strategy.regime import RegimeConfig, USDTDRegimeDetector

_TF = re.compile(r"^(\d+)([mhdw])$")
_UNIT = {"m": "min", "h": "h", "d": "D", "w": "W"}

REGIME_FEED = "usdtd"   # the only feed with an active consumer (USDTDRegimeDetector -> ctx.regime)


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    m = _TF.match(tf.strip().lower())
    if not m:
        raise ValueError(f"unsupported timeframe {tf!r} (expected e.g. 15m, 4h, 1d)")
    return pd.Timedelta(int(m.group(1)), unit=_UNIT[m.group(2)])


def _utc(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, utc=True)
    return out


def align_aux_indices(primary_ts: pd.Series, primary_tf: str, aux_ts: pd.Series, aux_tf: str) -> pd.Series:
    """Return, for each primary bar, the index into `aux_ts` of the latest fully closed aux candle (or -1)."""
    if len(primary_ts) == 0:
        return pd.Series([], dtype=int)
    p_close = _utc(primary_ts) + timeframe_to_timedelta(primary_tf)
    if len(aux_ts) == 0:
        return pd.Series([-1] * len(primary_ts), index=primary_ts.index, dtype=int)
    a_close = _utc(aux_ts) + timeframe_to_timedelta(aux_tf)
    if not a_close.is_monotonic_increasing:
        raise ValueError("auxiliary timestamps must be sorted ascending")
    # searchsorted(side='right') - 1 == number of aux closes <= primary close, minus one
    pos = a_close.searchsorted(p_close.to_numpy(), side="right") - 1
    return pd.Series(pos, index=primary_ts.index, dtype=int)


def aux_filename(symbol: str, timeframe: str) -> str:
    """USDT.D 4h -> USDTD_4h.csv (same convention as CSVMarketData)."""
    return f"{symbol.replace('/', '').replace('.', '')}_{timeframe}.csv"


def coerce_aux_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a full OHLCV frame or a (timestamp, close) frame; return sorted (timestamp, close)."""
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" not in cols or "close" not in cols:
        raise ValueError("auxiliary frame needs 'timestamp' and 'close' columns")
    out = pd.DataFrame({"timestamp": _utc(df[cols["timestamp"]]), "close": df[cols["close"]].astype(float)})
    return out.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------------- feeds
@dataclass(frozen=True)
class AuxPoint:
    """State exposed for a feed that has no dedicated consumer: the latest CLOSED aux row."""
    name: str
    timestamp: Any
    close: float
    index: int


@dataclass
class AuxFeed:
    name: str
    symbol: str
    timeframe: str
    frame: pd.DataFrame                                   # coerced (timestamp, close), sorted
    consumer_factory: Optional[Callable[[], Any]] = None  # zero-arg -> object with update(ts, close) & .state

    def __post_init__(self):
        self.frame = coerce_aux_frame(self.frame)
        timeframe_to_timedelta(self.timeframe)  # validate early

    def replayer(self) -> "AuxReplayer":
        return AuxReplayer(self)


class AuxReplayer:
    """Incremental point-in-time cursor over one AuxFeed (never looks beyond the primary bar's close)."""

    def __init__(self, feed: AuxFeed):
        self.feed = feed
        self.name = feed.name
        self._close_times = (feed.frame["timestamp"] + timeframe_to_timedelta(feed.timeframe)).to_numpy()
        self._ts = feed.frame["timestamp"].to_numpy()
        self._closes = feed.frame["close"].to_numpy()
        self.consumer = feed.consumer_factory() if feed.consumer_factory is not None else None
        self.fed_index = -1          # index of the last aux row fed
        self.state: Any = None       # None until the first aux row has closed

    def advance(self, primary_close_time: pd.Timestamp) -> Any:
        """Feed every aux row whose close time <= `primary_close_time`; return the current state."""
        n = len(self._close_times)
        limit = pd.Timestamp(primary_close_time)
        if limit.tzinfo is None:
            limit = limit.tz_localize("UTC")
        while self.fed_index + 1 < n and self._close_times[self.fed_index + 1] <= limit:
            self.fed_index += 1
            ts, close = pd.Timestamp(self._ts[self.fed_index]), float(self._closes[self.fed_index])
            if self.consumer is not None:
                self.consumer.update(ts, close)
                self.state = self.consumer.state
            else:
                self.state = AuxPoint(self.name, ts, close, self.fed_index)
        return self.state


@dataclass(frozen=True)
class AuxFeedSpec:
    name: str
    symbol: str
    timeframe: str
    enabled: bool = True
    consumer: Optional[str] = None   # "usdtd_regime" | None


def aux_specs_from_config(config: Dict[str, Any]) -> List[AuxFeedSpec]:
    """`usdtd:` block -> the REGIME_FEED spec; `auxiliary.feeds` -> extra named feeds (no consumer yet)."""
    specs: List[AuxFeedSpec] = []
    rc = RegimeConfig.from_config(config)
    specs.append(AuxFeedSpec(REGIME_FEED, rc.symbol, rc.timeframe, rc.enabled, "usdtd_regime"))
    extra = (config.get("auxiliary", {}) or {}).get("feeds", {}) or {}
    for name, spec in extra.items():
        if name == REGIME_FEED:
            raise ValueError(f"auxiliary.feeds.{REGIME_FEED} is reserved; configure it in the 'usdtd' block")
        spec = spec or {}
        specs.append(AuxFeedSpec(str(name), str(spec["symbol"]), str(spec["timeframe"]),
                                 bool(spec.get("enabled", True)), spec.get("consumer")))
    return specs


def build_aux_feeds(config: Dict[str, Any], data_root: Optional[Union[str, Path]] = None,
                    frames: Optional[Dict[str, pd.DataFrame]] = None) -> List[AuxFeed]:
    """Instantiate every enabled feed. `frames` overrides CSV loading per feed name (tests)."""
    frames = frames or {}
    feeds: List[AuxFeed] = []
    root = Path(data_root) if data_root is not None else Path.cwd()
    for spec in aux_specs_from_config(config):
        if not spec.enabled:
            continue
        df = frames.get(spec.name)
        if df is None:
            path = root / config["data"]["directory"] / aux_filename(spec.symbol, spec.timeframe)
            if not path.exists():
                raise FileNotFoundError(f"auxiliary feed '{spec.name}' is enabled but {path} does not exist")
            df = pd.read_csv(path)
        factory = None
        if spec.consumer == "usdtd_regime":
            rc = RegimeConfig.from_config(config)
            factory = (lambda rc=rc: USDTDRegimeDetector(rc))
        elif spec.consumer is not None:
            raise ValueError(f"unknown auxiliary consumer {spec.consumer!r} for feed '{spec.name}'")
        feeds.append(AuxFeed(spec.name, spec.symbol, spec.timeframe, df, factory))
    return feeds
