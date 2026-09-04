"""Public-endpoint OHLCV source built on CCXT. THE ONLY MODULE IN THE PROJECT THAT TOUCHES THE NETWORK.

- `ccxt` is imported lazily here, so every other path (backtest, paper, improve, tests) runs without it.
- Only `load_markets()` and `fetch_ohlcv()` are ever called - both public market-data endpoints.
- Credentials are refused: any apiKey / secret / password / uid / token in the options raises, and the
  constructed client is checked to carry none. No environment variables are read.
- Any CCXT exchange id works (binance default; bybit, okx, ... later) - the Downloader never sees which.
"""

import time
from typing import Any, Dict, List, Optional

from src.data.fetch.source import OHLCVSource, Row, SourceError

_CREDENTIAL_KEYS = ("apiKey", "secret", "password", "uid", "token", "privateKey", "walletAddress", "login")
_DEFAULT_MAX_PAGE = {"binance": 1000, "bybit": 1000, "okx": 100, "kraken": 720, "coinbase": 300}


def _assert_no_credentials(options: Dict[str, Any]) -> None:
    bad = [k for k in options if k in _CREDENTIAL_KEYS or k.lower() in ("apikey", "api_key", "secret")]
    if bad:
        raise SourceError(f"credentials are not allowed for the public data source (got {bad})")


class CCXTPublicSource(OHLCVSource):
    online = True
    kind = "ohlcv"

    def __init__(self, exchange_id: str = "binance", options: Optional[Dict[str, Any]] = None,
                 fetch_cfg: Optional[Dict[str, Any]] = None, client: Any = None):
        options = dict(options or {})
        _assert_no_credentials(options)
        fetch_cfg = fetch_cfg or {}
        self.id = exchange_id
        self.max_retries = int(fetch_cfg.get("max_retries", 5))
        self.backoff = float(fetch_cfg.get("backoff_seconds", 2))
        self.max_page = int(fetch_cfg.get("page_limit") or _DEFAULT_MAX_PAGE.get(exchange_id, 500))
        self._sleep = time.sleep
        if client is None:                       # `client` injection is for tests (stub exchange)
            try:
                import ccxt  # noqa: WPS433 - lazy on purpose
            except ImportError as exc:
                raise SourceError("ccxt is not installed; `pip install ccxt` is only needed for `data download/update`") from exc
            if not hasattr(ccxt, exchange_id):
                raise SourceError(f"unknown ccxt exchange id {exchange_id!r}")
            options.setdefault("enableRateLimit", bool(fetch_cfg.get("rate_limit", True)))
            client = getattr(ccxt, exchange_id)(options)
            self.ccxt_version = getattr(ccxt, "__version__", "?")
        else:
            self.ccxt_version = getattr(client, "version", "stub")
        for key in _CREDENTIAL_KEYS:
            if getattr(client, key, None):
                raise SourceError(f"exchange client carries a credential ({key}); refusing to use it")
        has = getattr(client, "has", {}) or {}
        if has and not has.get("fetchOHLCV", True):
            raise SourceError(f"{exchange_id} does not support fetchOHLCV")
        self.client = client
        self._markets_loaded = False

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"ccxt_version": self.ccxt_version, "endpoints": ["load_markets", "fetch_ohlcv"], "credentials": False})
        return d

    def _ensure_markets(self, symbol: str) -> None:
        if self._markets_loaded:
            return
        markets = self._retry(lambda: self.client.load_markets())
        if markets and symbol not in markets:
            raise SourceError(f"{self.id} does not list market {symbol!r}")
        self._markets_loaded = True

    def _retry(self, fn):
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - ccxt raises many network/ratelimit types
                last = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(self.backoff * (2 ** attempt))
        raise SourceError(f"{self.id}: giving up after {self.max_retries} attempts: {last}") from last

    def fetch(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> List[Row]:
        self._ensure_markets(symbol)
        limit = min(int(limit), self.max_page)
        rows = self._retry(lambda: self.client.fetch_ohlcv(symbol, timeframe, since=int(since_ms), limit=limit))
        return [[int(r[0])] + [float(x) for x in r[1:6]] for r in rows or []]
