"""SMC long-only spot strategy (Step 7).

Implements the backtester's `Strategy.on_candle(ctx) -> Optional[Signal]`.
Everything is derived from `ctx` at bar i: the SMC result (events with
detected_index <= i), the indicator row for bar i, the candle i, and the
optional USDT.D RegimeState (already aligned point-in-time by the engine).

State machine
-------------
IDLE      -(new bullish BOS with POIs)->  ARMED
ARMED     -(POIs all mitigated / setup too old / bearish CHoCH)-> IDLE
ARMED     -(touch POI + rejection + filters + regime + geometry)-> emit BUY -> IN_TRADE
IN_TRADE  -(ctx.has_position False: engine filled stop/target or rejected the entry)-> COOLDOWN / ARMED
IN_TRADE  -(bearish CHoCH / time stop)-> emit EXIT -> COOLDOWN
COOLDOWN  -(reentry_cooldown_bars elapsed)-> IDLE

Entry gates (all must pass; see README for the full spec):
  A structure bullish, trigger BOS young enough
  B a valid unmitigated POI (OB from the trigger BOS, or FVG from its impulse)
  C candle touched the POI and did not close below it
  D rejection: D1 bullish close in upper half | D2 bullish sweep this bar | D3 close back above zone
  E filters: EMA trend, EMA extension, volume (each switchable)
  F geometry: stop distance within [min_stop_atr, max_stop_atr] * ATR, RR >= min_risk_reward
  G USDT.D regime (RISING only tightens D, E-volume, F-RR, and requires a deeper POI)
  H cooldown / max entries per setup

Stop  : min(zone.low, sweep wick) - buffer_atr * ATR
Target: nearest unbroken swing high above entry giving RR >= min (mode structure), or fixed RR, or hybrid.

USDT.D never touches RiskState/TradeValidator. Signals are still validated by the
Risk Engine and filled by the backtester at the NEXT open; if the next open is at or
below the stop, the validator rejects the proposal (REJECTED_INVALID_STOP) - no
second risk system exists here.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtesting.strategy import BUY, EXIT, BacktestContext, Signal
from src.strategy.poi import (
    FVG,
    ORDER_BLOCK,
    POI,
    Setup,
    build_setup,
    confluence_score,
    deeper_pois,
    mitigate,
    select_poi,
)
from src.strategy.regime import Regime, RegimeState
from src.strategy.smc_types import BEARISH, BULLISH, BOSEvent

IDLE, ARMED, IN_TRADE, COOLDOWN = "IDLE", "ARMED", "IN_TRADE", "COOLDOWN"
TP_STRUCTURE, TP_FIXED, TP_HYBRID = "structure", "fixed_rr", "hybrid"
REJ_D1, REJ_D2, REJ_D3 = "D1_bullish_close", "D2_sweep", "D3_close_above_zone"


@dataclass(frozen=True)
class SMCStrategyConfig:
    # entry
    setup_max_age_bars: int = 60
    choch_age_multiplier: float = 1.5
    poi_priority: tuple = (ORDER_BLOCK, FVG)
    poi_max_atr_multiple: float = 3.0
    require_rejection: bool = True
    rejection_close_position_min: float = 0.5
    max_entries_per_setup: int = 1
    reentry_cooldown_bars: int = 3
    # filters
    ema_trend_enabled: bool = True
    ema_extension_enabled: bool = True
    ema_extension_atr: float = 1.0
    volume_enabled: bool = True
    vol_ratio_min: float = 0.8
    bos_vol_ratio_min: float = 1.2
    # stops / targets
    stop_buffer_atr: float = 0.25
    min_stop_atr: float = 0.5
    max_stop_atr: float = 3.0
    tp_mode: str = TP_STRUCTURE
    fixed_rr: float = 2.0
    min_risk_reward: float = 2.0
    # exits
    exit_on_bearish_choch: bool = True
    max_bars_in_trade: int = 96
    # usdtd risk-off tightening (only applied when a RISING regime is supplied)
    usdtd_enabled: bool = False
    riskoff_min_confluence: int = 2
    riskoff_rr_add: float = 0.5
    riskoff_vol_ratio_min: float = 1.0
    riskoff_require_deeper_poi: bool = True
    # indicator column names
    atr_col: str = "atr_14"
    ema_fast_col: str = "ema_20"
    ema_mid_col: str = "ema_50"
    ema_slow_col: str = "ema_200"
    vol_ratio_col: str = "volume_ratio_20"

    def __post_init__(self):
        if self.tp_mode not in (TP_STRUCTURE, TP_FIXED, TP_HYBRID):
            raise ValueError("tp_mode must be structure | fixed_rr | hybrid")
        if self.min_stop_atr <= 0 or self.max_stop_atr < self.min_stop_atr:
            raise ValueError("require 0 < min_stop_atr <= max_stop_atr")
        if self.min_risk_reward <= 0 or self.fixed_rr <= 0:
            raise ValueError("risk/reward values must be > 0")
        if self.max_entries_per_setup < 1 or self.reentry_cooldown_bars < 0:
            raise ValueError("bad entry limits")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SMCStrategyConfig":
        s = config.get("strategy", {})
        e, f, st, tg, ex = (s.get(k, {}) or {} for k in ("entry", "filters", "stops", "targets", "exits"))
        u = config.get("usdtd", {}) or {}
        ro = u.get("risk_off", {}) or {}
        ind = config["indicators"]
        ema_p = sorted(set(int(p) for p in ind["ema"].get("periods", [20, 50, 200])))
        return cls(
            setup_max_age_bars=int(e.get("setup_max_age_bars", 60)),
            choch_age_multiplier=float(e.get("choch_age_multiplier", 1.5)),
            poi_priority=tuple(e.get("poi_priority", [ORDER_BLOCK, FVG])),
            poi_max_atr_multiple=float(e.get("poi_max_atr_multiple", 3.0)),
            require_rejection=bool(e.get("require_rejection", True)),
            rejection_close_position_min=float(e.get("rejection_close_position_min", 0.5)),
            max_entries_per_setup=int(e.get("max_entries_per_setup", 1)),
            reentry_cooldown_bars=int(e.get("reentry_cooldown_bars", 3)),
            ema_trend_enabled=bool((f.get("ema_trend") or {}).get("enabled", True)),
            ema_extension_enabled=bool((f.get("ema_extension") or {}).get("enabled", True)),
            ema_extension_atr=float((f.get("ema_extension") or {}).get("atr_multiple", 1.0)),
            volume_enabled=bool((f.get("volume") or {}).get("enabled", True)),
            vol_ratio_min=float((f.get("volume") or {}).get("ratio_min", 0.8)),
            bos_vol_ratio_min=float((f.get("volume") or {}).get("bos_ratio_min", 1.2)),
            stop_buffer_atr=float(st.get("buffer_atr", 0.25)),
            min_stop_atr=float(st.get("min_stop_atr", 0.5)),
            max_stop_atr=float(st.get("max_stop_atr", 3.0)),
            tp_mode=str(tg.get("mode", TP_STRUCTURE)),
            fixed_rr=float(tg.get("fixed_rr", 2.0)),
            min_risk_reward=float(config.get("risk", {}).get("min_risk_reward", 2.0)),
            exit_on_bearish_choch=bool(ex.get("exit_on_bearish_choch", True)),
            max_bars_in_trade=int(ex.get("max_bars_in_trade", 96)),
            usdtd_enabled=bool(u.get("enabled", False)),
            riskoff_min_confluence=int(ro.get("min_confluence_score", 2)),
            riskoff_rr_add=float(ro.get("rr_add", 0.5)),
            riskoff_vol_ratio_min=float(ro.get("vol_ratio_min", 1.0)),
            riskoff_require_deeper_poi=bool(ro.get("require_deeper_poi", True)),
            atr_col=f"atr_{ind['atr']['period']}",
            ema_fast_col=f"ema_{ema_p[0]}",
            ema_mid_col=f"ema_{ind['ema'].get('fast', ema_p[min(1, len(ema_p) - 1)])}",
            ema_slow_col=f"ema_{ind['ema'].get('slow', ema_p[-1])}",
            vol_ratio_col=f"volume_ratio_{ind['volume']['ma_period']}",
        )


@dataclass
class _Trade:
    entry_index: int
    entry_ref: float
    stop: float
    target: float
    exit_requested: bool = False


@dataclass
class StrategyDiagnostics:
    """Counters for tests / reporting. Not used for decisions."""
    setups_armed: int = 0
    buy_signals: int = 0
    exit_signals: int = 0
    gate_failures: Dict[str, int] = field(default_factory=dict)
    riskoff_skips: int = 0

    def fail(self, gate: str) -> None:
        self.gate_failures[gate] = self.gate_failures.get(gate, 0) + 1


class SMCStrategy:
    def __init__(self, config: Optional[SMCStrategyConfig] = None):
        self.cfg = config or SMCStrategyConfig()
        self.state = IDLE
        self.setup: Optional[Setup] = None
        self.trade: Optional[_Trade] = None
        self.last_exit_index = -10**9
        self._seen_bos = 0
        self._pending_entry: Optional[_Trade] = None
        self._bos_vol: Dict[int, float] = {}       # bos.detected_index -> volume ratio on that bar
        self._handled_bos: Optional[BOSEvent] = None   # last BOS that produced (or failed to produce) a setup
        self.diag = StrategyDiagnostics()
        self.last_poi: Optional[POI] = None            # POI touched on the last on_candle call (reporting only)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SMCStrategy":
        return cls(SMCStrategyConfig.from_config(config))

    # ------------------------------------------------------------------ main
    def on_candle(self, ctx: BacktestContext) -> Optional[Signal]:
        cfg = self.cfg
        i = ctx.index
        smc = ctx.smc
        c = ctx.candle
        self.last_poi = None

        # --- reconcile with the engine's actual position state
        if self.state == IN_TRADE and not ctx.has_position:
            # position closed by stop/target/exit-fill, or our BUY was rejected / never filled
            if self._pending_entry is not None and self.trade is None:
                self._pending_entry = None            # entry never happened -> back to ARMED if setup alive
                self.state = ARMED if self.setup else IDLE
            else:
                self.trade = None
                self.last_exit_index = i
                self.state = COOLDOWN
        if self.state == IN_TRADE and ctx.has_position and self.trade is None and self._pending_entry is not None:
            self.trade, self._pending_entry = self._pending_entry, None
            self.trade.entry_index = i               # filled at this bar's open

        # --- new BOS events (incremental)
        new_bos = smc.bos_events[self._seen_bos:]
        self._seen_bos = len(smc.bos_events)
        for b in new_bos:
            if b.detected_index == i:
                vr = _num(ctx.indicators.get(cfg.vol_ratio_col))
                if vr is not None:
                    self._bos_vol[b.detected_index] = vr

        bearish_choch_now = any(b.direction == BEARISH and b.is_choch and b.detected_index == i for b in new_bos)

        # --- in trade: structural / time exits
        if self.state == IN_TRADE:
            if self.trade is not None and not self.trade.exit_requested:
                if cfg.exit_on_bearish_choch and bearish_choch_now:
                    return self._exit("bearish_choch")
                bars = i - self.trade.entry_index
                risk = self.trade.entry_ref - self.trade.stop
                if bars >= cfg.max_bars_in_trade and c.close - self.trade.entry_ref < risk:
                    return self._exit(f"time_stop_{bars}b")
            return None

        if self.state == COOLDOWN:
            if i - self.last_exit_index < cfg.reentry_cooldown_bars:
                return None
            self.state = IDLE

        # --- maintain / (re)arm setup
        atr = _num(ctx.indicators.get(cfg.atr_col))
        latest_bull = _latest_bullish_bos(smc)
        if latest_bull is not None and latest_bull is not self._handled_bos:
            self._handled_bos = latest_bull
            self.setup = build_setup(smc, latest_bull, atr or 0.0, cfg.poi_max_atr_multiple,
                                     list(cfg.poi_priority), self._bos_vol.get(latest_bull.detected_index))
            if self.setup.pois:
                self.diag.setups_armed += 1
                self.state = ARMED
            else:
                self.setup, self.state = None, IDLE
        if self.setup is None:
            return None

        mitigate(self.setup, c.close)
        if bearish_choch_now or smc.structure != BULLISH:
            self._drop_setup()
            return None
        max_age = cfg.setup_max_age_bars * (cfg.choch_age_multiplier if self.setup.trigger_bos.is_choch else 1.0)
        if i - self.setup.trigger_index > max_age:
            self._drop_setup()
            self.diag.fail("A_setup_age")
            return None
        if not self.setup.valid_pois():
            self._drop_setup()
            return None
        self.state = ARMED
        if self.setup.entries >= cfg.max_entries_per_setup:
            return None
        if ctx.has_position:
            return None
        if i <= self.setup.trigger_index:
            return None   # the break candle itself is not a pullback

        # --- gate C: candle touched a POI
        poi = select_poi(self.setup, c.low, c.close, list(cfg.poi_priority))
        if poi is None:
            return None
        self.last_poi = poi

        # --- gate G(part): risk-off "wait for the dip"
        regime = _regime_of(ctx, cfg)
        risk_off = regime is Regime.RISING
        if risk_off and cfg.riskoff_require_deeper_poi:
            # "wait for the dip": the setup's FIRST (highest) POI is never bought in a RISING regime.
            # Any currently valid POI strictly below it - of any kind - qualifies.
            first = max(self.setup.pois, key=lambda p: p.high)
            if poi is first:
                if not poi.skipped:
                    poi.skipped = True
                    self.diag.riskoff_skips += 1
                if not deeper_pois(self.setup, poi):
                    self.diag.fail("G_no_deeper_poi")
                    return None
                poi = select_poi(self.setup, c.low, c.close, list(cfg.poi_priority))  # deeper zone touched too?
                if poi is None:
                    return None
                self.last_poi = poi

        # --- indicators sanity (E4)
        if atr is None or atr <= 0:
            self.diag.fail("E4_atr")
            return None

        # --- gate D: rejection
        sweep = next((s for s in smc.liquidity_sweeps if s.direction == BULLISH and s.detected_index == i), None)
        rng = c.high - c.low
        close_pos = (c.close - c.low) / rng if rng > 0 else 1.0
        d1 = c.close > c.open and close_pos >= cfg.rejection_close_position_min
        d2 = sweep is not None
        d3 = c.close > poi.high
        if cfg.require_rejection:
            if risk_off:
                if not (d2 or d3):
                    self.diag.fail("D_riskoff_rejection")
                    return None
            elif not (d1 or d2 or d3):
                self.diag.fail("D_rejection")
                return None
        rejection = REJ_D2 if d2 else (REJ_D3 if d3 else REJ_D1)

        # --- gate E: filters
        ema_f, ema_m, ema_s = (_num(ctx.indicators.get(k)) for k in (cfg.ema_fast_col, cfg.ema_mid_col, cfg.ema_slow_col))
        if cfg.ema_trend_enabled:
            if ema_m is None or ema_s is None or not (c.close > ema_s and ema_m > ema_s):
                self.diag.fail("E1_ema_trend")
                return None
        if cfg.ema_extension_enabled:
            if ema_f is None or c.close > ema_f + cfg.ema_extension_atr * atr:
                self.diag.fail("E2_extension")
                return None
        score = confluence_score(poi, self.setup, d2, cfg.bos_vol_ratio_min)
        if cfg.volume_enabled:
            vr = _num(ctx.indicators.get(cfg.vol_ratio_col))
            vmin = cfg.riskoff_vol_ratio_min if risk_off else cfg.vol_ratio_min
            bos_ok = self.setup.bos_volume_ratio is not None and self.setup.bos_volume_ratio >= cfg.bos_vol_ratio_min
            if not ((vr is not None and vr >= vmin) or bos_ok):
                self.diag.fail("E3_volume")
                return None
        if risk_off and score < cfg.riskoff_min_confluence:
            self.diag.fail("G_confluence")
            return None

        # --- stop (section 3) and gate F geometry
        entry_ref = c.close
        base = min(poi.low, sweep.wick_extreme) if sweep is not None else poi.low
        stop = base - cfg.stop_buffer_atr * atr
        dist = entry_ref - stop
        if stop <= 0 or dist < cfg.min_stop_atr * atr:
            self.diag.fail("F_stop_too_tight")
            return None
        if dist > cfg.max_stop_atr * atr:
            self.diag.fail("F_stop_too_wide")
            return None
        min_rr = cfg.min_risk_reward + (cfg.riskoff_rr_add if risk_off else 0.0)
        target = self._target(smc, entry_ref, dist, min_rr)
        if target is None or (target - entry_ref) / dist < min_rr:
            self.diag.fail("F_rr")
            return None

        # --- emit
        self.setup.entries += 1
        poi.entries += 1
        self._pending_entry = _Trade(i + 1, entry_ref, stop, target)
        self.state = IN_TRADE
        self.diag.buy_signals += 1
        reason = (f"smc:{poi.kind} rej={rejection} score={score} rr={(target - entry_ref) / dist:.2f} "
                  f"regime={regime.value if regime else 'NONE'}")
        return Signal(BUY, stop_loss=stop, take_profit=target, reason=reason)

    # ----------------------------------------------------------- persistence
    def to_snapshot(self) -> Dict[str, Any]:
        """JSON-serialisable state for restart recovery (Step 8). Bar indices are stored as-is;
        `restore_snapshot` shifts them by `index_shift` when the SMC engine was rebuilt from a
        shorter history. BOS events are referenced by key and relinked to the rebuilt SMC result."""
        def trade(t: Optional[_Trade]):
            return None if t is None else asdict(t)

        setup = None
        if self.setup is not None:
            setup = {
                "trigger_bos": bos_key(self.setup.trigger_bos),
                "trigger_index": self.setup.trigger_index,
                "impulse_start_index": self.setup.impulse_start_index,
                "bos_volume_ratio": self.setup.bos_volume_ratio,
                "entries": self.setup.entries,
                "pois": [{**asdict(p), "timestamp": _ts_str(p.timestamp)} for p in self.setup.pois],
            }
        return {
            "state": self.state,
            "setup": setup,
            "trade": trade(self.trade),
            "pending_entry": trade(self._pending_entry),
            "last_exit_index": self.last_exit_index,
            "seen_bos": self._seen_bos,
            "bos_vol": {str(k): v for k, v in self._bos_vol.items()},
            "handled_bos": bos_key(self._handled_bos) if self._handled_bos is not None else None,
            "diag": asdict(self.diag),
        }

    def restore_snapshot(self, snap: Dict[str, Any], smc, index_shift: int = 0) -> List[str]:
        """Restore from `to_snapshot()` against a rebuilt SMC result. Returns a list of warnings
        (e.g. a trigger BOS that no longer exists in the rebuilt history -> setup dropped)."""
        warnings: List[str] = []
        sh = int(index_shift)
        by_key = {bos_key(b): b for b in smc.bos_events}

        def trade(d):
            if d is None:
                return None
            t = _Trade(**d)
            t.entry_index += sh
            return t

        self.state = snap["state"]
        self.trade = trade(snap.get("trade"))
        self._pending_entry = trade(snap.get("pending_entry"))
        self.last_exit_index = int(snap["last_exit_index"]) + sh
        self._bos_vol = {int(k) + sh: float(v) for k, v in (snap.get("bos_vol") or {}).items()}
        self._seen_bos = len(smc.bos_events)   # everything currently in the rebuilt result is "seen"
        self.diag = StrategyDiagnostics(**snap.get("diag", {})) if snap.get("diag") else StrategyDiagnostics()

        hk = snap.get("handled_bos")
        self._handled_bos = by_key.get(hk) if hk else None
        if hk and self._handled_bos is None:
            warnings.append(f"handled BOS {hk} not found in rebuilt SMC history")

        self.setup = None
        sd = snap.get("setup")
        if sd is not None:
            bos = by_key.get(sd["trigger_bos"])
            if bos is None:
                warnings.append(f"setup trigger BOS {sd['trigger_bos']} not found in rebuilt SMC history; setup dropped")
                if self.state == ARMED:
                    self.state = IDLE
            else:
                pois = []
                for pd_ in sd["pois"]:
                    d = dict(pd_)
                    d["created_index"] = int(d["created_index"]) + sh
                    pois.append(POI(**d))
                self.setup = Setup(bos, int(sd["trigger_index"]) + sh, int(sd["impulse_start_index"]) + sh,
                                   pois, sd.get("bos_volume_ratio"), int(sd.get("entries", 0)))
        return warnings

    # --------------------------------------------------------------- helpers
    def _drop_setup(self) -> None:
        self.setup = None
        if self.state in (ARMED, IDLE):
            self.state = IDLE

    def _exit(self, why: str) -> Signal:
        self.diag.exit_signals += 1
        if self.trade is not None:
            self.trade.exit_requested = True   # engine fills at next open; do not re-emit meanwhile
        return Signal(EXIT, reason=why)

    def _target(self, smc, entry_ref: float, dist: float, min_rr: float) -> Optional[float]:
        cfg = self.cfg
        fixed = entry_ref + cfg.fixed_rr * dist
        if cfg.tp_mode == TP_FIXED:
            return fixed
        highs = sorted(s.price for s in smc.swing_highs if not s.broken and s.price > entry_ref)
        structural = next((h for h in highs if (h - entry_ref) / dist >= min_rr), None)
        if cfg.tp_mode == TP_STRUCTURE:
            return structural if structural is not None else fixed
        return max(structural, fixed) if structural is not None else fixed


def bos_key(b: BOSEvent) -> str:
    """Stable identity of a BOS event across engine rebuilds (independent of bar indices)."""
    return f"{b.direction}|{_ts_str(b.timestamp)}|{b.broken_swing_price!r}|{int(b.is_choch)}"


def _ts_str(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).isoformat()
    except (TypeError, ValueError):
        return str(ts)


def _latest_bullish_bos(smc) -> Optional[BOSEvent]:
    for b in reversed(smc.bos_events):
        if b.direction == BULLISH:
            return b
    return None


def _regime_of(ctx: BacktestContext, cfg: SMCStrategyConfig) -> Optional[Regime]:
    """UNKNOWN/None/disabled all behave as NEUTRAL. Only RISING changes anything."""
    if not cfg.usdtd_enabled:
        return None
    rs: Optional[RegimeState] = getattr(ctx, "regime", None)
    if rs is None or rs.regime is Regime.UNKNOWN:
        return Regime.NEUTRAL
    return rs.regime


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
