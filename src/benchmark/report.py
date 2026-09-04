"""Human-readable Markdown report for a benchmark run (pure formatting; no computation)."""

from typing import Any, Dict


def _f(v, fmt="{:.2f}", na="unavailable"):
    if v is None:
        return na
    if v == "inf":
        return "inf"
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _header(m: Dict[str, Any]) -> list:
    d = m["dataset"]
    L = [f"# {m['label']} - {m['benchmark_id']}", ""]
    if m["synthetic"]:
        L += ["> **WARNING: the input dataset is labelled SYNTHETIC / fixture data. Nothing below describes real market "
              "behaviour.**", ""]
    L += [f"> {m['disclaimer']}", "",
          "## Identity (reproduce from these inputs)", "",
          f"- benchmark id: `{m['benchmark_id']}`  generated {m['generated_utc']}  repository commit `{m['repository_commit']}`",
          f"- dataset id: `{d['dataset_id']}`  dataset_sha256 `{d['dataset_sha256']}`  ({d['directory']})",
          f"- primary: {d['primary']['symbol']} {d['primary']['timeframe']}  {d['primary']['rows']:,} rows  "
          f"{d['primary']['first_open']} -> {d['primary']['last_open']}  sha256 `{d['primary']['sha256']}`  source {d['primary']['source']}"]
    for a in d["auxiliary"]:
        L.append(f"- auxiliary `{a['name']}`: {a['symbol']} {a['timeframe']}  {a['rows']:,} rows  {a['first_open']} -> "
                 f"{a['last_open']}  sha256 `{a['sha256']}`")
    L += [f"- strategy: {m['strategy']['name']} ({m['strategy']['class']})  trading config hash `{m['strategy']['config_hash_trading']}`  "
          f"config snapshot sha256 `{m['configuration']['snapshot_sha256']}`",
          f"- symbol / timeframe: {m['symbol']} {m['timeframe']}   benchmark range {m['benchmark_date_range']['start']} -> "
          f"{m['benchmark_date_range']['end']}",
          f"- starting equity {m['starting_equity']:.2f}   fee {m['fee_rate_pct']}% per side   slippage {m['slippage_pct']}%",
          f"- validation status: **{m['validation_status']}**   dataset hash verified: yes", ""]
    return L


def _validation(v: Dict[str, Any]) -> list:
    L = ["## Data quality (Step 10 validation, nothing repaired or filled)", ""]
    p = v["series"]["primary"]
    L += [f"- primary: {p['rows']:,} bars, {p['first_open']} -> {p['last_open']}, timeframe {p['timeframe']}, "
          f"expected {p['expected_rows']} bars on grid, missing {p['missing_bars']} ({p['missing_pct']}%) in {len(p['gap_runs'])} gap runs, "
          f"duplicates {p['duplicates']}, status **{p['status']}**"]
    codes = {}
    for i in p["issues"]:
        codes.setdefault(i["code"], []).append(f"{i['severity']} x{i['count']}: {i['message']}")
    for code, label in (("V6", "OHLC violations"), ("V7", "outliers / zero-volume / flat runs"), ("V8", "final candle"),
                        ("V2", "timezone / grid"), ("V4", "duplicates"), ("V5", "gaps")):
        L.append(f"  - {label}: " + ("; ".join(codes[code]) if code in codes else "none"))
    for name, a in (v["series"].get("auxiliary") or {}).items():
        al = a.get("alignment") or {}
        L.append(f"- auxiliary `{name}`: {a['rows']:,} rows, {a['first_open']} -> {a['last_open']}, missing {a['missing_bars']}, "
                 f"duplicates {a['duplicates']}, status **{a['status']}**; alignment: first visible primary bar "
                 f"#{al.get('first_visible_primary_index')}, coverage {al.get('coverage_pct')}%, ends before primary: "
                 f"{al.get('aux_ends_before_primary')}")
    if v["warnings"]:
        L += ["", "Warnings (not hidden):"] + [f"- {w}" for w in v["warnings"]]
    if v["critical_problems"]:
        L += ["", "Critical problems:"] + [f"- {w}" for w in v["critical_problems"]]
    return L + [""]


def render_markdown(res) -> str:
    m, x, v = res.manifest, res.metrics, res.validation
    L = _header(m) + _validation(v)
    L += ["## Signal funnel (what the strategy did)", ""]
    f = x["signal_funnel"]
    L += ["| stage | count |", "|---|---|",
          f"| BUY signals emitted by SMCStrategy | {_f(f['buy_signals'], '{}')} |",
          f"| rejected by the Risk Engine | {f['risk_rejected_buys']} |",
          f"| risk-approved | {_f(f['risk_approved_buys'], '{}')} |",
          f"| executed (filled at next open) | {f['executed_buys']} |",
          f"| closed by stop / target / exit signal | {f['closed_trades']} |",
          f"| force-closed at end of data | {f['force_closed_end_of_data']} |", "",
          f"_{f['note']}_", ""]
    if x.get("strategy_diagnostics"):
        d = x["strategy_diagnostics"]
        L += [f"- setups armed {d['setups_armed']}, exit signals {d['exit_signals']}, risk-off skips {d['riskoff_skips']}, "
              f"gate failures {d['gate_failures']}", ""]
    L += ["## Performance", ""]
    if not x["statistics_available"]:
        L += [f"> **Ratio statistics unavailable:** {x['unavailable_reason']}. Counts and P&L are still reported.", ""]
    L += ["| metric | value |", "|---|---|",
          f"| starting equity | {x['starting_equity']:.2f} |", f"| ending equity | {x['ending_equity']:.2f} |",
          f"| net P&L | {x['net_pnl']:+.2f} |", f"| return | {x['return_pct']:+.2f}% |",
          f"| total trades | {x['total_trades']} (W {x['winning_trades']} / L {x['losing_trades']} / BE {x['breakeven_trades']}) |",
          f"| win rate | {_f(x['win_rate_pct'], '{:.1f}%')} |", f"| expectancy per trade | {_f(x['expectancy'])} |",
          f"| average R | {_f(x['average_r'], '{:+.3f}')} |", f"| median R | {_f(x['median_r'], '{:+.3f}')} |",
          f"| profit factor | {_f(x['profit_factor'])} |",
          f"| max drawdown | {x['max_drawdown_pct']:.2f}% ({x['max_drawdown_bars']} bars) |",
          f"| max consecutive losses | {x['max_consecutive_losses']} |",
          f"| average trade duration | {_f(x['average_trade_duration_bars'], '{:.1f} bars')} / {_f(x['average_trade_duration_hours'], '{:.1f} h')} |",
          f"| fees paid | {x['fees']['total_fees']:.2f} (rate {x['fees']['fee_rate_pct']}% per side) |",
          f"| slippage | {x['slippage']['slippage_pct']}% (entries {x['slippage']['on_entries']}, stops {x['slippage']['on_stops']}, "
          f"targets {x['slippage']['on_targets']}); estimated cost {x['slippage']['estimated_cost']:.2f} |",
          f"| exit reasons | {x['exit_reasons']} |", f"| risk rejections | {x['risk_rejections']} |", "",
          "## Interpretation", "",
          f"- This is a **{m['label']}** of the current, unmodified strategy and risk rules.",
          "- It is not evidence of profitability and must not be used to tune parameters by hand or automatically.",
          "- Historical backtest performance does not guarantee future performance.",
          "- Nothing was changed: config/config.yaml, strategy, risk rules and paper-trading state are untouched; "
          "no improvement proposal was generated or applied.", ""]
    return "\n".join(L)


def render_aborted(res) -> str:
    L = [f"# BENCHMARK ABORTED - {res.benchmark_id}", "", f"**Reason:** {res.abort_reason}", "",
         "No performance figures were produced because the input failed a critical check. Fix the dataset "
         "(re-download / re-export) instead of lowering validation thresholds.", ""]
    if res.validation:
        L += _validation(res.validation)
    return "\n".join(L)
