"""Report writer: report.md, ranking.csv, folds.csv, summary.json, proposals/P-<n>.yaml.

Writes ONLY under `<results_directory>/<run_id>/`. Never touches config/ or src/.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

SYNTHETIC_BANNER = ("> **SYNTHETIC DATA.** The input series is synthetic (bundled sample or generated test data), not market "
                    "data. Every number below verifies plumbing only and must NOT be read as trading performance.\n")
NO_CHANGE_FOOTER = ("---\n**No change has been applied.** This report and the files in `proposals/` are recommendations "
                    "for human review. Applying one is a separate manual step "
                    "(`python src/main.py proposal apply <id> --confirm <id>`) that writes "
                    "`config/config.proposed.<id>.yaml` only; `config/config.yaml` is never modified by the tool.\n")


def _f(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


class ReportWriter:
    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)

    def write(self, result: Dict[str, Any]) -> Dict[str, Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "proposals").mkdir(exist_ok=True)
        paths = {}
        paths["ranking.csv"] = self._ranking_csv(result["ranking"])
        paths["folds.csv"] = self._folds_csv(result["fold_rows"])
        paths["summary.json"] = self._summary(result)
        for p in result["proposals"]:
            paths[f"proposals/{p['proposal_id']}.yaml"] = self._proposal(p)
        paths["report.md"] = self._report_md(result)
        return paths

    def write_aborted(self, result: Dict[str, Any]) -> Dict[str, Path]:
        """Insufficient data / refused run: only report.md + summary.json, clearly marked."""
        self.out.mkdir(parents=True, exist_ok=True)
        paths = {"summary.json": self._summary(result)}
        lines = [f"# Improvement run `{result['run_id']}` - ABORTED", ""]
        if result.get("synthetic"):
            lines.append(SYNTHETIC_BANNER)
        lines += [f"**Reason:** {result['abort_reason']}", "", "No candidates were evaluated and no proposals were written.", "",
                  NO_CHANGE_FOOTER]
        p = self.out / "report.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        paths["report.md"] = p
        return paths

    # ------------------------------------------------------------ pieces
    def _ranking_csv(self, ranking: List[Dict[str, Any]]) -> Path:
        p = self.out / "ranking.csv"
        pd.DataFrame(ranking).to_csv(p, index=False)
        return p

    def _folds_csv(self, rows: List[Dict[str, Any]]) -> Path:
        p = self.out / "folds.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def _summary(self, result: Dict[str, Any]) -> Path:
        p = self.out / "summary.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result.get("summary", result), f, indent=2, sort_keys=True, default=str)
        return p

    def _proposal(self, prop: Dict[str, Any]) -> Path:
        p = self.out / "proposals" / f"{prop['proposal_id']}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(prop, f, sort_keys=False, default_flow_style=False)
        return p

    def _report_md(self, r: Dict[str, Any]) -> Path:
        s = r["summary"]
        L: List[str] = [f"# Improvement run `{r['run_id']}`", ""]
        if r.get("synthetic"):
            L.append(SYNTHETIC_BANNER)
        L += [
            f"- data: `{s['data']['file']}` {s['data']['symbol']} {s['data']['timeframe']}, {s['data']['bars']} bars, "
            f"{s['data']['first_timestamp']} -> {s['data']['last_timestamp']} (sha256 {s['data']['sha256'][:12]})",
            f"- baseline config hash: `{s['baseline_config_hash']}`",
            f"- split: holdout {s['split']['holdout']['bars']} bars (sealed), {len(s['split']['folds'])} anchored folds, "
            f"warm-up {s['warmup_bars']} bars per slice",
            f"- search: {s['search']['method']} (single-parameter coordinate descent), {s['search']['candidates']} candidates, "
            f"{s['search']['backtests_run']} backtests",
            f"- scoring: median over OOS folds of `avg_R * sqrt(trades) - {s['scoring']['dd_penalty']} * maxDD%`",
        ]
        skipped = s.get("skipped_parameters") or {}
        if skipped:
            L += [f"- parameters with no legal grid candidate under max_parameter_change_pct/bounds ({len(skipped)}):"]
            L += [f"  - `{k}`: {v}" for k, v in sorted(skipped.items())]
        L += [
            "",
            "## Baseline (same pipeline as every candidate)",
            "",
            "| fold | IS trades | IS avg R | IS maxDD% | OOS trades | OOS avg R | OOS net | OOS maxDD% | OOS score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in r["fold_rows"]:
            if row["candidate_id"] == "baseline" and row["role"] == "oos":
                k = row["fold"]
                is_row = next(x for x in r["fold_rows"] if x["candidate_id"] == "baseline" and x["fold"] == k and x["role"] == "is")
                L.append(f"| {k} | {is_row['trades']} | {_f(is_row['average_r_multiple'], 3)} | {_f(is_row['max_drawdown_pct'])} | "
                         f"{row['trades']} | {_f(row['average_r_multiple'], 3)} | {_f(row['net_profit'])} | "
                         f"{_f(row['max_drawdown_pct'])} | {_f(row['score'], 3)} |")
        base = next(x for x in r["ranking"] if x["candidate_id"] == "baseline")
        L += ["", f"Baseline OOS score median **{_f(base['oos_score_median'], 3)}**, min {_f(base['oos_score_min'], 3)}, "
                  f"total OOS trades {base['oos_trades_total']}.", ""]

        L += ["## Ranked candidates (top 10)", "",
              "| rank | candidate | params changed | OOS trades | OOS score median | OOS score min | improvement % | "
              "constraints | holdout | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in r["ranking"][:10]:
            L.append(f"| {row['rank']} | {row['label']} | {row['params_changed']} | {row['oos_trades_total']} | "
                     f"{_f(row['oos_score_median'], 3)} | {_f(row['oos_score_min'], 3)} | {_f(row['improvement_pct'], 1)} | "
                     f"{'pass' if row['passed'] else 'FAIL'} | {row.get('holdout_verdict', '-')} | {row['verdict']} |")
        L.append("")

        if r["proposals"]:
            L += ["## Recommendations", ""]
            for p in r["proposals"]:
                L += [f"### {p['proposal_id']} - {p['status']}", "",
                      f"- overlay: `{p['parameter_overlay']}` (baseline `{p['baseline_values']}`)",
                      f"- reason: {p['reason']}",
                      f"- holdout: {p['holdout_result']['verdict']} - net {_f(p['holdout_result']['metrics'].get('net_profit'))}, "
                      f"trades {p['holdout_result']['metrics'].get('trades')}, avg R {_f(p['holdout_result']['metrics'].get('average_r_multiple'), 3)}",
                      ""]
        else:
            L += ["## Recommendations", "", "_None. No candidate passed every walk-forward constraint and the holdout._", ""]

        rejected = [x for x in r["ranking"] if not x["passed"] and x["candidate_id"] != "baseline"]
        if rejected:
            L += ["<details><summary>Rejected candidates (" + str(len(rejected)) + ")</summary>", "", "| candidate | reasons |", "|---|---|"]
            for row in rejected:
                L.append(f"| {row['label']} | {row['reasons'].replace('|', '/')} |")
            L += ["", "</details>", ""]
        L += [f"Reproduce: `python src/main.py improve` with the same `config/config.yaml` and data (see summary.json).", "",
              NO_CHANGE_FOOTER]
        p = self.out / "report.md"
        p.write_text("\n".join(L), encoding="utf-8")
        return p
