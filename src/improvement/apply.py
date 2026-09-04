"""Manual proposal handling - the ONLY path by which a proposal becomes a config file.

    show(proposal_id)                    -> prints overlay, evidence and a unified diff; writes NOTHING
    apply(proposal_id, confirm=<id>)     -> writes config/config.proposed.<id>.yaml ONLY, never config.yaml

`apply` requires the confirm token to equal the proposal id exactly. It refuses to run if the
proposal's baseline config hash no longer matches the current config.yaml (the evidence would be
stale). It marks the proposal YAML `status: applied_to_proposed_file` (inside data/improvement/,
not config/). Nothing is activated: the operator must review the proposed file and copy values
by hand; paper-trading state will then require `--reset` because its config hash changes.
"""

import difflib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.execution.state_store import config_hash
from src.improvement.space import ParameterSpace, apply_overlay


class ProposalError(RuntimeError):
    pass


def find_proposal(results_dir: Path, proposal_id: str, run_id: Optional[str] = None) -> Path:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ProposalError(f"no improvement runs under {results_dir}")
    runs = [results_dir / run_id] if run_id else sorted((p for p in results_dir.iterdir() if p.is_dir()), reverse=True)
    for r in runs:
        p = r / "proposals" / f"{proposal_id}.yaml"
        if p.exists():
            return p
    raise ProposalError(f"proposal {proposal_id!r} not found under {results_dir}")


def load_proposal(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_proposed_config(config_path: Path, overlay: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Return (original_text, proposed_text, proposed_dict). The proposed text is a YAML dump of the
    current config with the overlay applied (comments are not preserved - it is a review artefact)."""
    original = Path(config_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(original)
    proposed = apply_overlay(cfg, overlay)
    buf = io.StringIO()
    yaml.safe_dump(proposed, buf, sort_keys=False, default_flow_style=False)
    return original, buf.getvalue(), proposed


def unified_diff(config_path: Path, overlay: Dict[str, Any]) -> str:
    original = Path(config_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(original)
    base_dump = io.StringIO()
    yaml.safe_dump(cfg, base_dump, sort_keys=False, default_flow_style=False)
    _, proposed_text, _ = render_proposed_config(config_path, overlay)
    return "".join(difflib.unified_diff(base_dump.getvalue().splitlines(True), proposed_text.splitlines(True),
                                        fromfile="config.yaml (normalised)", tofile="config.proposed (normalised)"))


def show(results_dir: Path, config_path: Path, proposal_id: str, run_id: Optional[str] = None) -> str:
    """Read-only. Returns the text to print."""
    p = find_proposal(results_dir, proposal_id, run_id)
    prop = load_proposal(p)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    lines: List[str] = [
        f"Proposal {prop['proposal_id']}  (run {prop['run_id']})  status: {prop['status']}",
        f"  file            : {p}",
        f"  baseline hash   : {prop['baseline_config_hash']}  current config hash: {config_hash(cfg)}"
        + ("" if prop["baseline_config_hash"] == config_hash(cfg) else "   ** MISMATCH - evidence is stale **"),
        f"  overlay         : {prop['parameter_overlay']}",
        f"  baseline values : {prop['baseline_values']}",
        f"  reason          : {prop['reason']}",
        f"  evidence        : {prop['evidence']}",
        f"  holdout         : {prop['holdout_result']['verdict']} {prop['holdout_result']['metrics']}",
        "", "Diff vs current config.yaml:", unified_diff(config_path, prop["parameter_overlay"]) or "  (no difference)",
        "", "Nothing has been written. To materialise a proposed config file:",
        f"  python src/main.py proposal apply {proposal_id} --confirm {proposal_id}",
    ]
    return "\n".join(lines)


def apply(results_dir: Path, config_path: Path, proposal_id: str, confirm: Optional[str],
          run_id: Optional[str] = None) -> Path:
    """Write config/config.proposed.<id>.yaml. Never touches config.yaml."""
    if confirm != proposal_id:
        raise ProposalError(f"refusing: --confirm must equal the proposal id exactly ({proposal_id!r})")
    config_path = Path(config_path)
    before = config_path.read_bytes()
    p = find_proposal(results_dir, proposal_id, run_id)
    prop = load_proposal(p)
    if prop.get("status") not in ("recommended_pending_human_review", "applied_to_proposed_file"):
        raise ProposalError(f"refusing: proposal status is {prop.get('status')!r}, not recommended")
    cfg = yaml.safe_load(before.decode("utf-8"))
    if prop["baseline_config_hash"] != config_hash(cfg):
        raise ProposalError("refusing: config.yaml changed since this proposal was produced (hash mismatch); re-run improve")
    overlay = dict(prop["parameter_overlay"])
    problems = ParameterSpace(cfg).validate_overlay(overlay)
    if problems:
        raise ProposalError("refusing: overlay no longer admissible: " + "; ".join(problems))

    _, proposed_text, _ = render_proposed_config(config_path, overlay)
    target = config_path.parent / f"config.proposed.{proposal_id}.yaml"
    header = (f"# PROPOSED configuration generated from improvement proposal {proposal_id} (run {prop['run_id']}).\n"
              f"# Overlay: {overlay}\n"
              f"# This file is NOT active. Review it and copy values into config/config.yaml by hand if you accept it.\n"
              f"# Paper-trading state (data/paper/state.json) will require --reset after a config change.\n")
    target.write_text(header + proposed_text, encoding="utf-8")

    prop["status"] = "applied_to_proposed_file"
    prop["proposed_file"] = str(target)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(prop, f, sort_keys=False, default_flow_style=False)

    assert config_path.read_bytes() == before, "config.yaml must never change"
    return target
