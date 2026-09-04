"""Controlled Improvement Framework (Step 9) - OFFLINE parameter analysis with human approval.

Searches a whitelisted strategy-parameter space with the existing BacktestEngine, validates
candidates with anchored walk-forward folds and a sealed holdout, and writes ranked reports and
proposal files under data/improvement/<run_id>/. It never modifies src/ or config/config.yaml,
never touches paper-trading state and never deploys anything.
"""

from src.improvement.apply import ProposalError, apply as apply_proposal, show as show_proposal
from src.improvement.candidates import Candidate, CandidateGenerator, SingleParameterStage
from src.improvement.evaluator import Evaluator
from src.improvement.runner import ImprovementDisabled, ImprovementResult, ImprovementRunner, InsufficientData
from src.improvement.space import WHITELIST, ParameterSpace, ParameterSpec, SpaceError
from src.improvement.splitter import DataSplitter, SplitConfig, SplitPlan

__all__ = ["ImprovementRunner", "ImprovementResult", "ImprovementDisabled", "InsufficientData", "ParameterSpace",
           "ParameterSpec", "SpaceError", "WHITELIST", "DataSplitter", "SplitConfig", "SplitPlan", "Candidate",
           "CandidateGenerator", "SingleParameterStage", "Evaluator", "apply_proposal", "show_proposal", "ProposalError"]
