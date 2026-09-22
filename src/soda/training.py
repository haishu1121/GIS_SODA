"""Model-agnostic data formatting and GRPO bookkeeping.

Heavy PyTorch/Transformers imports intentionally live in CLI scripts so domain
adapters and quality checks remain usable in a lightweight production service.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Iterable

from .rewards import RewardBreakdown
from .schema import SPODExample


def sft_text(example: SPODExample) -> str:
    """The causal-LM target: prompt followed by all four gold OODA stages."""
    return example.prompt() + example.completion()


def group_advantages(rewards: Iterable[float], epsilon: float = 1e-6) -> list[float]:
    """GRPO's critic-free, within-prompt standardized reward signal."""
    values = list(rewards)
    if not values:
        raise ValueError("a GRPO group cannot be empty")
    mean = fmean(values)
    deviation = pstdev(values)
    return [(value - mean) / (deviation + epsilon) for value in values]


@dataclass(frozen=True)
class CandidateScore:
    completion: str
    reward: RewardBreakdown


def score_group(example: SPODExample, completions: Iterable[str]) -> list[CandidateScore]:
    return [CandidateScore(completion, RewardBreakdown.score(completion, example.answer)) for completion in completions]
