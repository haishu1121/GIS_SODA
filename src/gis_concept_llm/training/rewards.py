"""GIS-first reward: executable spatial correctness outweighs trace formatting."""

from __future__ import annotations

from dataclasses import dataclass

from soda.rewards import extract_answer, normalize_answer, ooda_format_reward

from ..schema import ConceptExample
from ..verifiers import verifier_for


@dataclass(frozen=True)
class GISReward:
    answer: float
    gis: float
    reasoning: float
    format: float

    @property
    def total(self) -> float:
        return round(self.answer + self.gis + self.reasoning + self.format, 4)


def score_completion(example: ConceptExample, completion: str, *, gis_weight: float = 1.0, reasoning_weight: float = 0.1, format_weight: float = 0.05) -> GISReward:
    """Use a program verifier as the primary signal, not a formatting heuristic."""
    predicted = extract_answer(completion)
    answer = 1.0 if normalize_answer(predicted) == normalize_answer(example.gold_answer) else -1.0
    verification = verifier_for(example.task_type).verify(example, predicted)
    gis = gis_weight * verification.score
    minimal_tokens = [token.strip().casefold() for token in example.minimal_trace.replace("->", " ").split() if len(token.strip()) > 1]
    reasoning = reasoning_weight if any(token in completion.casefold() for token in minimal_tokens) else 0.0
    # Retain only a small outer-format incentive; it must never dominate GIS truth.
    format_reward = format_weight * (ooda_format_reward(completion) / 0.4)
    return GISReward(answer, gis, reasoning, format_reward)
