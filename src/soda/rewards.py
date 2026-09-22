"""Paper-aligned conclusion and OODA-format reward functions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import STAGES


def normalize_answer(value: str) -> str:
    """Normalize only whitespace/case/punctuation around a discrete answer."""
    return re.sub(r"\s+", " ", value.strip().casefold()).strip(" .,!\"'")


def extract_answer(completion: str) -> str:
    """Prefer an Act-stage output; accept a bare single-line answer for evaluation."""
    match = re.search(r"(?im)^\s*act\s*:\s*(?:output\s+)?(.+?)\s*$", completion)
    if match:
        value = match.group(1)
        value = re.sub(r"(?i)^the answer is\s+", "", value).strip()
        return value.rstrip(".")
    return completion.strip().splitlines()[-1].strip().rstrip(".")


def conclusion_reward(completion: str, expected: str) -> float:
    """+1 exact, -1 wrong, -1.4 when no recognizable reasoning, as specified."""
    has_reasoning = any(re.search(rf"(?im)^\s*{stage}\s*:", completion) for stage in STAGES)
    if not has_reasoning:
        return -1.4
    return 1.0 if normalize_answer(extract_answer(completion)) == normalize_answer(expected) else -1.0


def ooda_format_reward(completion: str) -> float:
    """Score stage inclusion and order in [-.4, .4], without claiming semantic proof."""
    positions = []
    for stage in STAGES:
        found = re.search(rf"(?im)^\s*{stage}\s*:\s*\S", completion)
        positions.append(found.start() if found else None)
    reward = sum(0.1 if pos is not None else -0.1 for pos in positions)
    present = [pos for pos in positions if pos is not None]
    if present != sorted(present):
        reward -= 0.1
    return max(-0.4, min(0.4, round(reward, 4)))


def soda_reward(completion: str, expected: str) -> float:
    # Keep the documented [-1.8, 1.4] decimal values stable for logging/tests.
    return round(conclusion_reward(completion, expected) + ooda_format_reward(completion), 4)


@dataclass(frozen=True)
class RewardBreakdown:
    answer: float
    ooda: float

    @property
    def total(self) -> float:
        return self.answer + self.ooda

    @classmethod
    def score(cls, completion: str, expected: str) -> "RewardBreakdown":
        return cls(conclusion_reward(completion, expected), ooda_format_reward(completion))
