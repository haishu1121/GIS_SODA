from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schema import ConceptExample


@dataclass(frozen=True)
class Verification:
    correct: bool
    score: float
    expected: str
    predicted: str
    diagnostic: str


class GISVerifier(Protocol):
    def verify(self, example: ConceptExample, predicted: str) -> Verification: ...
