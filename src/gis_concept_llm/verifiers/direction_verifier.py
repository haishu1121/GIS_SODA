from __future__ import annotations

from ..generators.direction_generator import direction_of
from ..schema import ConceptExample
from .base import Verification


class DirectionVerifier:
    def verify(self, example: ConceptExample, predicted: str) -> Verification:
        if example.task_type != "direction":
            raise ValueError("DirectionVerifier only accepts direction examples")
        expected = direction_of(tuple(example.metadata["a"]), tuple(example.metadata["b"]))
        output = predicted.strip().upper()
        correct = output == expected
        return Verification(correct, 1.0 if correct else -1.0, expected, output, f"sign(dx, dy) gives {expected}")
