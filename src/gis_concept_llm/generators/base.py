from __future__ import annotations

import random


class SeededGenerator:
    def __init__(self, seed: int = 2026):
        self.rng = random.Random(seed)
        self._serial = 0

    def next_id(self, task: str) -> str:
        self._serial += 1
        return f"{task}-{self._serial:07d}"
