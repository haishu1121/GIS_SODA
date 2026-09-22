"""Thin extension points retained for business-specific data, training, and inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

from .schema import SPODExample


class DatasetAdapter(Protocol):
    """Convert a business record into an approved question/answer/OODA example."""
    def build_examples(self, records: Iterable[Mapping]) -> Iterable[SPODExample]: ...


class ModelBackend(Protocol):
    """A minimal inference contract; adapters should not depend on a model vendor."""
    def generate(self, prompt: str) -> str: ...


class QualityGate(Protocol):
    """Implement domain checks beyond generic OODA order and answer matching."""
    def check(self, example: SPODExample) -> list[str]: ...
