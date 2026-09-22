"""Load and enforce the concept-to-skill taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ConceptExample


class ConceptTaxonomy:
    def __init__(self, raw: dict):
        self.raw = raw
        self.concepts: dict[str, dict] = raw["concepts"]

    @classmethod
    def load_default(cls) -> "ConceptTaxonomy":
        root = Path(__file__).resolve().parents[2]
        return cls(json.loads((root / "taxonomy" / "concept_skill_taxonomy.json").read_text(encoding="utf-8")))

    def validate_example(self, example: ConceptExample) -> None:
        example.validate()
        if example.concept not in self.concepts:
            raise ValueError(f"unknown concept {example.concept}")
        definition = self.concepts[example.concept]
        if example.skill not in definition["skills"] or example.task_type not in definition["task_types"]:
            raise ValueError("task type / skill does not belong to declared concept")
