"""Data contracts for atomic GIS-concept supervision and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TraceStyle(str, Enum):
    QA = "qa"
    MINIMAL = "minimal"
    OODA = "ooda"


@dataclass(frozen=True)
class OODATrace:
    observe: str
    orient: str
    decide: str
    act: str

    def render(self) -> str:
        return f"Observe: {self.observe}\nOrient: {self.orient}\nDecide: {self.decide}\nAct: {self.act}"

    def validate(self) -> None:
        if not all(value.strip() for value in asdict(self).values()):
            raise ValueError("every OODA stage must be nonempty")


@dataclass(frozen=True)
class ConceptExample:
    """A single-core-skill record. Geometry/network truth stays in metadata."""

    id: str
    task_type: str
    concept: str
    skill: str
    question: str
    gold_answer: str
    source_type: str  # synthetic | real_gis
    anonymous: bool
    city: str | None
    metadata: dict[str, Any]
    minimal_trace: str
    ooda: OODATrace | None = None
    split_tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        required = (self.id, self.task_type, self.concept, self.skill, self.question, self.gold_answer, self.source_type, self.minimal_trace)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise ValueError("example identity, task, answer, source and minimal trace must be nonempty")
        if self.source_type not in {"synthetic", "real_gis"}:
            raise ValueError("source_type must be synthetic or real_gis")
        if self.ooda:
            self.ooda.validate()

    def prompt(self, style: TraceStyle = TraceStyle.OODA) -> str:
        suffix = {
            TraceStyle.QA: "Answer only.",
            TraceStyle.MINIMAL: "Show only the minimal GIS rule, then the answer.",
            TraceStyle.OODA: "Use Observe, Orient, Decide, and Act; put the final answer in Act.",
        }[style]
        return f"Task type: {self.task_type}\nQuestion: {self.question}\n{suffix}\n"

    def completion(self, style: TraceStyle = TraceStyle.OODA) -> str:
        if style is TraceStyle.QA:
            return self.gold_answer
        if style is TraceStyle.MINIMAL:
            return f"{self.minimal_trace}\nAnswer: {self.gold_answer}"
        if not self.ooda:
            raise ValueError("OODA completion requested but no OODA trace exists")
        return self.ooda.render()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Canonical aliases make records compatible with the implementation plan.
        value["input"] = self.question
        value["answer"] = self.gold_answer
        value["reasoning"] = {"minimal_trace": self.minimal_trace, "ooda": asdict(self.ooda) if self.ooda else None}
        return value
