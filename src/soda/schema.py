"""Stable JSONL data contracts used by every stage of the reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STAGES = ("Observe", "Orient", "Decide", "Act")


@dataclass(frozen=True)
class OODATrace:
    observe: str
    orient: str
    decide: str
    act: str

    def render(self) -> str:
        return "\n".join(
            f"{label}: {value}"
            for label, value in zip(STAGES, (self.observe, self.orient, self.decide, self.act))
        )

    def valid(self) -> bool:
        return all(isinstance(value, str) and value.strip() for value in asdict(self).values())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OODATrace":
        return cls(**{key: str(value[key]) for key in ("observe", "orient", "decide", "act")})


@dataclass(frozen=True)
class SPODExample:
    id: str
    task: str
    tier: int
    question: str
    answer: str
    ooda: OODATrace
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id or not self.task or self.tier not in (1, 2, 3, 4):
            raise ValueError("id, task, and tier must be valid")
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("question and answer must be nonempty")
        if not self.ooda.valid():
            raise ValueError("all OODA stages must be nonempty")

    def completion(self) -> str:
        return self.ooda.render()

    def prompt(self) -> str:
        return f"Task: {self.task}\nQuestion: {self.question}\nProvide an OODA trace.\n"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ooda": asdict(self.ooda)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SPODExample":
        example = cls(
            id=str(value["id"]), task=str(value["task"]), tier=int(value["tier"]),
            question=str(value["question"]), answer=str(value["answer"]),
            ooda=OODATrace.from_dict(value["ooda"]), metadata=dict(value.get("metadata", {})),
        )
        example.validate()
        return example
