"""Atomic concept evaluation, stratified by provenance and OOD condition."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from soda.rewards import normalize_answer

from ..schema import ConceptExample, TraceStyle
from ..verifiers import verifier_for


@dataclass(frozen=True)
class AtomicEvaluation:
    count: int
    answer_accuracy: float
    verifier_accuracy: float
    by_slice: dict[str, float]


def evaluate_atomic(examples: Iterable[ConceptExample], predict: Callable[[str], str], style: TraceStyle = TraceStyle.QA) -> AtomicEvaluation:
    values = list(examples)
    if not values:
        raise ValueError("cannot evaluate an empty set")
    answer_correct, verifier_correct = [], []
    slices: dict[str, list[bool]] = defaultdict(list)
    for example in values:
        predicted = predict(example.prompt(style))
        correct = normalize_answer(predicted) == normalize_answer(example.gold_answer)
        verified = verifier_for(example.task_type).verify(example, predicted).correct
        answer_correct.append(correct)
        verifier_correct.append(verified)
        city_slice = f"city/{example.city}" if example.city else "city/none"
        for slice_name in (f"concept/{example.concept}", f"source/{example.source_type}", city_slice, "anonymous" if example.anonymous else "named", *example.split_tags):
            slices[slice_name].append(verified)
    return AtomicEvaluation(len(values), sum(answer_correct) / len(values), sum(verifier_correct) / len(values), {name: sum(group) / len(group) for name, group in sorted(slices.items())})
