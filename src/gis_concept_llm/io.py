"""JSONL storage for concept data; preserves provenance required by OOD evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import ConceptExample, OODATrace


def write_jsonl(path: str | Path, examples: Iterable[ConceptExample]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            example.validate()
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[ConceptExample]:
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            ooda = OODATrace(**raw["ooda"]) if raw.get("ooda") else None
            example = ConceptExample(
                id=raw["id"], task_type=raw["task_type"], concept=raw["concept"], skill=raw["skill"],
                question=raw["question"], gold_answer=raw["gold_answer"], source_type=raw["source_type"],
                anonymous=raw["anonymous"], city=raw.get("city"), metadata=raw["metadata"],
                minimal_trace=raw["minimal_trace"], ooda=ooda, split_tags=tuple(raw.get("split_tags", [])),
            )
            example.validate()
            examples.append(example)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid concept JSONL record at line {line_no}") from exc
    return examples
