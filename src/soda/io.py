"""JSONL I/O with validation at the dataset boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import SPODExample


def write_jsonl(path: str | Path, examples: Iterable[SPODExample]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            example.validate()
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[SPODExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    examples.append(SPODExample.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSONL record at line {line_no}") from exc
    return examples
