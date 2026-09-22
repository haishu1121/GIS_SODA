"""Concept-specific minimal traces: rules, not verbose template imitation."""

from __future__ import annotations


def build_minimal_trace(task_type: str, metadata: dict) -> str:
    if task_type == "direction":
        ax, ay = metadata["a"]
        bx, by = metadata["b"]
        return f"dx={bx}-{ax}={bx-ax}; dy={by}-{ay}={by-ay}; sign(dx,dy) -> {metadata['answer']}"
    if task_type == "topology":
        return f"Compare interiors and boundaries of A/B -> {metadata['answer']}"
    if task_type == "connectivity":
        path = metadata.get("witness_path")
        return f"{'Path ' + ' -> '.join(path) + ' exists' if path else 'No path exists between the two nodes'} -> {metadata['answer']}"
    raise ValueError(f"no minimal trace builder for {task_type}")
