"""Graph connectivity tasks; deliberately separate from shortest-path training."""

from __future__ import annotations

from ..reasoning import build_minimal_trace, build_ooda_trace
from ..schema import ConceptExample
from .base import SeededGenerator


def witness_path(edges: list[tuple[str, str]], source: str, target: str) -> list[str] | None:
    graph: dict[str, list[str]] = {}
    for left, right in edges:
        graph.setdefault(left, []).append(right)
        graph.setdefault(right, []).append(left)
    queue, seen = [[source]], {source}
    while queue:
        path = queue.pop(0)
        if path[-1] == target:
            return path
        for neighbor in graph.get(path[-1], []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(path + [neighbor])
    return None


class ConnectivityGenerator(SeededGenerator):
    def synthetic(self, structural_ood: bool = False) -> ConceptExample:
        connected = bool(self.rng.getrandbits(1))
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")] if connected else [("A", "B"), ("B", "C"), ("D", "E")]
        if structural_ood:
            edges = edges + [("C", "F"), ("F", "G")] if connected else edges + [("F", "G")]
        path = witness_path(edges, "A", "E")
        answer = "Connected" if path else "Disconnected"
        metadata = {"edges": [list(edge) for edge in edges], "source": "A", "target": "E", "witness_path": path, "answer": answer}
        tags = ("synthetic", "structural_ood") if structural_ood else ("synthetic", "in_domain")
        return ConceptExample(self.next_id("connectivity"), "connectivity", "Network", "connectivity",
            f"Undirected graph edges: {edges}. Are nodes A and E connected?", answer,
            "synthetic", True, None, metadata, build_minimal_trace("connectivity", metadata), build_ooda_trace("connectivity", metadata), tags)
