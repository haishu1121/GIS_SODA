"""Axis-aligned polygon tasks with exact executable topology truth."""

from __future__ import annotations

from ..reasoning import build_minimal_trace, build_ooda_trace
from ..schema import ConceptExample
from .base import SeededGenerator


def topology_of(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> str:
    """Classify axis-aligned boxes as Contains/Within/Touches/Intersects/Disjoint."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2:
        return "Contains"
    if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
        return "Within"
    overlap_x, overlap_y = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if overlap_x < 0 or overlap_y < 0:
        return "Disjoint"
    if overlap_x == 0 or overlap_y == 0:
        return "Touches"
    return "Intersects"


class TopologyGenerator(SeededGenerator):
    CASES = {
        "Contains": ((0, 0, 8, 8), (2, 2, 5, 5)),
        "Within": ((2, 2, 5, 5), (0, 0, 8, 8)),
        "Touches": ((0, 0, 3, 3), (3, 1, 6, 2)),
        "Intersects": ((0, 0, 4, 4), (3, 2, 7, 6)),
        "Disjoint": ((0, 0, 2, 2), (4, 4, 6, 6)),
    }

    def synthetic(self, structural_ood: bool = False) -> ConceptExample:
        relation = self.rng.choice(tuple(self.CASES))
        a, b = self.CASES[relation]
        scale = 3 if structural_ood else 1
        a, b = tuple(value * scale for value in a), tuple(value * scale for value in b)
        answer = topology_of(a, b)
        metadata = {"geometry_a": {"type": "bbox", "coordinates": list(a)}, "geometry_b": {"type": "bbox", "coordinates": list(b)}, "answer": answer}
        tags = ("synthetic", "structural_ood") if structural_ood else ("synthetic", "in_domain")
        return ConceptExample(self.next_id("topology"), "topology", "Object", "topology_relation",
            f"A={a} and B={b} are axis-aligned polygon bounding boxes. What is A's topological relation to B?", answer,
            "synthetic", True, None, metadata, build_minimal_trace("topology", metadata), build_ooda_trace("topology", metadata), tags)
