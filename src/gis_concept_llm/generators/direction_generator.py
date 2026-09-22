"""Synthetic and real-GIS-ready relative-direction data generator."""

from __future__ import annotations

from ..reasoning import build_minimal_trace, build_ooda_trace
from ..schema import ConceptExample
from .base import SeededGenerator


def direction_of(a: tuple[float, float], b: tuple[float, float]) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    labels = {(0, 1): "N", (1, 1): "NE", (1, 0): "E", (1, -1): "SE", (0, -1): "S", (-1, -1): "SW", (-1, 0): "W", (-1, 1): "NW"}
    if (sx, sy) == (0, 0):
        raise ValueError("direction is undefined for identical points")
    return labels[(sx, sy)]


class DirectionGenerator(SeededGenerator):
    QUESTION_TEMPLATES = (
        "What is the direction of {b_name} relative to {a_name}?",
        "Which direction from {a_name} leads to {b_name}?",
        "{b_name} lies on which side of {a_name}?",
    )

    def synthetic(self, linguistic_ood: bool = False) -> ConceptExample:
        a = (self.rng.randint(-20, 20), self.rng.randint(-20, 20))
        b = a
        while b == a:
            b = (self.rng.randint(-20, 20), self.rng.randint(-20, 20))
        answer = direction_of(a, b)
        template_index = 2 if linguistic_ood else self.rng.randrange(2)
        metadata = {"a": list(a), "b": list(b), "answer": answer}
        tags = ("synthetic", "linguistic_ood") if linguistic_ood else ("synthetic", "in_domain")
        return ConceptExample(self.next_id("direction"), "direction", "Location", "relative_direction",
            self.QUESTION_TEMPLATES[template_index].format(a_name=f"Object A at {a}", b_name=f"Object B at {b}"), answer,
            "synthetic", True, None, metadata, build_minimal_trace("direction", metadata), build_ooda_trace("direction", metadata), tags)

    def from_gis_points(self, *, point_a: tuple[float, float], point_b: tuple[float, float], name_a: str, name_b: str, city: str, anonymous: bool) -> ConceptExample:
        """Real-GIS adapter entry point; caller owns CRS projection and data licensing."""
        answer = direction_of(point_a, point_b)
        metadata = {"a": list(point_a), "b": list(point_b), "answer": answer, "crs_required": "projected/local"}
        a_name, b_name = ("Object A", "Object B") if anonymous else (name_a, name_b)
        return ConceptExample(self.next_id("direction"), "direction", "Location", "relative_direction",
            self.QUESTION_TEMPLATES[0].format(a_name=f"{a_name} at {point_a}", b_name=f"{b_name} at {point_b}"), answer,
            "real_gis", anonymous, city, metadata, build_minimal_trace("direction", metadata), build_ooda_trace("direction", metadata), ("real_gis", "anonymous" if anonymous else "named"))
