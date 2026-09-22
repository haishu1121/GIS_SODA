from __future__ import annotations

from ..generators.topology_generator import topology_of
from ..schema import ConceptExample
from .base import Verification


class TopologyVerifier:
    def verify(self, example: ConceptExample, predicted: str) -> Verification:
        if example.task_type != "topology":
            raise ValueError("TopologyVerifier only accepts topology examples")
        if example.metadata.get("geometry_format") == "geojson":
            try:
                from shapely.geometry import shape
            except ImportError as exc:
                raise RuntimeError("GeoJSON topology verification requires the optional 'gis' dependencies") from exc

            from ..real_gis import topology_relation

            a = shape(example.metadata["geometry_a"]["geometry"])
            b = shape(example.metadata["geometry_b"]["geometry"])
            expected = topology_relation(a, b)
        else:
            a = tuple(example.metadata["geometry_a"]["coordinates"])
            b = tuple(example.metadata["geometry_b"]["coordinates"])
            expected = topology_of(a, b)
        output = predicted.strip().title()
        correct = output == expected
        return Verification(correct, 1.0 if correct else -1.0, expected, output, "computed from geometry boundaries and interiors")
