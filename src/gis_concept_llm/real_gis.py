"""Licensed real-GIS grounding for atomic concept tasks.

The module accepts normalized caller-supplied records instead of bundling a
city dataset. Callers declare data source, licence and CRS so a training job
cannot silently treat a map extract as anonymous, licence-free data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .generators.connectivity_generator import witness_path
from .generators.direction_generator import direction_of
from .reasoning import build_minimal_trace, build_ooda_trace
from .schema import ConceptExample


class RealGISSampler(Protocol):
    """Adapter contract for POI, AOI and road-network sources."""

    def sample_direction(self, city: str, anonymous: bool) -> Iterable[ConceptExample]: ...
    def sample_topology(self, city: str, anonymous: bool) -> Iterable[ConceptExample]: ...
    def sample_connectivity(self, city: str, anonymous: bool) -> Iterable[ConceptExample]: ...


@dataclass(frozen=True)
class GISProvenance:
    """Lineage that must travel with every real-GIS training example."""

    city: str
    source_name: str
    source_license: str
    crs: str
    source_url: str | None = None
    dataset_version: str | None = None
    analysis_crs: str | None = None

    def validate(self) -> None:
        required = (self.city, self.source_name, self.source_license, self.crs)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise ValueError("real GIS provenance requires city, source_name, source_license and crs")

    def as_metadata(self) -> dict[str, str]:
        self.validate()
        result = {
            "city": self.city,
            "source_name": self.source_name,
            "source_license": self.source_license,
            "crs": self.crs,
        }
        if self.source_url:
            result["source_url"] = self.source_url
        if self.dataset_version:
            result["dataset_version"] = self.dataset_version
        if self.analysis_crs:
            result["analysis_crs"] = self.analysis_crs
        return result


def require_real_gis_metadata(record: Mapping[str, Any]) -> None:
    """Reject records that cannot later be traced to a city and licence."""

    required = {"city", "source_license", "crs"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"real GIS records require provenance fields: {sorted(missing)}")
    if not all(isinstance(record[field], str) and record[field].strip() for field in required):
        raise ValueError("real GIS provenance fields must be nonempty strings")


def topology_relation(geometry_a: Any, geometry_b: Any) -> str:
    """Return A's exact relation to B via Shapely predicates."""

    if geometry_a.contains(geometry_b):
        return "Contains"
    if geometry_a.within(geometry_b):
        return "Within"
    if geometry_a.touches(geometry_b):
        return "Touches"
    if geometry_a.intersects(geometry_b):
        return "Intersects"
    return "Disjoint"


class RealGISGrounder:
    """Build named/anonymous counterparts from normalized licensed GIS records.

    When source and analysis CRS differ, coordinates and GeoJSON geometries
    are transformed before truth computation. Use one local projected CRS per
    city, e.g. that city's UTM zone, for a stable directional interpretation.
    """

    def __init__(self, provenance: GISProvenance) -> None:
        provenance.validate()
        if provenance.crs.upper() in {"EPSG:4326", "OGC:CRS84", "CRS:84"} and not provenance.analysis_crs:
            raise ValueError("geographic source CRS requires a projected analysis_crs for real-GIS grounding")
        self.provenance = provenance

    def paired_direction(self, *, scenario_id: str, point_a: Mapping[str, Any], point_b: Mapping[str, Any]) -> tuple[ConceptExample, ConceptExample]:
        return (
            self.direction(scenario_id=scenario_id, point_a=point_a, point_b=point_b, anonymous=False),
            self.direction(scenario_id=scenario_id, point_a=point_a, point_b=point_b, anonymous=True),
        )

    def direction(self, *, scenario_id: str, point_a: Mapping[str, Any], point_b: Mapping[str, Any], anonymous: bool) -> ConceptExample:
        a, b = self._project_point(point_a), self._project_point(point_b)
        answer = direction_of(a, b)
        a_name = "Object A" if anonymous else self._entity_name(point_a, "point_a")
        b_name = "Object B" if anonymous else self._entity_name(point_b, "point_b")
        metadata = self._metadata(scenario_id, {"a": list(a), "b": list(b), "answer": answer})
        metadata["entity_refs"] = {"a": self._entity_ref(point_a), "b": self._entity_ref(point_b)}
        return self._example(scenario_id, "direction", "Location", "relative_direction", f"What is the direction of {b_name} relative to {a_name}?", answer, anonymous, metadata)

    def paired_topology(self, *, scenario_id: str, geometry_a: Mapping[str, Any], geometry_b: Mapping[str, Any]) -> tuple[ConceptExample, ConceptExample]:
        return (
            self.topology(scenario_id=scenario_id, geometry_a=geometry_a, geometry_b=geometry_b, anonymous=False),
            self.topology(scenario_id=scenario_id, geometry_a=geometry_a, geometry_b=geometry_b, anonymous=True),
        )

    def topology(self, *, scenario_id: str, geometry_a: Mapping[str, Any], geometry_b: Mapping[str, Any], anonymous: bool) -> ConceptExample:
        shape, mapping = self._shapely()
        a = self._project_geometry(shape(self._geometry(geometry_a)))
        b = self._project_geometry(shape(self._geometry(geometry_b)))
        if a.is_empty or b.is_empty or not a.is_valid or not b.is_valid:
            raise ValueError("topology examples require nonempty valid geometries")
        answer = topology_relation(a, b)
        a_name = "Object A" if anonymous else self._entity_name(geometry_a, "geometry_a")
        b_name = "Object B" if anonymous else self._entity_name(geometry_b, "geometry_b")
        metadata = self._metadata(scenario_id, {
            "geometry_format": "geojson",
            "geometry_a": {"type": "geojson", "geometry": mapping(a)},
            "geometry_b": {"type": "geojson", "geometry": mapping(b)},
            "answer": answer,
        })
        metadata["entity_refs"] = {"a": self._entity_ref(geometry_a), "b": self._entity_ref(geometry_b)}
        return self._example(scenario_id, "topology", "Object", "topology_relation", f"What is {a_name}'s topological relation to {b_name}?", answer, anonymous, metadata)

    def paired_connectivity(self, *, scenario_id: str, nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Sequence[str]], source: str, target: str) -> tuple[ConceptExample, ConceptExample]:
        return (
            self.connectivity(scenario_id=scenario_id, nodes=nodes, edges=edges, source=source, target=target, anonymous=False),
            self.connectivity(scenario_id=scenario_id, nodes=nodes, edges=edges, source=source, target=target, anonymous=True),
        )

    def connectivity(self, *, scenario_id: str, nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Sequence[str]], source: str, target: str, anonymous: bool) -> ConceptExample:
        if source not in nodes or target not in nodes:
            raise ValueError("source and target must be declared nodes")
        clean_edges = [tuple(edge) for edge in edges]
        if any(len(edge) != 2 or edge[0] not in nodes or edge[1] not in nodes for edge in clean_edges):
            raise ValueError("every edge must contain two declared node IDs")
        path = witness_path(clean_edges, source, target)
        answer = "Connected" if path else "Disconnected"
        node_alias = {node_id: f"Node {index + 1}" for index, node_id in enumerate(sorted(nodes))} if anonymous else {node_id: self._entity_name(node, node_id) for node_id, node in nodes.items()}
        display_edges = [(node_alias[left], node_alias[right]) for left, right in clean_edges]
        metadata = self._metadata(scenario_id, {
            "edges": [list(edge) for edge in clean_edges], "source": source, "target": target,
            "witness_path": path, "answer": answer,
            "node_refs": {node_id: self._entity_ref(node) for node_id, node in nodes.items()},
        })
        return self._example(scenario_id, "connectivity", "Network", "connectivity", f"Road-network links: {display_edges}. Are {node_alias[source]} and {node_alias[target]} connected?", answer, anonymous, metadata)

    def _example(self, scenario_id: str, task_type: str, concept: str, skill: str, question: str, answer: str, anonymous: bool, metadata: dict[str, Any]) -> ConceptExample:
        variant = "anonymous" if anonymous else "named"
        return ConceptExample(
            id=f"real-{task_type}-{scenario_id}-{variant}", task_type=task_type, concept=concept, skill=skill,
            question=question, gold_answer=answer, source_type="real_gis", anonymous=anonymous, city=self.provenance.city,
            metadata=metadata, minimal_trace=build_minimal_trace(task_type, metadata), ooda=build_ooda_trace(task_type, metadata),
            split_tags=("real_gis", variant, f"city:{self.provenance.city}"),
        )

    def _metadata(self, scenario_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = self.provenance.as_metadata()
        metadata.update(value)
        metadata["scenario_id"] = scenario_id
        require_real_gis_metadata(metadata)
        return metadata

    def _project_point(self, point: Mapping[str, Any]) -> tuple[float, float]:
        coordinates = point.get("coordinates")
        if not isinstance(coordinates, Sequence) or len(coordinates) != 2:
            raise ValueError("point records require a two-value 'coordinates' array")
        x, y = float(coordinates[0]), float(coordinates[1])
        if self.provenance.analysis_crs and self.provenance.analysis_crs != self.provenance.crs:
            x, y = self._transformer().transform(x, y)
        return x, y

    def _project_geometry(self, geometry: Any) -> Any:
        if not self.provenance.analysis_crs or self.provenance.analysis_crs == self.provenance.crs:
            return geometry
        try:
            from shapely.ops import transform
        except ImportError as exc:
            raise RuntimeError("GeoJSON grounding requires the optional 'gis' dependencies") from exc
        return transform(self._transformer().transform, geometry)

    def _transformer(self) -> Any:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError("CRS transformation requires the optional 'gis' dependencies") from exc
        if not self.provenance.analysis_crs:
            raise ValueError("analysis_crs is required to transform GIS coordinates")
        return Transformer.from_crs(self.provenance.crs, self.provenance.analysis_crs, always_xy=True)

    @staticmethod
    def _shapely() -> tuple[Any, Any]:
        try:
            from shapely.geometry import mapping, shape
        except ImportError as exc:
            raise RuntimeError("GeoJSON topology grounding requires the optional 'gis' dependencies") from exc
        return shape, mapping

    @staticmethod
    def _geometry(record: Mapping[str, Any]) -> Mapping[str, Any]:
        geometry = record.get("geometry")
        if not isinstance(geometry, Mapping) or "type" not in geometry or "coordinates" not in geometry:
            raise ValueError("geometry records require a GeoJSON 'geometry' object")
        return geometry

    @staticmethod
    def _entity_name(record: Mapping[str, Any], fallback: str) -> str:
        name = record.get("name")
        return str(name).strip() if name is not None and str(name).strip() else fallback

    @staticmethod
    def _entity_ref(record: Mapping[str, Any]) -> str:
        identifier = record.get("id")
        return str(identifier).strip() if identifier is not None and str(identifier).strip() else "unidentified"
