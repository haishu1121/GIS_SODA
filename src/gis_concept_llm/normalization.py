"""Normalize an OSM PBF snapshot into geometry layers and a routing graph."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


DRIVABLE_HIGHWAYS = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
})
POI_TAG_KEYS = ("amenity", "shop", "tourism", "office", "historic", "leisure", "public_transport")
AOI_TAG_KEYS = ("amenity", "building", "landuse", "leisure", "natural", "boundary", "place")


@dataclass(frozen=True)
class StudyBBox:
    """A WGS84 study boundary in ``min_lon, min_lat, max_lon, max_lat`` order."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def validate(self) -> None:
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError("study bbox must satisfy min_lon < max_lon and min_lat < max_lat")

    def as_list(self) -> list[float]:
        self.validate()
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


def is_drivable(tags: dict[str, str]) -> bool:
    return tags.get("highway") in DRIVABLE_HIGHWAYS and tags.get("access") not in {"no", "private"}


def directions_for(tags: dict[str, str]) -> tuple[bool, bool]:
    """Return forward/reverse travel permissions for an OSM way."""

    value = tags.get("oneway", "").lower()
    if value in {"yes", "true", "1"} or tags.get("junction") == "roundabout":
        return True, False
    if value == "-1":
        return False, True
    return True, True


def first_tag(tags: dict[str, str], keys: Iterable[str]) -> tuple[str | None, str | None]:
    for key in keys:
        if key in tags:
            return key, tags[key]
    return None, None


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import geopandas as gpd
        import networkx as nx
        import osmium
        from shapely.geometry import LineString, Point, box, shape
        from shapely.ops import transform
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "OSM normalization requires optional GIS dependencies. Install with: "
            "python -m pip install -e .[gis]"
        ) from exc
    return gpd, nx, osmium, LineString, Point, box, shape, transform, Transformer


def normalize_osm_pbf(
    *, pbf_path: str | Path, output_gpkg: str | Path, output_graphml: str | Path,
    output_manifest: str | Path, city: str, study_bbox: StudyBBox,
    analysis_crs: str = "EPSG:32650", overwrite: bool = False,
) -> dict[str, int]:
    """Build a projected GeoPackage and directed GraphML cache from one PBF.

    The study bbox filters output features but does not cut road geometries at
    the boundary, preserving source-node topology for routing. The raw PBF is
    never changed.
    """

    study_bbox.validate()
    pbf = Path(pbf_path)
    if not pbf.is_file():
        raise FileNotFoundError(f"PBF file does not exist: {pbf}")
    gpkg, graphml, manifest_path = (Path(output_gpkg), Path(output_graphml), Path(output_manifest))
    existing = [path for path in (gpkg, graphml, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"normalized outputs already exist; use --overwrite to replace: {names}")
    if overwrite:
        for path in existing:
            path.unlink()
    for path in (gpkg, graphml, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    gpd, nx, osmium, LineString, Point, box, shape, transform, Transformer = _dependencies()
    collector = _OSMCollector(osmium=osmium, line_string=LineString, point=Point, shape=shape)
    collector.apply_file(str(pbf), locations=True)
    filter_box = box(*study_bbox.as_list())

    road_rows = [row for row in collector.roads if row["geometry"].intersects(filter_box)]
    poi_rows = [row for row in collector.pois if row["geometry"].within(filter_box)]
    aoi_rows = [row for row in collector.aois if row["geometry"].intersects(filter_box)]
    edge_rows, junction_rows = _build_directed_edges(road_rows, LineString)

    crs = "EPSG:4326"
    layers = {
        "pois": _geodataframe(gpd, poi_rows, crs),
        "aois": _geodataframe(gpd, aoi_rows, crs),
        "road_centerlines": _geodataframe(gpd, road_rows, crs),
        "junctions": _geodataframe(gpd, junction_rows, crs),
        "road_edges": _geodataframe(gpd, edge_rows, crs),
    }
    layers = {name: frame.to_crs(analysis_crs) for name, frame in layers.items()}
    if not layers["road_edges"].empty:
        layers["road_edges"]["cost_m"] = layers["road_edges"].geometry.length.round(3)
    for index, (name, frame) in enumerate(layers.items()):
        frame.to_file(gpkg, layer=name, driver="GPKG", index=False, mode="w" if index == 0 else "a")

    _write_graph(nx, layers["junctions"], layers["road_edges"], graphml, city)
    counts = {name: int(len(frame)) for name, frame in layers.items()}
    manifest_path.write_text(json.dumps({
        "schema_version": "normalized-map/v1",
        "city": city,
        "input_pbf": str(pbf),
        "source_crs": "EPSG:4326",
        "analysis_crs": analysis_crs,
        "study_bbox_wgs84": study_bbox.as_list(),
        "layers": counts,
        "road_filter": "drivable highway classes; access=no/private excluded; oneway and roundabout direction applied",
        "geometry_policy": "features intersecting the study bbox are retained; roads are not boundary-clipped",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


class _OSMCollector:
    """Retains copied OSM primitives only while the handler callback is active."""

    def __init__(self, *, osmium: Any, line_string: Any, point: Any, shape: Any) -> None:
        self._base = osmium.SimpleHandler
        self._line_string = line_string
        self._point = point
        self._shape = shape
        self._geojson_factory = osmium.geom.GeoJSONFactory()
        self.pois: list[dict[str, Any]] = []
        self.roads: list[dict[str, Any]] = []
        self.aois: list[dict[str, Any]] = []

        class Handler(osmium.SimpleHandler):
            def node(inner_self, node: Any) -> None:
                self._on_node(node)

            def way(inner_self, way: Any) -> None:
                self._on_way(way)

            def area(inner_self, area: Any) -> None:
                self._on_area(area)

        self._handler = Handler()

    def apply_file(self, filename: str, locations: bool) -> None:
        self._handler.apply_file(filename, locations=locations)

    def _on_node(self, node: Any) -> None:
        tags = dict(node.tags)
        kind, category = first_tag(tags, POI_TAG_KEYS)
        if kind is None or not node.location.valid():
            return
        self.pois.append({
            "osm_type": "node", "osm_id": str(node.id), "name": tags.get("name"), "name_en": tags.get("name:en"),
            "category_key": kind, "category": category, "geometry": self._point(node.location.lon, node.location.lat),
        })

    def _on_way(self, way: Any) -> None:
        tags = dict(way.tags)
        if not is_drivable(tags):
            return
        try:
            coordinates = [(node.location.lon, node.location.lat) for node in way.nodes]
        except Exception:
            return
        if len(coordinates) < 2:
            return
        self.roads.append({
            "osm_type": "way", "osm_id": str(way.id), "name": tags.get("name"), "name_en": tags.get("name:en"),
            "highway": tags.get("highway"), "oneway": tags.get("oneway", ""), "access": tags.get("access", ""),
            "junction": tags.get("junction", ""), "node_ids": [str(node.ref) for node in way.nodes],
            "coordinates": coordinates, "tags": tags, "geometry": self._line_string(coordinates),
        })

    def _on_area(self, area: Any) -> None:
        tags = dict(area.tags)
        kind, category = first_tag(tags, AOI_TAG_KEYS)
        if kind is None:
            return
        try:
            geometry = self._shape(json.loads(self._geojson_factory.create_multipolygon(area)))
        except Exception:
            return
        if geometry.is_empty or not geometry.is_valid:
            return
        self.aois.append({
            "osm_type": "area", "osm_id": str(area.id), "name": tags.get("name"), "name_en": tags.get("name:en"),
            "category_key": kind, "category": category, "geometry": geometry,
        })


def _build_directed_edges(roads: list[dict[str, Any]], line_string: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_uses: dict[str, int] = {}
    node_coordinates: dict[str, tuple[float, float]] = {}
    for road in roads:
        for node_id, coordinate in zip(road["node_ids"], road["coordinates"]):
            node_uses[node_id] = node_uses.get(node_id, 0) + 1
            node_coordinates[node_id] = coordinate
    break_nodes = {
        node_id for road in roads for node_id in (road["node_ids"][0], road["node_ids"][-1])
    }
    break_nodes.update(node_id for node_id, count in node_uses.items() if count > 1)
    junctions = [
        {"osm_type": "node", "osm_id": node_id, "is_intersection": node_uses[node_id] > 1,
         "geometry": _point_from_coordinates(node_coordinates[node_id])}
        for node_id in sorted(break_nodes, key=int)
    ]
    edges: list[dict[str, Any]] = []
    edge_number = 0
    for road in roads:
        node_ids, coordinates = road["node_ids"], road["coordinates"]
        segment_start = 0
        forward, reverse = directions_for(road["tags"])
        for index in range(1, len(node_ids)):
            if node_ids[index] not in break_nodes:
                continue
            segment_ids, segment_coordinates = node_ids[segment_start:index + 1], coordinates[segment_start:index + 1]
            segment_start = index
            if len(segment_coordinates) < 2:
                continue
            if forward:
                edge_number += 1
                edges.append(_edge_row(road, edge_number, segment_ids[0], segment_ids[-1], line_string(segment_coordinates)))
            if reverse:
                edge_number += 1
                edges.append(_edge_row(road, edge_number, segment_ids[-1], segment_ids[0], line_string(list(reversed(segment_coordinates)))))
    return edges, junctions


def _point_from_coordinates(coordinates: tuple[float, float]) -> Any:
    # Imported lazily through the geometry object already held by road rows.
    from shapely.geometry import Point
    return Point(coordinates)


def _edge_row(road: dict[str, Any], edge_number: int, source: str, target: str, geometry: Any) -> dict[str, Any]:
    return {
        "edge_id": f"edge-{edge_number}", "osm_way_id": road["osm_id"], "from_node": source, "to_node": target,
        "highway": road["highway"], "oneway": road["oneway"], "name": road["name"], "name_en": road["name_en"],
        "geometry": geometry,
    }


def _geodataframe(gpd: Any, rows: list[dict[str, Any]], crs: str) -> Any:
    if rows:
        cleaned = [{key: value for key, value in row.items() if key not in {"coordinates", "node_ids", "tags"}} for row in rows]
        return gpd.GeoDataFrame(cleaned, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _write_graph(nx: Any, junctions: Any, edges: Any, output_path: Path, city: str) -> None:
    graph = nx.MultiDiGraph(city=city, crs=str(junctions.crs))
    for row in junctions.itertuples(index=False):
        graph.add_node(str(row.osm_id), x=float(row.geometry.x), y=float(row.geometry.y), source_ref=f"node/{row.osm_id}")
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.from_node), str(row.to_node), key=str(row.edge_id), edge_id=str(row.edge_id),
            osm_way_id=str(row.osm_way_id), cost_m=float(row.cost_m), highway=str(row.highway),
        )
    nx.write_graphml(graph, output_path)
