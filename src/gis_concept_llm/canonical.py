"""Build and verify language-free canonical GIS concept scenarios."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import zlib
from typing import Any, Iterable, Mapping


TOPOLOGY_LABELS = frozenset({"Contains", "Within", "Touches", "Overlaps", "Intersects", "Disjoint", "Equals"})


@dataclass(frozen=True)
class CanonicalBuildConfig:
    city: str
    normalized_dir: Path
    output_dir: Path
    per_task: int = 500
    seed: int = 20260920
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    max_graph_nodes: int = 14
    max_graph_edges: int = 28
    max_geometry_vertices: int = 160
    analysis_crs: str = ""
    overwrite: bool = False

    def validate(self) -> None:
        if self.per_task < 1:
            raise ValueError("per_task must be positive")
        if not 0 < self.train_ratio < 1 or not 0 < self.validation_ratio < 1:
            raise ValueError("split ratios must be between zero and one")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be less than one")
        if self.max_graph_nodes < 3 or self.max_graph_edges < 2:
            raise ValueError("graph limits are too small for network tasks")
        if self.max_geometry_vertices < 4:
            raise ValueError("max_geometry_vertices must be at least four")


def direction_8(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        raise ValueError("direction is undefined for identical coordinates")
    angle = math.degrees(math.atan2(dy, dx)) % 360
    labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    return labels[int((angle + 22.5) // 45) % 8]


def topology_relation(geometry_a: Any, geometry_b: Any) -> str:
    """Return one mutually exclusive relation for two polygonal geometries."""

    if geometry_a.equals(geometry_b):
        return "Equals"
    if geometry_a.contains(geometry_b):
        return "Contains"
    if geometry_a.within(geometry_b):
        return "Within"
    if geometry_a.touches(geometry_b):
        return "Touches"
    if geometry_a.overlaps(geometry_b):
        return "Overlaps"
    if geometry_a.disjoint(geometry_b):
        return "Disjoint"
    return "Intersects"


def validate_canonical(record: Mapping[str, Any]) -> None:
    """Independently recompute one canonical answer from its stored scene."""

    required = {"example_id", "scenario_id", "task", "provenance", "scene", "gold"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"canonical record is missing fields: {sorted(missing)}")
    task = record["task"]
    scene = record["scene"]
    answer = record["gold"]["answer"]
    skill = task["skill"]
    if skill == "relative_direction":
        a, b = scene["points"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if answer != {"direction": direction_8(dx, dy)}:
            raise ValueError("direction gold answer does not match the point coordinates")
    elif skill == "euclidean_distance":
        a, b = scene["points"]
        expected = round(math.hypot(b["x"] - a["x"], b["y"] - a["y"]), 3)
        if answer != {"distance_m": expected}:
            raise ValueError("distance gold answer does not match the point coordinates")
    elif skill == "topology_relation":
        from shapely.geometry import shape

        relation = topology_relation(shape(scene["geometry_a"]), shape(scene["geometry_b"]))
        if answer != {"relation": relation}:
            raise ValueError("topology gold answer does not match stored GeoJSON")
    elif skill in {"connectivity", "shortest_path"}:
        graph = _scene_graph(scene)
        source, target = scene["query"]["source"], scene["query"]["target"]
        if skill == "connectivity":
            import networkx as nx

            expected = nx.has_path(graph, source, target)
            if answer != {"connected": expected}:
                raise ValueError("connectivity gold answer does not match the stored graph")
        else:
            import networkx as nx

            expected_cost = round(nx.shortest_path_length(graph, source, target, weight="cost_m"), 3)
            path = answer.get("path")
            if answer.get("optimal_cost_m") != expected_cost or not _is_valid_weighted_path(graph, path, expected_cost):
                raise ValueError("shortest-path gold answer does not match the stored graph")
            if "runner_up_path" in record["gold"]["witness"]:
                expected_witness = _shortest_path_witness(graph, path)
                witness = record["gold"]["witness"]
                for key in ("runner_up_path", "runner_up_cost_m", "candidate_paths_at_least"):
                    if witness.get(key) != expected_witness[key]:
                        raise ValueError(f"shortest-path witness field does not match the program solver: {key}")
    else:
        raise ValueError(f"unknown canonical skill: {skill}")


def build_canonical_dataset(config: CanonicalBuildConfig) -> dict[str, int]:
    """Create Beijing-style canonical records and leakage-resistant splits."""

    config.validate()
    paths = _dataset_paths(config)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("canonical output exists; pass overwrite=True to replace generated files")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for path in existing:
            path.unlink()

    gpd, nx = _dependencies()
    gpkg_path = config.normalized_dir / f"{config.city}_map.gpkg"
    graph_path = config.normalized_dir / "network.graphml"
    map_manifest_path = config.normalized_dir / "map_manifest.json"
    if not gpkg_path.is_file() or not graph_path.is_file() or not map_manifest_path.is_file():
        raise FileNotFoundError("normalized directory must contain map.gpkg, network.graphml, and map_manifest.json")
    map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    config = replace(config, analysis_crs=map_manifest["analysis_crs"])
    rng = random.Random(config.seed)
    pois = gpd.read_file(gpkg_path, layer="pois")
    aois = gpd.read_file(gpkg_path, layer="aois")
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    graph = nx.MultiDiGraph(graph)

    records: dict[str, list[dict[str, Any]]] = {
        "location_direction": _sample_location_records(
            pois, aois, config, rng, skill="relative_direction", file_key="location_direction"
        ),
        "location_distance": _sample_location_records(
            pois, aois, config, rng, skill="euclidean_distance", file_key="location_distance"
        ),
        "object_topology": _sample_topology_records(aois, config, rng),
        "network_connectivity": _sample_network_records(graph, config, rng, skill="connectivity"),
        "network_shortest_path": _sample_network_records(graph, config, rng, skill="shortest_path"),
    }
    all_records = [record for task_records in records.values() for record in task_records]
    splits = _assign_splits(all_records, config)
    for task_name, task_records in records.items():
        _write_jsonl(paths[task_name], task_records)
    _write_jsonl(paths["splits"], splits)
    counts = {name: len(task_records) for name, task_records in records.items()}
    paths["manifest"].write_text(json.dumps({
        "schema_version": "gis-concept-canonical/v1",
        "city": config.city,
        "normalized_map": str(gpkg_path),
        "network_graph": str(graph_path),
        "source_map_manifest": str(map_manifest_path),
        "analysis_crs": map_manifest["analysis_crs"],
        "study_bbox_wgs84": map_manifest["study_bbox_wgs84"],
        "seed": config.seed,
        "records": counts,
        "split_policy": {
            "unit": "scenario_id",
            "group": "spatial_block_id",
            "train_ratio": config.train_ratio,
            "validation_ratio": config.validation_ratio,
            "test_ratio": round(1 - config.train_ratio - config.validation_ratio, 6),
        },
        "invariants": [
            "canonical records contain no model-facing prompt or reasoning text",
            "all gold answers and witnesses are generated and verified by program logic",
            "SFT views must inherit the canonical scenario split",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


def _sample_location_records(points: Any, aois: Any, config: CanonicalBuildConfig, rng: random.Random, *, skill: str, file_key: str) -> list[dict[str, Any]]:
    locations = _location_pool(points, aois)
    if len(locations) < 2:
        raise ValueError("at least two POIs/AOIs are required for location tasks")
    records: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(records) < config.per_task and attempts < config.per_task * 50:
        attempts += 1
        first, second = rng.sample(locations, 2)
        pair = (first["source_ref"], second["source_ref"])
        if pair in used_pairs:
            continue
        first_x, first_y = round(first["x"], 3), round(first["y"], 3)
        second_x, second_y = round(second["x"], 3), round(second["y"], 3)
        dx, dy = second_x - first_x, second_y - first_y
        distance = math.hypot(dx, dy)
        if distance < 25 or distance > 15_000:
            continue
        index = len(records) + 1
        task = {"concept": "Location", "skill": skill, "coordinate_crs": str(points.crs), "coordinate_unit": "m"}
        scene = {
            "representation": "projected_point_pair",
            "points": [
                {"id": "p1", "x": first_x, "y": first_y},
                {"id": "p2", "x": second_x, "y": second_y},
            ],
            "query": {"reference": "p1", "target": "p2"},
        }
        if skill == "relative_direction":
            answer = {"direction": direction_8(dx, dy)}
            witness = {"delta_x_m": round(dx, 3), "delta_y_m": round(dy, 3)}
        else:
            answer = {"distance_m": round(distance, 3)}
            witness = {"metric": "euclidean", "delta_x_m": round(dx, 3), "delta_y_m": round(dy, 3)}
        record = _record(
            config, task_name=file_key, index=index, task=task,
            provenance=_provenance(config, [first["source_ref"], second["source_ref"]]),
            scene=scene, answer=answer, witness=witness, spatial_anchor=((first["x"] + second["x"]) / 2, (first["y"] + second["y"]) / 2),
        )
        validate_canonical(record)
        records.append(record)
        used_pairs.add(pair)
    _require_count(file_key, records, config.per_task)
    return records


def _sample_topology_records(aois: Any, config: CanonicalBuildConfig, rng: random.Random) -> list[dict[str, Any]]:
    from shapely.geometry import mapping

    candidates = [
        row for row in aois.itertuples(index=False)
        if not row.geometry.is_empty and row.geometry.is_valid and _vertex_count(row.geometry) <= config.max_geometry_vertices
    ]
    if len(candidates) < 2:
        raise ValueError("not enough valid, bounded-complexity AOIs for topology tasks")
    spatial_index = aois.sindex
    buckets: dict[str, list[tuple[Any, Any]]] = {label: [] for label in TOPOLOGY_LABELS}
    ordered = candidates[:]
    rng.shuffle(ordered)
    for first in ordered[: min(len(ordered), 12_000)]:
        nearby_indices = list(spatial_index.query(first.geometry, predicate="intersects"))
        rng.shuffle(nearby_indices)
        for nearby_index in nearby_indices[:20]:
            second = aois.iloc[int(nearby_index)]
            if str(first.osm_id) == str(second.osm_id) or second.geometry.is_empty or not second.geometry.is_valid:
                continue
            if _vertex_count(second.geometry) > config.max_geometry_vertices:
                continue
            relation = topology_relation(first.geometry, second.geometry)
            if relation in buckets and len(buckets[relation]) < config.per_task * 2:
                buckets[relation].append((first, second))
    # Disjoint examples need no spatial-index intersection and are deliberately sampled far apart.
    for _ in range(config.per_task * 8):
        first, second = rng.sample(candidates, 2)
        if first.geometry.disjoint(second.geometry):
            buckets["Disjoint"].append((first, second))
            if len(buckets["Disjoint"]) >= config.per_task * 2:
                break

    usable_labels = [label for label, pairs in buckets.items() if pairs]
    if not usable_labels:
        raise ValueError("no topology relations could be sampled from AOIs")
    records: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    label_cursor = 0
    while len(records) < config.per_task:
        label = usable_labels[label_cursor % len(usable_labels)]
        label_cursor += 1
        pairs = buckets[label]
        if not pairs:
            usable_labels.remove(label)
            if not usable_labels:
                break
            continue
        first, second = pairs.pop()
        pair = (str(first.osm_id), str(second.osm_id))
        if pair in used_pairs:
            continue
        relation = topology_relation(first.geometry, second.geometry)
        scene = {
            "representation": "geojson_geometry_pair",
            "geometry_crs": config.analysis_crs,
            "geometry_a": mapping(first.geometry),
            "geometry_b": mapping(second.geometry),
            "query": {"subject": "a", "reference": "b"},
        }
        record = _record(
            config, task_name="object_topology", index=len(records) + 1,
            task={"concept": "Object", "skill": "topology_relation", "relation_set": sorted(TOPOLOGY_LABELS)},
            provenance=_provenance(config, [f"area/{first.osm_id}", f"area/{second.osm_id}"]),
            scene=scene, answer={"relation": relation}, witness={"de9im": first.geometry.relate(second.geometry)},
            spatial_anchor=((first.geometry.representative_point().x + second.geometry.representative_point().x) / 2,
                            (first.geometry.representative_point().y + second.geometry.representative_point().y) / 2),
        )
        validate_canonical(record)
        records.append(record)
        used_pairs.add(pair)
    _require_count("object_topology", records, config.per_task)
    return records


def _sample_network_records(graph: Any, config: CanonicalBuildConfig, rng: random.Random, *, skill: str) -> list[dict[str, Any]]:
    import networkx as nx

    records: list[dict[str, Any]] = []
    nodes = list(graph.nodes)
    used_queries: set[tuple[str, str]] = set()
    attempts = 0
    while len(records) < config.per_task and attempts < config.per_task * 300:
        attempts += 1
        source = rng.choice(nodes)
        local = _local_subgraph(graph, source, config, rng)
        if local is None:
            continue
        if skill == "connectivity":
            query = _connectivity_query(local, source, rng, want_positive=len(records) % 2 == 0)
        else:
            query = _shortest_path_query(local, source, rng)
        if query is None:
            continue
        target, path = query
        source_ref, target_ref = str(source), str(target)
        if (source_ref, target_ref) in used_queries:
            continue
        scene, node_ref_map = _network_scene(local, source, target)
        if skill == "connectivity":
            connected = nx.has_path(_scene_graph(scene), "n1", "n2")
            witness = {"path": _alias_path(path, node_ref_map) if connected and path else None}
            answer = {"connected": connected}
            task_name = "network_connectivity"
        else:
            solver_graph = _scene_graph(scene)
            optimal_cost = round(nx.shortest_path_length(solver_graph, "n1", "n2", weight="cost_m"), 3)
            local_path = nx.shortest_path(solver_graph, "n1", "n2", weight="cost_m")
            answer = {"optimal_cost_m": optimal_cost, "path": local_path}
            witness = _shortest_path_witness(solver_graph, local_path)
            task_name = "network_shortest_path"
        source_point = graph.nodes[source]
        record = _record(
            config, task_name=task_name, index=len(records) + 1,
            task={"concept": "Network", "skill": skill, "directed": True, "cost_field": "cost_m"},
            provenance=_provenance(config, [f"node/{node}" for node in local.nodes] + [f"way/{data['osm_way_id']}" for _, _, _, data in local.edges(keys=True, data=True)]),
            scene=scene, answer=answer, witness=witness,
            spatial_anchor=(float(source_point["x"]), float(source_point["y"])),
        )
        validate_canonical(record)
        records.append(record)
        used_queries.add((source_ref, target_ref))
    _require_count(f"network_{skill}", records, config.per_task)
    return records


def _local_subgraph(graph: Any, source: str, config: CanonicalBuildConfig, rng: random.Random) -> Any | None:
    import networkx as nx

    undirected = graph.to_undirected(as_view=True)
    nearby = list(nx.single_source_shortest_path_length(undirected, source, cutoff=3))
    if len(nearby) < 4:
        return None
    rng.shuffle(nearby)
    chosen = [source] + [node for node in nearby if node != source][:config.max_graph_nodes - 1]
    subgraph = graph.subgraph(chosen).copy()
    if subgraph.number_of_edges() < 3 or subgraph.number_of_edges() > config.max_graph_edges:
        return None
    return subgraph


def _connectivity_query(graph: Any, source: str, rng: random.Random, *, want_positive: bool) -> tuple[str, list[str] | None] | None:
    import networkx as nx

    reachable = nx.single_source_shortest_path(graph, source)
    if want_positive:
        options = [node for node, path in reachable.items() if node != source and 2 <= len(path) <= 5]
        if not options:
            return None
        target = rng.choice(options)
        return target, reachable[target]
    weak_component = nx.node_connected_component(graph.to_undirected(), source)
    options = [node for node in weak_component if node != source and node not in reachable]
    if not options:
        return None
    return rng.choice(options), None


def _shortest_path_query(graph: Any, source: str, rng: random.Random) -> tuple[str, list[str]] | None:
    import networkx as nx

    simple = _minimum_cost_digraph(graph)
    if source not in simple:
        return None
    options = list(simple.nodes)
    rng.shuffle(options)
    for target in options:
        if target == source or not nx.has_path(simple, source, target):
            continue
        try:
            paths = list(_first_simple_paths(simple, source, target, limit=2))
        except nx.NetworkXNoPath:
            continue
        if len(paths) < 2:
            continue
        optimal = list(nx.all_shortest_paths(simple, source, target, weight="cost_m"))
        if len(optimal) != 1:
            continue
        if len(optimal[0]) < 3:
            continue
        return target, optimal[0]
    return None


def _first_simple_paths(graph: Any, source: str, target: str, limit: int) -> Iterable[list[str]]:
    import networkx as nx

    iterator = nx.shortest_simple_paths(graph, source, target, weight="cost_m")
    for _, path in zip(range(limit), iterator):
        yield path


def _shortest_path_witness(graph: Any, optimal_path: list[str]) -> dict[str, Any]:
    """Compute the program-certified first and second ranked simple paths."""

    simple = _minimum_cost_digraph(graph)
    source, target = optimal_path[0], optimal_path[-1]
    ranked = list(_first_simple_paths(simple, source, target, limit=2))
    if len(ranked) < 2:
        raise ValueError("shortest-path scenario must have an optimal and runner-up simple path")
    if ranked[0] != optimal_path:
        raise ValueError("stored shortest path does not match the program-ranked first path")
    return {
        "selected_edge_ids": _path_edge_ids(graph, optimal_path),
        "candidate_paths_at_least": 2,
        "runner_up_path": ranked[1],
        "runner_up_cost_m": _simple_path_cost(simple, ranked[1]),
    }


def enrich_shortest_path_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add a solver-derived runner-up witness without changing scene or gold."""

    if record["task"]["skill"] != "shortest_path":
        raise ValueError("only shortest_path canonical records can be enriched")
    enriched = json.loads(json.dumps(record))
    graph = _scene_graph(enriched["scene"])
    enriched["gold"]["witness"] = _shortest_path_witness(graph, enriched["gold"]["answer"]["path"])
    validate_canonical(enriched)
    return enriched


def _minimum_cost_digraph(graph: Any) -> Any:
    import networkx as nx

    simple = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        simple.add_node(node, **data)
    for source, target, data in graph.edges(data=True):
        cost = float(data["cost_m"])
        if not simple.has_edge(source, target) or cost < simple[source][target]["cost_m"]:
            simple.add_edge(
                source, target, cost_m=cost, edge_id=str(data["edge_id"]),
                osm_way_id=str(data.get("osm_way_id", data.get("source_ref", ""))),
            )
    return simple


def _simple_path_cost(graph: Any, path: list[str]) -> float:
    return round(sum(float(graph[source][target]["cost_m"]) for source, target in zip(path, path[1:])), 3)


def _network_scene(graph: Any, source: str, target: str) -> tuple[dict[str, Any], dict[str, str]]:
    remaining = sorted(node for node in graph.nodes if node not in {source, target})
    ordered = [source, target, *remaining]
    aliases = {node: f"n{index + 1}" for index, node in enumerate(ordered)}
    nodes = [{"id": aliases[node]} for node in ordered]
    edges = [
        {
            "id": str(data["edge_id"]), "from": aliases[left], "to": aliases[right],
            "cost_m": round(float(data["cost_m"]), 3), "source_ref": f"way/{data['osm_way_id']}",
        }
        for left, right, _, data in graph.edges(keys=True, data=True)
    ]
    return {
        "representation": "directed_weighted_graph",
        "nodes": nodes,
        "edges": edges,
        "query": {"source": "n1", "target": "n2"},
    }, aliases


def _scene_graph(scene: Mapping[str, Any]) -> Any:
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node["id"] for node in scene["nodes"])
    for edge in scene["edges"]:
        graph.add_edge(edge["from"], edge["to"], key=edge["id"], cost_m=float(edge["cost_m"]), edge_id=edge["id"])
    return graph


def _is_valid_weighted_path(graph: Any, path: list[str] | None, expected_cost: float) -> bool:
    if not path or len(path) < 2:
        return False
    cost = 0.0
    for source, target in zip(path, path[1:]):
        candidates = graph.get_edge_data(source, target)
        if not candidates:
            return False
        cost += min(float(data["cost_m"]) for data in candidates.values())
    return round(cost, 3) == expected_cost


def _path_edge_ids(graph: Any, path: list[str]) -> list[str]:
    result = []
    for source, target in zip(path, path[1:]):
        candidates = graph.get_edge_data(source, target)
        _, data = min(candidates.items(), key=lambda item: float(item[1]["cost_m"]))
        result.append(str(data["edge_id"]))
    return result


def _alias_path(path: list[str] | None, aliases: Mapping[str, str]) -> list[str] | None:
    return [aliases[node] for node in path] if path else None


def _location_pool(pois: Any, aois: Any) -> list[dict[str, Any]]:
    result = [
        {"source_ref": f"node/{row.osm_id}", "x": float(row.geometry.x), "y": float(row.geometry.y)}
        for row in pois.itertuples(index=False) if not row.geometry.is_empty
    ]
    result.extend(
        {"source_ref": f"area/{row.osm_id}", "x": float(row.geometry.representative_point().x), "y": float(row.geometry.representative_point().y)}
        for row in aois.itertuples(index=False) if not row.geometry.is_empty and row.geometry.is_valid
    )
    return result


def _vertex_count(geometry: Any) -> int:
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) + sum(len(ring.coords) for ring in geometry.interiors)
    if geometry.geom_type == "MultiPolygon":
        return sum(_vertex_count(part) for part in geometry.geoms)
    return 0


def _record(
    config: CanonicalBuildConfig, *, task_name: str, index: int, task: dict[str, Any], provenance: dict[str, Any],
    scene: dict[str, Any], answer: dict[str, Any], witness: dict[str, Any], spatial_anchor: tuple[float, float],
) -> dict[str, Any]:
    scenario_id = f"{config.city}-{task_name}-{index:06d}"
    block_size_m = 2_000
    block = f"{int(math.floor(spatial_anchor[0] / block_size_m))}:{int(math.floor(spatial_anchor[1] / block_size_m))}"
    return {
        "schema_version": "gis-concept-canonical/v1",
        "example_id": f"osm-{scenario_id}",
        "scenario_id": scenario_id,
        "task": task,
        "provenance": provenance,
        "scene": scene,
        "gold": {"answer": answer, "witness": witness},
        "split_group": {"spatial_block_id": block, "block_size_m": block_size_m},
    }


def _provenance(config: CanonicalBuildConfig, entity_refs: list[str]) -> dict[str, Any]:
    return {
        "city": config.city,
        "map_ref": str(config.normalized_dir / f"{config.city}_map.gpkg"),
        "entity_refs": sorted(set(entity_refs)),
        "analysis_crs": config.analysis_crs,
    }


def _assign_splits(records: list[dict[str, Any]], config: CanonicalBuildConfig) -> list[dict[str, Any]]:
    groups = {record["split_group"]["spatial_block_id"] for record in records}
    assignment: dict[str, str] = {}
    for group in groups:
        value = (zlib.crc32(f"{config.seed}:{group}".encode("utf-8")) & 0xFFFFFFFF) / 2**32
        assignment[group] = "train" if value < config.train_ratio else "validation" if value < config.train_ratio + config.validation_ratio else "test"
    return [
        {
            "scenario_id": record["scenario_id"], "split": assignment[record["split_group"]["spatial_block_id"]],
            "reason": "in_city_spatial_block", "spatial_block_id": record["split_group"]["spatial_block_id"],
        }
        for record in records
    ]


def _dataset_paths(config: CanonicalBuildConfig) -> dict[str, Path]:
    records_dir = config.output_dir / "records" / config.city
    return {
        "location_direction": records_dir / "location_direction.jsonl",
        "location_distance": records_dir / "location_distance.jsonl",
        "object_topology": records_dir / "object_topology.jsonl",
        "network_connectivity": records_dir / "network_connectivity.jsonl",
        "network_shortest_path": records_dir / "network_shortest_path.jsonl",
        "splits": config.output_dir / "splits" / "scenario_split.jsonl",
        "manifest": config.output_dir / "dataset_manifest.json",
    }


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _require_count(task_name: str, records: list[dict[str, Any]], expected: int) -> None:
    if len(records) != expected:
        raise RuntimeError(f"{task_name}: generated {len(records)} records, expected {expected}")


def _dependencies() -> tuple[Any, Any]:
    try:
        import geopandas as gpd
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("canonical construction requires GIS dependencies; install with python -m pip install -e .[gis]") from exc
    return gpd, nx
