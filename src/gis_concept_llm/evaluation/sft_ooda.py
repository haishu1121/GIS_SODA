"""Executable scoring for the production GIS Concept OODA ``messages`` data.

This module scores a model-generated final ``Act`` against the program-written
answer stored in a held-out SFT record. It never uses the reference completion
as a model prediction.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable

_OODA_HEADINGS = ("Observe", "Orient", "Decide", "Act")
_EDGE_RE = re.compile(r"^\s*(n\d+) -> (n\d+), cost=([0-9]+(?:\.[0-9]+)?) m\s*$", re.MULTILINE)
_SOURCE_RE = re.compile(r"^Query source:\s*(n\d+)\s*$", re.MULTILINE)
_TARGET_RE = re.compile(r"^Query target:\s*(n\d+)\s*$", re.MULTILINE)


def task_from_scenario_id(scenario_id: str) -> str:
    """Return the stable task family encoded by a production scenario ID."""
    for task in (
        "location_direction",
        "location_distance",
        "object_topology",
        "network_connectivity",
        "network_shortest_path",
    ):
        if f"-{task}-" in scenario_id:
            return task
    raise ValueError(f"unknown GIS Concept scenario_id: {scenario_id!r}")


def extract_final_act(completion: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Parse the final ``Act: {JSON}`` stage without accepting trailing prose."""
    matches = list(re.finditer(r"(?im)^Act:\s*", completion))
    if not matches:
        return None, "missing Act heading", False
    remainder = completion[matches[-1].end():].lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(remainder)
    except json.JSONDecodeError as exc:
        return None, f"invalid Act JSON: {exc.msg}", False
    if not isinstance(value, dict):
        return None, "Act JSON must be an object", False
    is_final = not remainder[end:].strip()
    if not is_final:
        return value, "non-whitespace text follows Act JSON", False
    return value, None, True


def has_valid_ooda_schema(completion: str) -> bool:
    """Require one ordered set of visible OODA headings and a final JSON Act."""
    positions: list[int] = []
    for heading in _OODA_HEADINGS:
        found = list(re.finditer(rf"(?im)^{heading}:\s*", completion))
        if len(found) != 1:
            return False
        positions.append(found[0].start())
    if positions != sorted(positions):
        return False
    _, _, is_final = extract_final_act(completion)
    return is_final


def _answers_equal(predicted: Any, expected: Any, *, float_tolerance: float = 1e-3) -> bool:
    """Compare JSON answers while allowing only harmless decimal rendering drift."""
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return type(predicted) is type(expected) and predicted == expected
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return math.isclose(float(predicted), float(expected), abs_tol=float_tolerance, rel_tol=0.0)
    if isinstance(expected, str) or isinstance(predicted, str):
        return type(predicted) is type(expected) and predicted == expected
    if isinstance(expected, list) and isinstance(predicted, list):
        return len(predicted) == len(expected) and all(
            _answers_equal(left, right, float_tolerance=float_tolerance)
            for left, right in zip(predicted, expected)
        )
    if isinstance(expected, dict) and isinstance(predicted, dict):
        return set(predicted) == set(expected) and all(
            _answers_equal(predicted[key], expected[key], float_tolerance=float_tolerance)
            for key in expected
        )
    return False


def _parse_network_scene(question: str) -> tuple[dict[str, dict[str, float]], str, str] | None:
    edges: dict[str, dict[str, float]] = defaultdict(dict)
    for source, target, cost in _EDGE_RE.findall(question):
        # A real road subgraph can contain parallel directed links. The SFT
        # scene exposes only node IDs (not edge IDs), so a node-only model path
        # denotes the cheapest displayed link for that ordered node pair.
        parsed_cost = float(cost)
        edges[source][target] = min(edges[source].get(target, math.inf), parsed_cost)
    source_match, target_match = _SOURCE_RE.search(question), _TARGET_RE.search(question)
    if not edges or source_match is None or target_match is None:
        return None
    return dict(edges), source_match.group(1), target_match.group(1)


def _reachable(edges: dict[str, dict[str, float]], source: str, target: str) -> bool:
    pending, visited = [source], {source}
    while pending:
        current = pending.pop()
        if current == target:
            return True
        for neighbour in edges.get(current, {}):
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    return False


def _shortest_cost(edges: dict[str, dict[str, float]], source: str, target: str) -> float | None:
    queue: list[tuple[float, str]] = [(0.0, source)]
    distances = {source: 0.0}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        if current == target:
            return distance
        for neighbour, cost in edges.get(current, {}).items():
            candidate = distance + cost
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return None


def _path_cost(edges: dict[str, dict[str, float]], path: Any, source: str, target: str) -> float | None:
    if not isinstance(path, list) or not all(isinstance(node, str) for node in path):
        return None
    if len(path) < 2 or path[0] != source or path[-1] != target:
        return None
    total = 0.0
    for left, right in zip(path, path[1:]):
        if right not in edges.get(left, {}):
            return None
        total += edges[left][right]
    return total


def validate_network_gold(record: dict[str, Any]) -> bool | None:
    """Independently recheck network gold against the immutable rendered graph."""
    task = task_from_scenario_id(record["scenario_id"])
    if task not in {"network_connectivity", "network_shortest_path"}:
        return None
    scene = _parse_network_scene(record["messages"][0]["content"])
    if scene is None:
        return False
    edges, source, target = scene
    gold = record["acts"]["program"]
    if task == "network_connectivity":
        return gold == {"connected": _reachable(edges, source, target)}
    optimal = _shortest_cost(edges, source, target)
    path_total = _path_cost(edges, gold.get("path"), source, target)
    return (
        optimal is not None
        and path_total is not None
        and math.isclose(path_total, float(gold.get("optimal_cost_m")), abs_tol=1e-3)
        and math.isclose(optimal, float(gold.get("optimal_cost_m")), abs_tol=1e-3)
    )


def score_completion(record: dict[str, Any], completion: str) -> dict[str, Any]:
    """Score one generated completion and retain diagnostics for error review."""
    scenario_id = record["scenario_id"]
    task = task_from_scenario_id(scenario_id)
    gold = record["acts"]["program"]
    predicted, parse_error, act_is_final = extract_final_act(completion)
    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "example_id": record.get("example_id"),
        "task": task,
        "gold_act": gold,
        "predicted_act": predicted,
        "act_json_valid": predicted is not None and act_is_final,
        "act_parse_error": parse_error,
        "ooda_schema_valid": has_valid_ooda_schema(completion),
        "act_exact": predicted is not None and act_is_final and _answers_equal(predicted, gold),
        "gold_network_scene_valid": validate_network_gold(record),
    }
    if task == "network_shortest_path":
        scene = _parse_network_scene(record["messages"][0]["content"])
        path_cost = None
        if scene is not None and isinstance(predicted, dict):
            edges, source, target = scene
            path_cost = _path_cost(edges, predicted.get("path"), source, target)
        reported_cost = predicted.get("optimal_cost_m") if isinstance(predicted, dict) else None
        result["path_is_legal"] = path_cost is not None
        result["reported_cost_matches_path"] = (
            path_cost is not None
            and isinstance(reported_cost, (int, float))
            and not isinstance(reported_cost, bool)
            and math.isclose(float(reported_cost), path_cost, abs_tol=1e-3)
        )
        result["path_is_optimal"] = (
            path_cost is not None
            and math.isclose(path_cost, float(gold["optimal_cost_m"]), abs_tol=1e-3)
        )
    return result


def summarize_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build task-balanced and overall metrics from per-scenario score rows."""
    values = list(scores)
    if not values:
        raise ValueError("cannot summarize an empty evaluation")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        def rate(field: str) -> float:
            return round(sum(row.get(field) is True for row in rows) / len(rows), 6)

        payload: dict[str, Any] = {
            "count": len(rows),
            "act_exact_accuracy": rate("act_exact"),
            "act_json_valid_rate": rate("act_json_valid"),
            "ooda_schema_valid_rate": rate("ooda_schema_valid"),
        }
        path_rows = [row for row in rows if row["task"] == "network_shortest_path"]
        if path_rows:
            payload["shortest_path"] = {
                "count": len(path_rows),
                "path_legal_rate": round(sum(row.get("path_is_legal") is True for row in path_rows) / len(path_rows), 6),
                "reported_cost_matches_path_rate": round(sum(row.get("reported_cost_matches_path") is True for row in path_rows) / len(path_rows), 6),
                "path_optimal_rate": round(sum(row.get("path_is_optimal") is True for row in path_rows) / len(path_rows), 6),
            }
        return payload

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        grouped[row["task"]].append(row)
    per_task = {task: aggregate(rows) for task, rows in sorted(grouped.items())}
    macro_accuracy = round(sum(item["act_exact_accuracy"] for item in per_task.values()) / len(per_task), 6)
    return {"overall": aggregate(values), "macro_act_exact_accuracy": macro_accuracy, "by_task": per_task}
