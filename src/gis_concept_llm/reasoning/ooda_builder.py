"""OODA is an outer scaffold; the minimal GIS rule is the semantic core."""

from __future__ import annotations

from ..schema import OODATrace
from .minimal_trace_builder import build_minimal_trace


def build_ooda_trace(task_type: str, metadata: dict) -> OODATrace:
    answer = metadata["answer"]
    minimal = build_minimal_trace(task_type, metadata)
    if task_type == "direction":
        observe = f"Coordinates are A={tuple(metadata['a'])}, B={tuple(metadata['b'])}."
        orient = "This is a relative-direction task; compare coordinate differences."
    elif task_type == "topology":
        geometry_kind = "geographic geometries" if metadata.get("geometry_format") == "geojson" else "bounding boxes"
        observe = f"Two geometries A and B are given as {geometry_kind}."
        orient = "Classify their interior and boundary relationship."
    elif task_type == "connectivity":
        observe = f"Graph edges are {metadata['edges']}; query nodes are {metadata['source']} and {metadata['target']}."
        orient = "Search the graph for any legal path, independently of path optimality."
    else:
        raise ValueError(f"no OODA builder for {task_type}")
    return OODATrace(observe, orient, minimal, answer)
