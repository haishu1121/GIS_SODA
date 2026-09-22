from __future__ import annotations

from ..generators.connectivity_generator import witness_path
from ..schema import ConceptExample
from .base import Verification


class ConnectivityVerifier:
    def verify(self, example: ConceptExample, predicted: str) -> Verification:
        if example.task_type != "connectivity":
            raise ValueError("ConnectivityVerifier only accepts connectivity examples")
        edges = [tuple(edge) for edge in example.metadata["edges"]]
        path = witness_path(edges, example.metadata["source"], example.metadata["target"])
        expected = "Connected" if path else "Disconnected"
        output = predicted.strip().title()
        correct = output == expected
        detail = "witness path: " + " -> ".join(path) if path else "no path in graph"
        return Verification(correct, 1.0 if correct else -1.0, expected, output, detail)
