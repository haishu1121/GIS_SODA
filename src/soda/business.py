"""Business-facing adapter contracts for text-only spatial decision systems.

The model never receives raw internal objects.  A domain adapter converts a
business state to text, validates a proposed structured action, and delegates
execution to the host system.  This is the production boundary around SODA.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .rewards import extract_answer


class ScenarioAdapter(Protocol):
    """Implement this protocol to connect warehouse, GIS, robot, or other domains."""

    def encode_state(self, state: Mapping[str, Any]) -> str: ...
    def validate_action(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> list[str]: ...
    def apply_action(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def is_complete(self, state: Mapping[str, Any]) -> bool: ...


@dataclass(frozen=True)
class DomainSpec:
    """Declarative common case.  Custom domains can implement `ScenarioAdapter` directly."""

    name: str
    state_description: str
    actions: dict[str, tuple[str, ...]]
    bounds: tuple[int, int] | None = None
    blocked_key: str = "obstacles"
    goal_key: str = "goal"

    @classmethod
    def from_json(cls, path: str) -> "DomainSpec":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        actions = {name: tuple(value["required_fields"]) for name, value in raw["actions"].items()}
        bounds = tuple(raw["bounds"]) if raw.get("bounds") else None
        return cls(raw["name"], raw.get("state_description", ""), actions, bounds, raw.get("blocked_key", "obstacles"), raw.get("goal_key", "goal"))


@dataclass
class DeclarativeSpatialAdapter:
    """Safe default adapter for a coordinate/grid business state.

    State convention: `entities` maps stable IDs to dictionaries, position is
    `[row, column]`, and blocked cells use `obstacles`.  It is intentionally
    small: domain-specific topologies, capacities, schedules, and permissions
    belong in an overriding `validate_action` implementation.
    """

    spec: DomainSpec
    executor: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None

    def encode_state(self, state: Mapping[str, Any]) -> str:
        entities = state.get("entities", {})
        entity_lines = [f"- {entity_id}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}" for entity_id, value in sorted(entities.items())]
        return "\n".join([
            f"Domain: {self.spec.name}", self.spec.state_description,
            "Current state:", *entity_lines,
            f"Blocked locations: {json.dumps(state.get(self.spec.blocked_key, []), ensure_ascii=False)}",
            f"Goal: {json.dumps(state.get(self.spec.goal_key), ensure_ascii=False)}",
            "Allowed operations: " + "; ".join(f"{name}({', '.join(fields)})" for name, fields in self.spec.actions.items()),
            "Return exactly one JSON action inside the Act stage.",
        ])

    def validate_action(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        operation = action.get("operation")
        if operation not in self.spec.actions:
            return [f"unknown operation: {operation!r}"]
        for key in self.spec.actions[str(operation)]:
            if key not in action:
                errors.append(f"missing required field: {key}")
        entity_id = action.get("entity_id")
        if entity_id is not None and entity_id not in state.get("entities", {}):
            errors.append(f"unknown entity_id: {entity_id!r}")
        target = action.get("target")
        if target is not None:
            if not isinstance(target, list) or len(target) != 2 or not all(isinstance(v, int) for v in target):
                errors.append("target must be [row, column] integer coordinates")
            elif self.spec.bounds and not (0 <= target[0] < self.spec.bounds[0] and 0 <= target[1] < self.spec.bounds[1]):
                errors.append("target is outside configured bounds")
            elif target in state.get(self.spec.blocked_key, []):
                errors.append("target is blocked")
        return errors

    def apply_action(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.executor is None:
            raise RuntimeError("No executor configured. Connect your business API in adapter.apply_action().")
        return self.executor(state, action)

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        return bool(state.get("complete", False))


def extract_json_action(model_text: str) -> dict[str, Any]:
    """Extract one JSON object from `Act:` and reject surrounding ambiguity."""
    act = extract_answer(model_text)
    match = re.search(r"\{.*\}", act, flags=re.DOTALL)
    if not match:
        raise ValueError("Act stage does not contain a JSON object")
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Act JSON is invalid") from exc
    if not isinstance(result, dict):
        raise ValueError("Act JSON must be an object")
    return result


@dataclass
class Decision:
    state_text: str
    model_text: str
    action: dict[str, Any]
    violations: list[str] = field(default_factory=list)


def one_decision(adapter: ScenarioAdapter, state: Mapping[str, Any], generate: Callable[[str], str]) -> Decision:
    """One OODA decision, with validation before any state-changing operation."""
    state_text = adapter.encode_state(state)
    model_text = generate(state_text)
    action = extract_json_action(model_text)
    return Decision(state_text, model_text, action, adapter.validate_action(state, action))


def closed_loop(adapter: ScenarioAdapter, initial_state: Mapping[str, Any], generate: Callable[[str], str], max_steps: int = 50) -> list[Decision]:
    """Observe fresh business state after each valid action; never execute invalid output."""
    state: Mapping[str, Any] = initial_state
    decisions: list[Decision] = []
    for _ in range(max_steps):
        if adapter.is_complete(state):
            return decisions
        decision = one_decision(adapter, state, generate)
        decisions.append(decision)
        if decision.violations:
            raise ValueError("unsafe model action: " + "; ".join(decision.violations))
        state = adapter.apply_action(state, decision.action)
    raise RuntimeError("business loop exceeded max_steps")
