"""Deterministic synthetic SPOD-style task generation with executable ground truth."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .control import DIRECTIONS, Point, add, astar, chebyshev, direction_between
from .schema import OODATrace, SPODExample

TASKS = {
    "memory_path": 1, "object_location_distance": 1, "relative_direction": 1, "shortest_path": 1,
    "packing_shapes": 2, "mental_rotation": 2, "multi_point_relation": 2,
    "double_point_relation": 2, "door_rotation": 2,
    "door_rotation_multiturn": 3, "spatial_relation_multiturn": 3,
    "coordinates_movement_multiturn": 3, "mental_rotation_multiturn": 3,
    "OP": 4, "FOP": 4, "MCF": 4, "MCO": 4,
}

COMPASS = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}


def _compass(dx: int, dy: int) -> str:
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    return next(name for name, value in COMPASS.items() if value == (sx, sy))


def _rotate(direction: str, turns: int) -> str:
    order = ["N", "E", "S", "W"]
    return order[(order.index(direction) + turns) % 4]


@dataclass
class SPODGenerator:
    seed: int = 2026

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.serial = 0

    def _example(self, task: str, question: str, answer: str, trace: OODATrace, **metadata) -> SPODExample:
        self.serial += 1
        return SPODExample(
            id=f"{task}-{self.serial:07d}", task=task, tier=TASKS[task], question=question,
            answer=answer, ooda=trace, metadata={"synthetic": True, **metadata},
        )

    def memory_path(self) -> SPODExample:
        start = (self.rng.randint(-3, 3), self.rng.randint(-3, 3))
        moves = [self.rng.choice(["N", "S", "E", "W"]) for _ in range(self.rng.randint(5, 10))]
        x, y = start
        for move in moves:
            dx, dy = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[move]
            x, y = x + dx, y + dy
        answer = f"({x}, {y})"
        return self._example("memory_path", f"Start at {start}. Walk {' '.join(moves)}. What is the final coordinate?", answer,
            OODATrace(f"Start={start}; movements={moves}.", f"Sum horizontal and vertical displacements.", f"Net displacement gives {answer}.", f"Output {answer}."), start=start, moves=moves)

    def object_location_distance(self) -> SPODExample:
        a = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        b = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        while a == b:
            b = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        answer = f"{distance:.2f}"
        return self._example("object_location_distance", f"Object A is at {a}; B is at {b}. Give Euclidean distance rounded to 2 decimals.", answer,
            OODATrace(f"A={a}, B={b}.", f"Delta=({b[0]-a[0]}, {b[1]-a[1]}).", f"Use sqrt(dx^2 + dy^2)={answer}.", f"Output {answer}."), a=a, b=b)

    def relative_direction(self) -> SPODExample:
        a = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        b = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        while a == b or a[0] == b[0] and a[1] == b[1]:
            b = (self.rng.randint(-9, 9), self.rng.randint(-9, 9))
        answer = _compass(b[0] - a[0], b[1] - a[1])
        return self._example("relative_direction", f"A is at {a}; B is at {b}. What is B's compass direction relative to A?", answer,
            OODATrace(f"A={a}, B={b}.", f"Delta=({b[0]-a[0]}, {b[1]-a[1]}).", f"Map the signs of delta to {answer}.", f"Output {answer}."), a=a, b=b)

    def shortest_path(self) -> SPODExample:
        size, start, goal = 6, (self.rng.randrange(6), self.rng.randrange(6)), (self.rng.randrange(6), self.rng.randrange(6))
        while goal == start:
            goal = (self.rng.randrange(6), self.rng.randrange(6))
        blocked: set[Point] = set()
        path = astar(size, start, goal, blocked) or []
        answer = str(len(path))
        return self._example("shortest_path", f"On a {size}x{size} empty grid, find the minimum number of 8-neighbour moves from {start} to {goal}.", answer,
            OODATrace(f"Start={start}, goal={goal}; no obstacles.", f"Chebyshev distance is {chebyshev(start, goal)}.", f"That is the shortest 8-neighbour path length.", f"Output {answer}."), path=path)

    def packing_shapes(self) -> SPODExample:
        container = (self.rng.randint(3, 8), self.rng.randint(3, 8))
        shape = (self.rng.randint(1, 9), self.rng.randint(1, 9))
        fits = (shape[0] <= container[0] and shape[1] <= container[1]) or (shape[1] <= container[0] and shape[0] <= container[1])
        answer = "YES" if fits else "NO"
        return self._example("packing_shapes", f"Can a {shape[0]}x{shape[1]} rectangle fit in a {container[0]}x{container[1]} rectangle with 90-degree rotation allowed?", answer,
            OODATrace(f"Container={container}; shape={shape}.", "Check original and rotated dimensions.", f"Fit condition is {fits}.", f"Output {answer}."))

    def mental_rotation(self) -> SPODExample:
        direction, turns = self.rng.choice(["N", "E", "S", "W"]), self.rng.randint(1, 3)
        answer = _rotate(direction, turns)
        return self._example("mental_rotation", f"An arrow points {direction}. Rotate it {turns * 90} degrees clockwise. Where does it point?", answer,
            OODATrace(f"Initial direction={direction}; clockwise turns={turns}.", "Use the cycle N -> E -> S -> W.", f"Rotation lands on {answer}.", f"Output {answer}."))

    def multi_point_relation(self) -> SPODExample:
        a = (0, 0)
        b = self.rng.choice([(2, 1), (-2, 1), (1, -2), (-1, -2)])
        c = (b[0] + self.rng.choice([-2, -1, 1, 2]), b[1] + self.rng.choice([-2, -1, 1, 2]))
        answer = _compass(c[0] - a[0], c[1] - a[1])
        return self._example("multi_point_relation", f"A={a}, B={b}, C={c}. What is C's direction relative to A?", answer,
            OODATrace(f"A={a}, B={b}, C={c}.", f"Compare C with A: delta=({c[0]}, {c[1]}).", f"The composite relation is {answer}.", f"Output {answer}."))

    def double_point_relation(self) -> SPODExample:
        labels, shift = ["A", "B", "C", "D", "E", "F", "G", "H"], self.rng.randint(1, 7)
        answer = labels[(-shift) % len(labels)]
        return self._example("double_point_relation", f"A circular ring has clockwise labels {labels}. It rotates clockwise by {shift} positions. Which original label is now at the front?", answer,
            OODATrace(f"Fixed ring labels={labels}; rotation={shift} clockwise.", "The front references the original index shifted backward.", f"Index (-{shift}) mod 8 is label {answer}.", f"Output {answer}."))

    def door_rotation(self) -> SPODExample:
        doors, target, shift = 10, self.rng.randint(1, 10), self.rng.randint(1, 9)
        answer = str(((target - shift - 1) % doors) + 1)
        return self._example("door_rotation", f"A circular room has {doors} doors. Target is behind door {target}. The room rotates clockwise by {shift} doors. Which door is now in front of the target?", answer,
            OODATrace(f"Target door={target}; rotation={shift} clockwise.", "Convert rotation to a negative fixed-room offset.", f"New front door index is {answer}.", f"Output {answer}."))

    def _multiturn(self, task: str, label: str) -> SPODExample:
        start = (self.rng.randint(0, 4), self.rng.randint(0, 4))
        moves = [self.rng.choice(list(DIRECTIONS)) for _ in range(4)]
        point = start
        states = []
        for move in moves:
            point = add(point, move)
            states.append(point)
        answer = str(point) if task != "mental_rotation_multiturn" else _rotate("N", sum(moves.count(x) for x in ["E", "SE", "NE"]) % 4)
        question = f"{label}: start at {start}; execute step-wise actions {' '.join(moves)}. Give the final state."
        return self._example(task, question, answer,
            OODATrace(f"Initial state={start}; actions={moves}.", f"Track states after each action: {states}.", f"The final required state is {answer}.", f"Output {answer}."), steps=states)

    def _path_control(self, task: str, obstacle_count: int) -> SPODExample:
        size = self.rng.choice([5, 7, 9])
        cells = [(r, c) for r in range(size) for c in range(size)]
        for _ in range(100):
            start, goal = self.rng.sample(cells, 2)
            obstacles = set(self.rng.sample([p for p in cells if p not in (start, goal)], min(obstacle_count, size)))
            path = astar(size, start, goal, obstacles)
            if path:
                answer = " ".join(path)
                return self._example(task, f"Grid {size}x{size}; start={start}; goal={goal}; obstacles={sorted(obstacles)}. Return a shortest sequence of 8-neighbour moves.", answer,
                    OODATrace(f"State: agent={start}, goal={goal}, obstacles={sorted(obstacles)}.", "Build a collision-free reachable-neighbour map.", f"A* gives a {len(path)}-step shortest path.", f"Output {answer}."), size=size, start=start, goal=goal, obstacles=sorted(obstacles), path=path)
        raise RuntimeError("could not sample a reachable control instance")

    def _multi_control(self, task: str, with_obstacles: bool) -> SPODExample:
        size, count = 7, 3
        starts = [(0, 0), (0, 6), (6, 0)]
        goals = [(2, 2), (2, 4), (4, 2)]
        obstacles = {(3, 3)} if with_obstacles else set()
        paths = [astar(size, start, goal, obstacles) for start, goal in zip(starts, goals)]
        assert all(paths)
        first = " ".join(f"{i + 1}:{path[0]}" for i, path in enumerate(paths) if path)
        return self._example(task, f"Synchronously control {count} agents on {size}x{size}. Starts={starts}; goals={goals}; obstacles={sorted(obstacles)}. Give their first actions.", first,
            OODATrace(f"Agents start={starts}; goals={goals}; obstacles={sorted(obstacles)}.", "Plan each safe path and check the first destinations do not collide.", f"First joint action is {first}.", f"Output {first}."), starts=starts, goals=goals, obstacles=sorted(obstacles))

    def generate_one(self, task: str) -> SPODExample:
        methods: dict[str, Callable[[], SPODExample]] = {
            "memory_path": self.memory_path, "object_location_distance": self.object_location_distance,
            "relative_direction": self.relative_direction, "shortest_path": self.shortest_path,
            "packing_shapes": self.packing_shapes, "mental_rotation": self.mental_rotation,
            "multi_point_relation": self.multi_point_relation, "double_point_relation": self.double_point_relation,
            "door_rotation": self.door_rotation,
            "door_rotation_multiturn": lambda: self._multiturn("door_rotation_multiturn", "Door rotation sequence"),
            "spatial_relation_multiturn": lambda: self._multiturn("spatial_relation_multiturn", "Spatial relation sequence"),
            "coordinates_movement_multiturn": lambda: self._multiturn("coordinates_movement_multiturn", "Coordinate movement sequence"),
            "mental_rotation_multiturn": lambda: self._multiturn("mental_rotation_multiturn", "Repeated mental rotation sequence"),
            "OP": lambda: self._path_control("OP", 5), "FOP": lambda: self._path_control("FOP", 0),
            "MCF": lambda: self._multi_control("MCF", False), "MCO": lambda: self._multi_control("MCO", True),
        }
        if task not in methods:
            raise KeyError(f"Unknown task {task}. Available: {', '.join(TASKS)}")
        return methods[task]()

    def generate(self, per_task: int, tasks: list[str] | None = None) -> list[SPODExample]:
        selected = tasks or list(TASKS)
        return [self.generate_one(task) for task in selected for _ in range(per_task)]
