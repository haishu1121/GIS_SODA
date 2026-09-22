"""Text-state grid worlds used for OP/FOP/MCF/MCO and deployment demos."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

Point = tuple[int, int]
DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def add(point: Point, move: str) -> Point:
    dr, dc = DIRECTIONS[move]
    return point[0] + dr, point[1] + dc


def direction_between(start: Point, end: Point) -> str:
    dr, dc = end[0] - start[0], end[1] - start[1]
    return next(name for name, delta in DIRECTIONS.items() if delta == (dr, dc))


def chebyshev(a: Point, b: Point) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def astar(size: int, start: Point, goal: Point, obstacles: Iterable[Point] = ()) -> list[str] | None:
    """Shortest 8-neighbour path; `None` denotes an unreachable goal."""
    blocked = set(obstacles)
    if start in blocked or goal in blocked:
        return None
    queue: list[tuple[int, int, Point, list[str]]] = [(chebyshev(start, goal), 0, start, [])]
    best = {start: 0}
    while queue:
        _, cost, current, moves = heapq.heappop(queue)
        if current == goal:
            return moves
        if cost != best.get(current):
            continue
        for name in DIRECTIONS:
            nxt = add(current, name)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in blocked:
                continue
            candidate = cost + 1
            if candidate < best.get(nxt, 10**9):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate + chebyshev(nxt, goal), candidate, nxt, moves + [name]))
    return None


@dataclass
class GridWorld:
    size: int
    position: Point
    goal: Point
    obstacles: set[Point] = field(default_factory=set)
    history: list[str] = field(default_factory=list)

    def state_text(self) -> str:
        return (
            f"Grid: {self.size}x{self.size}. Agent: {self.position}. Goal: {self.goal}. "
            f"Obstacles: {sorted(self.obstacles)}. Allowed moves: {', '.join(DIRECTIONS)}."
        )

    def step(self, action: str) -> Point:
        action = action.strip().upper()
        if action not in DIRECTIONS:
            raise ValueError(f"invalid action {action!r}")
        target = add(self.position, action)
        if not (0 <= target[0] < self.size and 0 <= target[1] < self.size):
            raise ValueError("action leaves the grid")
        if target in self.obstacles:
            raise ValueError("action collides with an obstacle")
        self.position = target
        self.history.append(action)
        return target

    @property
    def complete(self) -> bool:
        return self.position == self.goal


def oracle_action(world: GridWorld) -> str:
    path = astar(world.size, world.position, world.goal, world.obstacles)
    if path is None:
        raise RuntimeError("goal is unreachable")
    return "NONE" if not path else path[0]


def run_closed_loop(world: GridWorld, policy, max_steps: int = 100) -> GridWorld:
    """Execute one action at a time, re-encoding state after every physical step."""
    for _ in range(max_steps):
        if world.complete:
            return world
        action = policy(world.state_text())
        if action == "NONE":
            raise RuntimeError("policy terminated before reaching the goal")
        world.step(action)
    raise RuntimeError("control loop exceeded max_steps")
