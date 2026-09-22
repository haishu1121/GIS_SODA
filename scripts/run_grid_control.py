"""Demonstrate SODA's external closed loop with a deterministic safe planner."""

from __future__ import annotations

from soda.control import GridWorld, oracle_action, run_closed_loop


def main() -> None:
    world = GridWorld(size=7, position=(0, 0), goal=(6, 6), obstacles={(1, 1), (2, 2), (3, 3), (4, 4)})
    result = run_closed_loop(world, lambda _: oracle_action(world))
    print(f"complete={result.complete}; path={' '.join(result.history)}")


if __name__ == "__main__":
    main()
