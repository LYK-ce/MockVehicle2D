"""Incremental D* Lite over one finite window of a sparse observed grid."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from mockvehicle2d.collision import (
    cell_overlaps_circle,
    is_strict_overlap,
    segment_aabb_distance_squared,
)
from mockvehicle2d.local_state import (
    FORBIDDEN,
    FREE,
    OCCUPIED,
    UNKNOWN,
    MapCellUpdate,
    ObservedGrid,
)


Cell = tuple[int, int]
Key = tuple[float, float]
SQRT_2 = math.sqrt(2)
EGRESS_PROBE_FRACTION = 1e-6
MOVES = tuple(
    (dx, dy, SQRT_2 if dx and dy else 1.0)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if dx or dy
)


class DStarLitePlanner:
    """Reuse shortest-path state as the robot moves and observed cells change."""

    def __init__(
        self,
        grid: ObservedGrid,
        *,
        vehicle_radius_m: float = 0.5,
        unknown_cost: float = 3.0,
        bounds_margin_m: float = 16.0,
        max_goal_distance_m: float = 256.0,
        max_cells: int = 100_000,
    ) -> None:
        if (
            not math.isfinite(vehicle_radius_m)
            or vehicle_radius_m < 0
            or not math.isfinite(unknown_cost)
            or unknown_cost < 1
            or not math.isfinite(bounds_margin_m)
            or bounds_margin_m < 0
            or not math.isfinite(max_goal_distance_m)
            or max_goal_distance_m <= 0
            or type(max_cells) is not int
            or max_cells <= 0
        ):
            raise ValueError("invalid D* Lite configuration")
        self.resolution_m = grid.resolution_m
        self.vehicle_radius_m = vehicle_radius_m
        self.unknown_cost = unknown_cost
        self.bounds_margin_m = bounds_margin_m
        self._bounds_margin_cells = math.ceil(bounds_margin_m / grid.resolution_m)
        self.max_goal_distance_m = max_goal_distance_m
        self.max_cells = max_cells
        self._vehicle_radius_cells = vehicle_radius_m / grid.resolution_m
        self._inflation_cells = math.ceil(self._vehicle_radius_cells)
        self._states = {
            (cell["gx"], cell["gy"]): cell["state"]
            for cell in grid.snapshot()["cells"]
        }
        self._bounds: tuple[int, int, int, int] | None = None
        self._start: Cell | None = None
        self._last_start: Cell | None = None
        self._goal: Cell | None = None
        self._key_modifier_cost = 0.0
        self._g: dict[Cell, float] = {}
        self._rhs: dict[Cell, float] = {}
        self._queue: list[tuple[float, float, int, int]] = []
        self._open_keys: dict[Cell, Key] = {}
        self._expansions = 0
        self._incremental_updates = 0
        self._replans = 0
        self._resets = 0
        self.last_failure: str | None = None

    @property
    def stats(self) -> dict[str, float | int]:
        return {
            "expansions": self._expansions,
            "incremental_updates": self._incremental_updates,
            "replans": self._replans,
            "resets": self._resets,
            "key_modifier_cost": self._key_modifier_cost,
        }

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        return self._bounds

    def planning_budget_allows(self, start: Cell, goal: Cell) -> bool:
        self._validate_cell(start, "start")
        self._validate_cell(goal, "goal")
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) * (
            self.resolution_m
        ) > self.max_goal_distance_m:
            return False
        return self._bounds_cell_count(
            self._planning_bounds(start, goal)
        ) <= self.max_cells

    def is_segment_passable(
        self,
        source_m: tuple[float, float],
        destination_m: tuple[float, float],
        *,
        extra_clearance_m: float = 0.0,
        require_observed: bool = False,
    ) -> bool:
        if (
            not isinstance(source_m, tuple)
            or len(source_m) != 2
            or not isinstance(destination_m, tuple)
            or len(destination_m) != 2
        ):
            raise ValueError("segment endpoints must be finite metric pairs")
        values = (*source_m, *destination_m, extra_clearance_m)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ) or extra_clearance_m < 0 or type(require_observed) is not bool:
            raise ValueError(
                "segment endpoints and clearance must be finite; "
                "clearance cannot be negative and require_observed must be boolean"
            )
        return not self._segment_blocked(
            (
                source_m[0] / self.resolution_m,
                source_m[1] / self.resolution_m,
            ),
            (
                destination_m[0] / self.resolution_m,
                destination_m[1] / self.resolution_m,
            ),
            radius_cells=(
                self.vehicle_radius_m + extra_clearance_m
            ) / self.resolution_m,
            block_tangent=extra_clearance_m > 0,
            require_observed=require_observed,
        )

    def best_start_connection(
        self,
        source_m: tuple[float, float],
        current: Cell,
    ) -> Cell | None:
        self._validate_cell(current, "current")
        if self._goal is None or not self._inside(current):
            return None
        candidates = (current, *self._neighbours(current))
        source = (
            source_m[0] / self.resolution_m,
            source_m[1] / self.resolution_m,
        )
        choices = []
        for candidate in candidates:
            if self._blocked(candidate) or math.isinf(self._g_value(candidate)):
                continue
            destination = candidate[0] + 0.5, candidate[1] + 0.5
            if self._segment_blocked(
                source,
                destination,
                allow_forbidden_egress=True,
            ):
                continue
            connector = (
                math.hypot(
                    destination[0] - source[0],
                    destination[1] - source[1],
                )
                * self.resolution_m
                * (
                    1.0
                    if self._states.get(candidate, UNKNOWN) == FREE
                    else self.unknown_cost
                )
            )
            choices.append(
                (
                    connector + self._g_value(candidate),
                    _octile(candidate, self._goal, self.resolution_m),
                    candidate,
                )
            )
        return None if not choices else min(choices)[2]

    def plan(
        self,
        start: Cell,
        goal: Cell,
        *,
        changed_cells: Iterable[MapCellUpdate] = (),
    ) -> list[Cell] | None:
        self._validate_cell(start, "start")
        self._validate_cell(goal, "goal")
        self.last_failure = None
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) * self.resolution_m > (
            self.max_goal_distance_m
        ):
            raise ValueError("goal exceeds maximum distance")

        changes = tuple(changed_cells)
        for change in changes:
            self._validate_update(change)
            if change.state == UNKNOWN:
                self._states.pop((change.gx, change.gy), None)
            else:
                self._states[(change.gx, change.gy)] = change.state

        needs_reset = self._goal != goal or self._bounds is None
        if not needs_reset and not self._inside(start):
            needs_reset = True
        if needs_reset:
            self._reset(start, goal)
        else:
            assert self._last_start is not None
            self._key_modifier_cost += _octile(
                self._last_start, start, self.resolution_m
            )
            self._start = start
            self._last_start = start
            if changes:
                self._apply_changes(changes)
        self._replans += 1
        if self._blocked(start):
            self.last_failure = "start_blocked"
            return None
        if self._blocked(goal):
            self.last_failure = "goal_blocked"
            return None
        self._compute_shortest_path()
        path = self._extract_path()
        if path is None:
            self.last_failure = (
                "search_exhausted"
                if math.isinf(self._g_value(start))
                else "path_extraction"
            )
        return path

    def _reset(self, start: Cell, goal: Cell) -> None:
        bounds = self._planning_bounds(start, goal)
        if self._bounds_cell_count(bounds) > self.max_cells:
            raise ValueError("planning cell budget exceeded")
        self._bounds = bounds
        self._start = self._last_start = start
        self._goal = goal
        self._key_modifier_cost = 0.0
        self._g.clear()
        self._rhs = {goal: 0.0}
        self._queue.clear()
        self._open_keys.clear()
        self._push(goal)
        self._resets += 1

    def _apply_changes(self, changes: tuple[MapCellUpdate, ...]) -> None:
        affected: set[Cell] = set()
        radius = self._inflation_cells + 1
        for change in changes:
            for gx in range(change.gx - radius, change.gx + radius + 1):
                for gy in range(change.gy - radius, change.gy + radius + 1):
                    cell = gx, gy
                    if self._inside(cell):
                        affected.add(cell)
                        affected.update(self._neighbours(cell))
        for cell in sorted(affected):
            self._update_vertex(cell)
        self._incremental_updates += len(affected)

    def _compute_shortest_path(self) -> None:
        assert self._start is not None
        limit = self._cell_count() * 20
        expansions = 0
        while (
            _key_less(self._top_key(), self._calculate_key(self._start))
            or not math.isclose(self._rhs_value(self._start), self._g_value(self._start))
        ):
            popped = self._pop()
            if popped is None:
                break
            old_key, current = popped
            new_key = self._calculate_key(current)
            if _key_less(old_key, new_key):
                self._push(current)
            elif self._g_value(current) > self._rhs_value(current):
                self._g[current] = self._rhs_value(current)
                for predecessor in self._neighbours(current):
                    self._update_vertex(predecessor)
            else:
                self._g[current] = math.inf
                self._update_vertex(current)
                for predecessor in self._neighbours(current):
                    self._update_vertex(predecessor)
            expansions += 1
            if expansions > limit:
                raise RuntimeError("D* Lite expansion limit exceeded")
        self._expansions += expansions

    def _extract_path(self) -> list[Cell] | None:
        assert self._start is not None and self._goal is not None
        if math.isinf(self._g_value(self._start)):
            return None
        current = self._start
        path = [current]
        visited = {current}
        while current != self._goal:
            choices = []
            for neighbour in self._neighbours(current):
                edge = self._cost(current, neighbour)
                total = edge + self._g_value(neighbour)
                if math.isfinite(total):
                    choices.append(
                        (
                            total,
                            _octile(neighbour, self._goal, self.resolution_m),
                            neighbour,
                        )
                    )
            if not choices:
                return None
            _, _, current = min(choices)
            if current in visited:
                return None
            visited.add(current)
            path.append(current)
            if len(path) > self._cell_count():
                return None
        return path

    def _update_vertex(self, cell: Cell) -> None:
        assert self._goal is not None
        if cell != self._goal:
            self._rhs[cell] = min(
                (
                    self._cost(cell, successor) + self._g_value(successor)
                    for successor in self._neighbours(cell)
                ),
                default=math.inf,
            )
        self._open_keys.pop(cell, None)
        if not math.isclose(self._g_value(cell), self._rhs_value(cell)):
            self._push(cell)

    def _cost(self, source: Cell, destination: Cell) -> float:
        if self._blocked(source) or self._blocked(destination):
            return math.inf
        dx, dy = destination[0] - source[0], destination[1] - source[1]
        if dx and dy and (
            self._blocked((source[0] + dx, source[1]))
            or self._blocked((source[0], source[1] + dy))
        ):
            return math.inf
        if self._segment_blocked(
            (source[0] + 0.5, source[1] + 0.5),
            (destination[0] + 0.5, destination[1] + 0.5),
        ):
            return math.inf
        step = self.resolution_m * (SQRT_2 if dx and dy else 1.0)
        state = self._states.get(destination, UNKNOWN)
        return step * (1.0 if state == FREE else self.unknown_cost)

    def _segment_blocked(
        self,
        source: tuple[float, float],
        destination: tuple[float, float],
        *,
        allow_forbidden_egress: bool = False,
        radius_cells: float | None = None,
        block_tangent: bool = False,
        require_observed: bool = False,
    ) -> bool:
        source_x, source_y = source
        destination_x, destination_y = destination
        radius = (
            self._vehicle_radius_cells
            if radius_cells is None
            else radius_cells
        )
        radius_squared = radius**2
        for gy in range(
            math.floor(min(source_y, destination_y) - radius) - 1,
            math.floor(max(source_y, destination_y) + radius) + 1,
        ):
            for gx in range(
                math.floor(min(source_x, destination_x) - radius) - 1,
                math.floor(max(source_x, destination_x) + radius) + 1,
            ):
                state = self._states.get((gx, gy), UNKNOWN)
                if state not in {OCCUPIED, FORBIDDEN} and not (
                    require_observed and state != FREE
                ):
                    continue
                distance_squared = segment_aabb_distance_squared(
                    source_x,
                    source_y,
                    destination_x,
                    destination_y,
                    gx,
                    gy,
                    gx + 1,
                    gy + 1,
                )
                if is_strict_overlap(
                    distance_squared, radius_squared
                ) or (
                    block_tangent
                    and math.isclose(
                        distance_squared,
                        radius_squared,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    if allow_forbidden_egress and state == FORBIDDEN:
                        source_distance_squared = segment_aabb_distance_squared(
                            source_x,
                            source_y,
                            source_x,
                            source_y,
                            gx,
                            gy,
                            gx + 1,
                            gy + 1,
                        )
                        destination_distance_squared = (
                            segment_aabb_distance_squared(
                                destination_x,
                                destination_y,
                                destination_x,
                                destination_y,
                                gx,
                                gy,
                                gx + 1,
                                gy + 1,
                            )
                        )
                        probe_x = source_x + EGRESS_PROBE_FRACTION * (
                            destination_x - source_x
                        )
                        probe_y = source_y + EGRESS_PROBE_FRACTION * (
                            destination_y - source_y
                        )
                        probe_distance_squared = segment_aabb_distance_squared(
                            probe_x,
                            probe_y,
                            probe_x,
                            probe_y,
                            gx,
                            gy,
                            gx + 1,
                            gy + 1,
                        )
                        if (
                            math.isclose(
                                distance_squared,
                                source_distance_squared,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            and probe_distance_squared
                            > source_distance_squared
                            and not math.isclose(
                                probe_distance_squared,
                                source_distance_squared,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            and destination_distance_squared
                            > source_distance_squared
                            and not math.isclose(
                                destination_distance_squared,
                                source_distance_squared,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                        ):
                            continue
                    return True
        return False

    def _blocked(self, cell: Cell) -> bool:
        if not self._inside(cell):
            return True
        radius = self._inflation_cells
        centre_x, centre_y = cell[0] + 0.5, cell[1] + 0.5
        radius_squared = self._vehicle_radius_cells**2
        return any(
            self._states.get((gx, gy), UNKNOWN) in {OCCUPIED, FORBIDDEN}
            and (
                (gx, gy) == cell
                or cell_overlaps_circle(
                    gx, gy, centre_x, centre_y, radius_squared
                )
            )
            for gx in range(cell[0] - radius, cell[0] + radius + 1)
            for gy in range(cell[1] - radius, cell[1] + radius + 1)
        )

    def _neighbours(self, cell: Cell) -> tuple[Cell, ...]:
        return tuple(
            (cell[0] + dx, cell[1] + dy)
            for dx, dy, _ in MOVES
            if self._inside((cell[0] + dx, cell[1] + dy))
        )

    def _calculate_key(self, cell: Cell) -> Key:
        assert self._start is not None
        value = min(self._g_value(cell), self._rhs_value(cell))
        return (
            value
            + _octile(self._start, cell, self.resolution_m)
            + self._key_modifier_cost,
            value,
        )

    def _push(self, cell: Cell) -> None:
        key = self._calculate_key(cell)
        self._open_keys[cell] = key
        heapq.heappush(self._queue, (key[0], key[1], cell[0], cell[1]))

    def _top_key(self) -> Key:
        while self._queue:
            key = self._queue[0][0], self._queue[0][1]
            cell = self._queue[0][2], self._queue[0][3]
            if self._open_keys.get(cell) == key:
                return key
            heapq.heappop(self._queue)
        return math.inf, math.inf

    def _pop(self) -> tuple[Key, Cell] | None:
        while self._queue:
            first, second, x, y = heapq.heappop(self._queue)
            cell = x, y
            key = first, second
            if self._open_keys.get(cell) == key:
                del self._open_keys[cell]
                return key, cell
        return None

    def _g_value(self, cell: Cell) -> float:
        return self._g.get(cell, math.inf)

    def _rhs_value(self, cell: Cell) -> float:
        return self._rhs.get(cell, math.inf)

    def _inside(self, cell: Cell) -> bool:
        if self._bounds is None:
            return False
        return (
            self._bounds[0] <= cell[0] <= self._bounds[2]
            and self._bounds[1] <= cell[1] <= self._bounds[3]
        )

    def _cell_count(self) -> int:
        assert self._bounds is not None
        return self._bounds_cell_count(self._bounds)

    def _planning_bounds(
        self,
        start: Cell,
        goal: Cell,
    ) -> tuple[int, int, int, int]:
        margin = self._bounds_margin_cells
        return (
            min(start[0], goal[0]) - margin,
            min(start[1], goal[1]) - margin,
            max(start[0], goal[0]) + margin,
            max(start[1], goal[1]) + margin,
        )

    @staticmethod
    def _bounds_cell_count(bounds: tuple[int, int, int, int]) -> int:
        return (bounds[2] - bounds[0] + 1) * (
            bounds[3] - bounds[1] + 1
        )

    @staticmethod
    def _validate_cell(cell: Cell, name: str) -> None:
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
        ):
            raise ValueError(f"{name} must be an integer cell")

    @staticmethod
    def _validate_update(update: MapCellUpdate) -> None:
        if (
            not isinstance(update, MapCellUpdate)
            or type(update.gx) is not int
            or type(update.gy) is not int
            or update.state not in {UNKNOWN, FREE, OCCUPIED, FORBIDDEN}
        ):
            raise ValueError("invalid map cell update")


def _octile(first: Cell, second: Cell, resolution_m: float) -> float:
    dx, dy = abs(first[0] - second[0]), abs(first[1] - second[1])
    return resolution_m * (max(dx, dy) + (SQRT_2 - 1) * min(dx, dy))


def _key_less(first: Key, second: Key) -> bool:
    if not math.isclose(first[0], second[0]):
        return first[0] < second[0]
    return not math.isclose(first[1], second[1]) and first[1] < second[1]
