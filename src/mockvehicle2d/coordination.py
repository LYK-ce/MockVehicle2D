"""Small deterministic space-time reservation planner for cooperative goto."""

from __future__ import annotations

from dataclasses import dataclass
import math

from mockvehicle2d.collision import swept_circles_overlap


Cell = tuple[int, int]
MAX_SIPP_DELAY_ITERATIONS = 128


@dataclass(frozen=True)
class TimedCell:
    """One cell occupancy interval, relative to a motion-intent receipt."""

    cell: Cell
    enter_offset_s: float
    leave_offset_s: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell, tuple)
            or len(self.cell) != 2
            or any(type(value) is not int for value in self.cell)
            or not all(
                math.isfinite(value)
                for value in (self.enter_offset_s, self.leave_offset_s)
            )
            or self.enter_offset_s < 0.0
            or self.leave_offset_s < self.enter_offset_s
        ):
            raise ValueError("timed cell must contain an integer cell and ordered offsets")


@dataclass(frozen=True)
class CellReservation:
    source_vehicle_id: str
    cell: Cell
    enter_time_s: float
    leave_time_s: float
    radius_m: float


@dataclass(frozen=True)
class EdgeReservation:
    source_vehicle_id: str
    from_cell: Cell
    to_cell: Cell
    enter_time_s: float
    leave_time_s: float
    radius_m: float


@dataclass(frozen=True)
class GoalReservation(CellReservation):
    pass


class ReservationTable:
    """Peer cell/edge/goal intervals queried by one local circular vehicle."""

    def __init__(
        self,
        resolution_m: float,
        *,
        own_radius_m: float,
        clearance_m: float,
    ) -> None:
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (resolution_m, own_radius_m, clearance_m)
        ) or resolution_m == 0.0:
            raise ValueError("reservation geometry must be finite and non-negative")
        self.resolution_m = resolution_m
        self.own_radius_m = own_radius_m
        self.clearance_m = clearance_m
        self.cells: list[CellReservation] = []
        self.edges: list[EdgeReservation] = []
        self.goals: list[GoalReservation] = []

    def add(
        self,
        source_vehicle_id: str,
        trajectory: tuple[TimedCell, ...],
        *,
        base_time_s: float,
        radius_m: float,
        time_margin_s: float,
        goal_hold: bool,
    ) -> None:
        if (
            not source_vehicle_id
            or not trajectory
            or not all(isinstance(item, TimedCell) for item in trajectory)
            or not all(
                math.isfinite(value)
                for value in (base_time_s, radius_m, time_margin_s)
            )
            or radius_m <= 0.0
            or time_margin_s < 0.0
            or type(goal_hold) is not bool
            or any(
                first.leave_offset_s > second.enter_offset_s
                for first, second in zip(trajectory, trajectory[1:])
            )
        ):
            raise ValueError("invalid reservation trajectory")

        for index, timed in enumerate(trajectory):
            start = base_time_s + timed.enter_offset_s - time_margin_s
            end = base_time_s + timed.leave_offset_s + time_margin_s
            reservation_type = (
                GoalReservation
                if goal_hold and index == len(trajectory) - 1
                else CellReservation
            )
            reservation = reservation_type(
                source_vehicle_id,
                timed.cell,
                start,
                math.inf if reservation_type is GoalReservation else end,
                radius_m,
            )
            self.cells.append(reservation)
            if isinstance(reservation, GoalReservation):
                self.goals.append(reservation)
        for first, second in zip(trajectory, trajectory[1:]):
            self.edges.append(
                EdgeReservation(
                    source_vehicle_id,
                    first.cell,
                    second.cell,
                    base_time_s + first.leave_offset_s - time_margin_s,
                    base_time_s + second.enter_offset_s + time_margin_s,
                    radius_m,
                )
            )

    def cell_conflict_end(
        self,
        cell: Cell,
        enter_time_s: float,
        leave_time_s: float,
    ) -> float | None:
        conflicts = [
            reservation.leave_time_s
            for reservation in self.cells
            if _intervals_overlap(
                enter_time_s,
                leave_time_s,
                reservation.enter_time_s,
                reservation.leave_time_s,
            )
            and self._cells_overlap(cell, reservation.cell, reservation.radius_m)
        ]
        point = self._center(cell)
        conflicts.extend(
            reservation.leave_time_s
            for reservation in self.edges
            if _intervals_overlap(
                enter_time_s,
                leave_time_s,
                reservation.enter_time_s,
                reservation.leave_time_s,
            )
            and self._stationary_edge_overlap(
                point,
                enter_time_s,
                leave_time_s,
                reservation,
            )
        )
        return max(conflicts, default=None)

    def edge_conflict_end(
        self,
        from_cell: Cell,
        to_cell: Cell,
        enter_time_s: float,
        leave_time_s: float,
    ) -> float | None:
        conflicts = [
            reservation.leave_time_s
            for reservation in self.edges
            if _intervals_overlap(
                enter_time_s,
                leave_time_s,
                reservation.enter_time_s,
                reservation.leave_time_s,
            )
            and self._edges_overlap(
                from_cell,
                to_cell,
                enter_time_s,
                leave_time_s,
                reservation,
            )
        ]
        conflicts.extend(
            reservation.leave_time_s
            for reservation in self.cells
            if _intervals_overlap(
                enter_time_s,
                leave_time_s,
                reservation.enter_time_s,
                reservation.leave_time_s,
            )
            and self._edge_stationary_overlap(
                from_cell,
                to_cell,
                enter_time_s,
                leave_time_s,
                reservation,
            )
        )
        return max(conflicts, default=None)

    def safe_intervals(
        self,
        cell: Cell,
        start_time_s: float,
        end_time_s: float,
    ) -> tuple[tuple[float, float], ...]:
        """Return conservative intervals where a vehicle may occupy ``cell``."""
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
            or not all(math.isfinite(value) for value in (start_time_s, end_time_s))
            or end_time_s < start_time_s
        ):
            raise ValueError("safe interval query is invalid")
        blocked = [
            (reservation.enter_time_s, reservation.leave_time_s)
            for reservation in self.cells
            if self._cells_overlap(cell, reservation.cell, reservation.radius_m)
        ]
        point = self._center(cell)
        blocked.extend(
            (reservation.enter_time_s, reservation.leave_time_s)
            for reservation in self.edges
            if self._stationary_edge_overlap(
                point,
                reservation.enter_time_s,
                reservation.leave_time_s,
                reservation,
            )
        )
        cursor = start_time_s
        safe = []
        for blocked_start, blocked_end in sorted(blocked):
            if blocked_end <= cursor or blocked_start >= end_time_s:
                continue
            if blocked_start > cursor:
                safe.append((cursor, min(blocked_start, end_time_s)))
            cursor = max(cursor, blocked_end)
            if cursor >= end_time_s:
                break
        if cursor < end_time_s:
            safe.append((cursor, end_time_s))
        return tuple(safe)

    def _center(self, cell: Cell) -> tuple[float, float]:
        return (
            (cell[0] + 0.5) * self.resolution_m,
            (cell[1] + 0.5) * self.resolution_m,
        )

    def _cells_overlap(self, first: Cell, second: Cell, radius_m: float) -> bool:
        return math.dist(self._center(first), self._center(second)) < (
            self.own_radius_m + radius_m + self.clearance_m
        )

    def _edges_overlap(
        self,
        from_cell: Cell,
        to_cell: Cell,
        enter_time_s: float,
        leave_time_s: float,
        reservation: EdgeReservation,
    ) -> bool:
        overlap = _overlap_interval(
            enter_time_s,
            leave_time_s,
            reservation.enter_time_s,
            reservation.leave_time_s,
        )
        if overlap is None:
            return False
        start, end = overlap
        return swept_circles_overlap(
            _edge_position(
                self._center(from_cell),
                self._center(to_cell),
                enter_time_s,
                leave_time_s,
                start,
            ),
            _edge_position(
                self._center(from_cell),
                self._center(to_cell),
                enter_time_s,
                leave_time_s,
                end,
            ),
            self.own_radius_m + self.clearance_m,
            _edge_position(
                self._center(reservation.from_cell),
                self._center(reservation.to_cell),
                reservation.enter_time_s,
                reservation.leave_time_s,
                start,
            ),
            _edge_position(
                self._center(reservation.from_cell),
                self._center(reservation.to_cell),
                reservation.enter_time_s,
                reservation.leave_time_s,
                end,
            ),
            reservation.radius_m,
        )

    def _stationary_edge_overlap(
        self,
        point: tuple[float, float],
        enter_time_s: float,
        leave_time_s: float,
        reservation: EdgeReservation,
    ) -> bool:
        overlap = _overlap_interval(
            enter_time_s,
            leave_time_s,
            reservation.enter_time_s,
            reservation.leave_time_s,
        )
        if overlap is None:
            return False
        start, end = overlap
        return swept_circles_overlap(
            point,
            point,
            self.own_radius_m + self.clearance_m,
            _edge_position(
                self._center(reservation.from_cell),
                self._center(reservation.to_cell),
                reservation.enter_time_s,
                reservation.leave_time_s,
                start,
            ),
            _edge_position(
                self._center(reservation.from_cell),
                self._center(reservation.to_cell),
                reservation.enter_time_s,
                reservation.leave_time_s,
                end,
            ),
            reservation.radius_m,
        )

    def _edge_stationary_overlap(
        self,
        from_cell: Cell,
        to_cell: Cell,
        enter_time_s: float,
        leave_time_s: float,
        reservation: CellReservation,
    ) -> bool:
        overlap = _overlap_interval(
            enter_time_s,
            leave_time_s,
            reservation.enter_time_s,
            reservation.leave_time_s,
        )
        if overlap is None:
            return False
        start, end = overlap
        point = self._center(reservation.cell)
        return swept_circles_overlap(
            _edge_position(
                self._center(from_cell),
                self._center(to_cell),
                enter_time_s,
                leave_time_s,
                start,
            ),
            _edge_position(
                self._center(from_cell),
                self._center(to_cell),
                enter_time_s,
                leave_time_s,
                end,
            ),
            self.own_radius_m + self.clearance_m,
            point,
            point,
            reservation.radius_m,
        )


def prioritized_sipp(
    path: tuple[Cell, ...],
    reservations: ReservationTable,
    *,
    now_s: float,
    horizon_s: float,
    linear_speed_mps: float,
    angular_speed_rps: float,
    initial_yaw_rad: float,
    time_margin_s: float,
) -> tuple[TimedCell, ...] | None:
    """Schedule the earliest safe traversal of one D*-guided spatial path."""
    values = (
        now_s,
        horizon_s,
        linear_speed_mps,
        angular_speed_rps,
        initial_yaw_rad,
        time_margin_s,
    )
    if (
        not path
        or any(
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
            for cell in path
        )
        or not all(math.isfinite(value) for value in values)
        or min(horizon_s, linear_speed_mps, angular_speed_rps) <= 0.0
        or time_margin_s < 0.0
    ):
        raise ValueError("invalid prioritized SIPP request")

    cells = tuple(
        cell
        for index, cell in enumerate(path)
        if not index or cell != path[index - 1]
    )
    interval_end_s = now_s + horizon_s + time_margin_s
    safe_intervals = tuple(
        reservations.safe_intervals(
            cell,
            now_s - time_margin_s,
            interval_end_s,
        )
        for cell in cells
    )
    states: list[dict[int, tuple[float, int | None, float | None]]] = [
        {} for _ in cells
    ]
    for interval_index, (safe_start, safe_end) in enumerate(safe_intervals[0]):
        if (
            safe_start <= now_s - time_margin_s
            and safe_end >= now_s + time_margin_s
        ):
            states[0][interval_index] = (0.0, None, None)
            break
    if not states[0]:
        return None

    heading = initial_yaw_rad
    for path_index, (current, following) in enumerate(zip(cells, cells[1:])):
        target_heading = math.atan2(
            following[1] - current[1],
            following[0] - current[0],
        )
        turn_s = abs(
            math.atan2(
                math.sin(target_heading - heading),
                math.cos(target_heading - heading),
            )
        ) / angular_speed_rps
        travel_s = (
            math.dist(current, following)
            * reservations.resolution_m
            / linear_speed_mps
        )
        for current_interval, (arrival, _, _) in states[path_index].items():
            current_safe_start, current_safe_end = safe_intervals[path_index][
                current_interval
            ]
            if now_s + arrival - time_margin_s < current_safe_start:
                continue
            for destination_interval, (
                destination_start,
                destination_end,
            ) in enumerate(safe_intervals[path_index + 1]):
                departure = max(
                    arrival + turn_s,
                    destination_start - now_s + time_margin_s - travel_s,
                )
                for _ in range(MAX_SIPP_DELAY_ITERATIONS):
                    next_arrival = departure + travel_s
                    if (
                        next_arrival > horizon_s
                        or now_s + departure + time_margin_s
                        > current_safe_end
                        or now_s + next_arrival - time_margin_s
                        < destination_start
                        or now_s + next_arrival + time_margin_s
                        > destination_end
                    ):
                        break
                    conflict_end = reservations.edge_conflict_end(
                        current,
                        following,
                        now_s + departure - time_margin_s,
                        now_s + next_arrival + time_margin_s,
                    )
                    if conflict_end is None:
                        previous = states[path_index + 1].get(destination_interval)
                        if previous is None or next_arrival < previous[0] - 1e-12:
                            states[path_index + 1][destination_interval] = (
                                next_arrival,
                                current_interval,
                                departure,
                            )
                        break
                    if math.isinf(conflict_end):
                        break
                    delayed = conflict_end - now_s + time_margin_s
                    if delayed <= departure:
                        break
                    departure = delayed
        heading = target_heading

    selected: tuple[int, int] | None = None
    for path_index in range(len(cells) - 1, -1, -1):
        candidates = [
            (arrival, interval_index)
            for interval_index, (arrival, _, _) in states[path_index].items()
            if safe_intervals[path_index][interval_index][1]
            >= interval_end_s
        ]
        if candidates:
            _, interval_index = min(candidates)
            selected = path_index, interval_index
            break
    if selected is None or (selected[0] == 0 and len(cells) > 1):
        return None

    final_index, interval_index = selected
    arrivals = [0.0] * (final_index + 1)
    departures = [0.0] * final_index
    for path_index in range(final_index, -1, -1):
        arrival, previous_interval, departure = states[path_index][interval_index]
        arrivals[path_index] = arrival
        if path_index:
            assert previous_interval is not None and departure is not None
            departures[path_index - 1] = departure
            interval_index = previous_interval
    return tuple(
        TimedCell(
            cell,
            arrivals[index],
            horizon_s if index == final_index else departures[index],
        )
        for index, cell in enumerate(cells[: final_index + 1])
    )


def _intervals_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    return first_start < second_end and second_start < first_end


def _overlap_interval(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> tuple[float, float] | None:
    start, end = max(first_start, second_start), min(first_end, second_end)
    return None if start >= end else (start, end)


def _edge_position(
    start: tuple[float, float],
    end: tuple[float, float],
    enter_time_s: float,
    leave_time_s: float,
    at_time_s: float,
) -> tuple[float, float]:
    duration = leave_time_s - enter_time_s
    ratio = 0.0 if duration <= 1e-12 else max(
        0.0,
        min(1.0, (at_time_s - enter_time_s) / duration),
    )
    return (
        start[0] + ratio * (end[0] - start[0]),
        start[1] + ratio * (end[1] - start[1]),
    )
