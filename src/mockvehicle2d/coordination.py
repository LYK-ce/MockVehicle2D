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
    arrivals = [0.0]
    departures: list[float] = []
    heading = initial_yaw_rad
    for current, following in zip(cells, cells[1:]):
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
        departure = arrivals[-1] + turn_s
        for _ in range(MAX_SIPP_DELAY_ITERATIONS):
            arrival = departure + travel_s
            if arrival > horizon_s + 1e-12:
                break
            conflict_end = reservations.edge_conflict_end(
                current,
                following,
                now_s + departure - time_margin_s,
                now_s + arrival + time_margin_s,
            )
            destination_conflict_end = reservations.cell_conflict_end(
                following,
                now_s + arrival - time_margin_s,
                now_s + arrival + time_margin_s,
            )
            conflict_end = max(
                (
                    value
                    for value in (conflict_end, destination_conflict_end)
                    if value is not None
                ),
                default=None,
            )
            if conflict_end is None:
                break
            if math.isinf(conflict_end):
                departure = math.inf
                break
            departure = max(departure, conflict_end - now_s + time_margin_s)
            hold_conflict = reservations.cell_conflict_end(
                current,
                now_s + arrivals[-1] - time_margin_s,
                now_s + departure + time_margin_s,
            )
            if hold_conflict is not None:
                departure = math.inf
                break
        else:
            departure = math.inf
        if not math.isfinite(departure) or departure + travel_s > horizon_s + 1e-12:
            break
        departures.append(departure)
        arrivals.append(departure + travel_s)
        heading = target_heading

    scheduled_cells = cells[: len(arrivals)]
    if len(scheduled_cells) == 1 and len(cells) > 1:
        return None
    result = [
        TimedCell(cell, arrival, departures[index])
        for index, (cell, arrival) in enumerate(
            zip(scheduled_cells[:-1], arrivals[:-1])
        )
    ]
    final_leave = horizon_s
    final_conflict = reservations.cell_conflict_end(
        scheduled_cells[-1],
        now_s + arrivals[-1] - time_margin_s,
        now_s + horizon_s + time_margin_s,
    )
    if final_conflict is not None:
        final_leave = arrivals[-1]
    result.append(TimedCell(scheduled_cells[-1], arrivals[-1], final_leave))
    return tuple(result)


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
