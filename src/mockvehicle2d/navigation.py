"""Finite-view D* Lite go-to-goal controller."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import (
    HARD_STOP_CLEARANCE_M,
    LocalSafetyRuntime,
    SafetyAdvanceResult,
)
from mockvehicle2d.vehicle import Vehicle

if TYPE_CHECKING:
    from mockvehicle2d.local_state import (
        LocalMapDelta,
        MapCellUpdate,
        ObservedGrid,
        PoseEstimate,
    )


DEGRADED_LINEAR_SCALE = 0.5
UNKNOWN_LINEAR_SCALE = 0.4
MAX_REPORTED_PATH_CELLS = 64
NEARBY_SAFE_BODY_DISTANCE_M = 1.0
# Hardware calibration knob: lower this if one 6 Hz planning slice misses its
# deadline on the target Jetson.
PLANNING_EXPANSIONS_PER_UPDATE = 256
SafeCandidate = tuple[tuple[float, float], tuple[int, int], bool]


class GotoController:
    """Plan from one estimated pose and observed map; never read simulator pose truth."""

    goal_tolerance_m = 0.1
    yaw_tolerance_rad = math.radians(2)
    turn_in_place_threshold_rad = math.radians(20)

    def __init__(self) -> None:
        self.status = "idle"
        self.mode = "position"
        self.requested_goal: tuple[float, float] | None = None
        self.goal: tuple[float, float] | None = None
        self.reported_goal: tuple[float, float] | None = None
        self.goal_mode: str | None = None
        self.yaw_goal_rad: float | None = None
        self.reported_yaw_goal_rad: float | None = None
        self.reason: str | None = None
        self.detail: str | None = None
        self._planner: DStarLitePlanner | None = None
        self._path: list[tuple[int, int]] = []
        self._path_revision = 0
        self._replan_count = 0
        self._current_waypoint: tuple[float, float] | None = None
        self._path_resolution_m = 1.0
        self._nearby_detail: str | None = None
        self._vehicle_radius_m = 0.5
        self._planning_kind: str | None = None
        self._planning_map_changed = False
        self._planning_previous_path: list[tuple[int, int]] = []
        self._safe_candidates: list[SafeCandidate] = []
        self._safe_candidate_index = 0
        self._pending_candidate: SafeCandidate | None = None

    @property
    def control_mode(self) -> str:
        return "autonomous" if self.status == "active" else "manual"

    def start(
        self,
        x_m: float,
        y_m: float,
        *,
        reported_goal: tuple[float, float] | None = None,
        local_map: ObservedGrid,
        pose: PoseEstimate,
        vehicle_radius_m: float = 0.5,
    ) -> None:
        self.mode = "position"
        self.requested_goal = (x_m, y_m)
        self.goal = self.requested_goal
        self.reported_goal = self.goal if reported_goal is None else reported_goal
        self.goal_mode = "exact"
        self.yaw_goal_rad = None
        self.reported_yaw_goal_rad = None
        self.status = "active"
        self.reason = None
        self.detail = None
        self._planner = None
        self._path = []
        self._path_revision = 0
        self._replan_count = 0
        self._current_waypoint = None
        self._path_resolution_m = local_map.resolution_m
        self._nearby_detail = None
        self._vehicle_radius_m = vehicle_radius_m
        self._planning_kind = "goal"
        self._planning_map_changed = False
        self._planning_previous_path = []
        self._safe_candidates = []
        self._safe_candidate_index = 0
        self._pending_candidate = None
        self._planner = DStarLitePlanner(
            local_map, vehicle_radius_m=vehicle_radius_m
        )
        self._advance_planning(
            pose,
            local_map,
            (),
            PLANNING_EXPANSIONS_PER_UPDATE,
        )

    def start_rotation(
        self,
        yaw_goal_rad: float,
        *,
        reported_yaw_rad: float | None = None,
    ) -> None:
        if not math.isfinite(yaw_goal_rad) or (
            reported_yaw_rad is not None and not math.isfinite(reported_yaw_rad)
        ):
            raise ValueError("yaw goal must be finite")
        self.mode = "rotation"
        self.requested_goal = None
        self.goal = None
        self.reported_goal = None
        self.goal_mode = None
        self.yaw_goal_rad = _wrap_angle(yaw_goal_rad)
        self.reported_yaw_goal_rad = _wrap_angle(
            self.yaw_goal_rad
            if reported_yaw_rad is None
            else reported_yaw_rad
        )
        self.status = "active"
        self.reason = None
        self.detail = None
        self._planner = None
        self._path = []
        self._path_revision = 0
        self._replan_count = 0
        self._current_waypoint = None
        self._nearby_detail = None
        self._clear_pending_planning()

    def cancel(self, reason: str) -> None:
        if self.status == "active":
            self.status = "cancelled"
            self.reason = reason
            self.detail = None
            self._clear_pending_planning()

    def block_for_localization_loss(
        self, vehicle: Vehicle, pose: PoseEstimate, now: float | None = None
    ) -> bool:
        if self.status != "active" or pose.quality != "lost":
            return False
        vehicle.stop(now)
        self.status = "blocked"
        self.reason = "localization_lost"
        self.detail = None
        self._clear_pending_planning()
        return True

    def snapshot(self) -> dict[str, object]:
        if self.mode == "rotation" and self.reported_yaw_goal_rad is not None:
            goal = {"yaw_rad": self.reported_yaw_goal_rad}
        else:
            goal = (
                None
                if self.reported_goal is None
                else {"x_m": self.reported_goal[0], "y_m": self.reported_goal[1]}
            )
        snapshot: dict[str, object] = {
            "status": self.status,
            "mode": self.mode,
            "goal": goal,
            "requested_goal": (
                None
                if self.requested_goal is None
                else {
                    "frame_id": "anchor_map",
                    "x_m": self.requested_goal[0],
                    "y_m": self.requested_goal[1],
                }
            ),
            "goal_mode": self.goal_mode,
            "effective_goal": (
                None
                if self.goal is None
                else {
                    "frame_id": "anchor_map",
                    "x_m": self.goal[0],
                    "y_m": self.goal[1],
                }
            ),
            "reason": self.reason,
            "detail": self.detail,
            "approach_distance_m": self._approach_distance_m(),
            "planning": self._planning_kind is not None,
        }
        if self._planner is None:
            return snapshot
        snapshot.update({
            "algorithm": "d_star_lite",
            "path_revision": self._path_revision,
            "replan_count": self._replan_count,
            "current_waypoint": (
                None
                if self._current_waypoint is None
                else {"x_m": self._current_waypoint[0], "y_m": self._current_waypoint[1]}
            ),
            "path": [
                {
                    "x_m": (gx + 0.5) * self._path_resolution_m,
                    "y_m": (gy + 0.5) * self._path_resolution_m,
                }
                for gx, gy in self._path[:MAX_REPORTED_PATH_CELLS]
            ],
            "planner_stats": self._planner.stats,
        })
        return snapshot

    def replan(
        self,
        pose: PoseEstimate,
        map_delta: LocalMapDelta | None,
        local_map: ObservedGrid,
    ) -> None:
        if self._planner is None:
            raise RuntimeError("active navigation has no D* Lite planner")
        changes = () if map_delta is None else map_delta.changed_cells
        if self._planning_kind is None:
            self._planning_kind = "goal"
            self._planning_previous_path = list(self._path)
        self._planning_map_changed |= bool(changes)
        self._advance_planning(
            pose,
            local_map,
            changes,
            PLANNING_EXPANSIONS_PER_UPDATE,
        )

    def update(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        now: float,
        safety: LocalSafetyRuntime | None = None,
        *,
        pose: PoseEstimate | None = None,
        advance_result: SafetyAdvanceResult | None = None,
        local_map: ObservedGrid | None = None,
        map_delta: LocalMapDelta | None = None,
    ) -> None:
        was_active = self.status == "active"
        missing_position_state = self.mode == "position" and (
            local_map is None or self._planner is None
        )
        if was_active and (pose is None or missing_position_state):
            vehicle.stop(now)
            self.status = "blocked"
            self.reason = "local_state_unavailable"
            self.detail = None
            self._clear_pending_planning()
            return
        if was_active:
            assert pose is not None
            if self.block_for_localization_loss(vehicle, pose, now):
                return
        if advance_result is not None:
            collided = advance_result.collided
            safety_stop = advance_result.reason if advance_result.stopped else None
        elif safety is None:
            collided = vehicle.advance(grid, now)
            safety_stop = None
        else:
            result = safety.advance(vehicle, grid, now, automatic=was_active)
            collided = result.collided
            safety_stop = result.reason if result.stopped else None
        if not was_active:
            return
        if collided:
            self.status = "blocked"
            self.reason = "collision"
            self.detail = None
            self._clear_pending_planning()
            vehicle.stop()
            return
        if safety_stop is not None and not (
            self._planner is not None
            and (
                safety_stop == "safety_obstacle"
                or (
                    safety_stop == "safety_edge"
                    and safety is not None
                    and pose is not None
                    and local_map is not None
                    and _edge_evidence_is_mapped(safety, pose, local_map)
                )
            )
        ):
            self.status = "blocked"
            self.reason = safety_stop
            self.detail = None
            self._clear_pending_planning()
            vehicle.stop()
            return

        if self.mode == "rotation":
            assert pose is not None and self.yaw_goal_rad is not None
            yaw_error = _wrap_angle(self.yaw_goal_rad - pose.yaw_rad)
            if abs(yaw_error) <= self.yaw_tolerance_rad:
                self.status = "reached"
                self.reason = "yaw_tolerance"
                self.detail = None
                self._clear_pending_planning()
                vehicle.stop()
                return
            angular_rps = max(
                -vehicle.angular_speed,
                min(vehicle.angular_speed, 2 * yaw_error),
            )
            if safety is not None:
                decision = safety.evaluate(
                    vehicle, grid, 0.0, angular_rps, automatic=True
                )
                if decision.state in {"stopped", "fault"}:
                    self.status = "blocked"
                    self.reason = decision.reason
                    self.detail = None
                    self._clear_pending_planning()
                    vehicle.stop()
                    return
                angular_rps = decision.angular_rps
            vehicle.install_drive(0.0, angular_rps, now)
            return

        assert self.goal is not None
        assert pose is not None and local_map is not None and self._planner is not None
        x_m, y_m, yaw_rad = pose.x_m, pose.y_m, pose.yaw_rad
        start_changed = (
            not self._path
            or self._path[0] != self._pose_cell(pose, local_map)
        )
        changes = () if map_delta is None else map_delta.changed_cells
        if self._planning_kind is None and (start_changed or changes):
            self._planning_kind = "goal"
            self._planning_previous_path = list(self._path)
            self._current_waypoint = None
        self._planning_map_changed |= bool(changes)
        if self._planning_kind is not None:
            self._advance_planning(
                pose,
                local_map,
                changes,
                PLANNING_EXPANSIONS_PER_UPDATE,
            )
        if self._planning_kind is not None or self.status != "active":
            vehicle.stop()
            return
        if not self._path:
            self.status = "blocked"
            self.reason = "no_path"
            self._clear_pending_planning()
            vehicle.stop()
            return

        dx, dy = self.goal[0] - x_m, self.goal[1] - y_m
        distance = math.hypot(dx, dy)
        within_approach_limit = (
            self.goal_mode == "exact"
            or self._pose_approach_distance_m(pose)
            <= NEARBY_SAFE_BODY_DISTANCE_M + 1e-9
        )
        if distance <= self.goal_tolerance_m and within_approach_limit:
            if self.goal_mode == "approaching_safe_stop":
                vehicle.stop()
                return
            self.status = "reached"
            if self.goal_mode == "nearby_safe":
                self.reason = "nearby_safe_stop"
                self.detail = self._nearby_detail
            else:
                self.reason = "goal_tolerance"
                self.detail = None
            self._clear_pending_planning()
            vehicle.stop()
            return

        target_x, target_y = self.goal
        target_cell: tuple[int, int] | None = None
        if len(self._path) > 1:
            target_cell = self._path[1]
            target_x = (target_cell[0] + 0.5) * local_map.resolution_m
            target_y = (target_cell[1] + 0.5) * local_map.resolution_m
            if not self._planner.is_segment_passable(
                (x_m, y_m), (target_x, target_y)
            ):
                target_cell = self._planner.best_start_connection(
                    (x_m, y_m),
                    self._path[0],
                )
                if target_cell is None:
                    self.status = "blocked"
                    self.reason = "no_path"
                    self.detail = "start_connection_unsafe"
                    self._current_waypoint = None
                    self._clear_pending_planning()
                    vehicle.stop()
                    return
                target_x = (target_cell[0] + 0.5) * local_map.resolution_m
                target_y = (target_cell[1] + 0.5) * local_map.resolution_m
        self._current_waypoint = target_x, target_y
        target_distance = math.hypot(target_x - x_m, target_y - y_m)
        desired_yaw = math.atan2(target_y - y_m, target_x - x_m)
        heading_error = math.atan2(
            math.sin(desired_yaw - yaw_rad), math.cos(desired_yaw - yaw_rad)
        )
        angular_rps = max(-vehicle.angular_speed, min(vehicle.angular_speed, 2 * heading_error))
        linear_mps = (
            0.0
            if abs(heading_error) > self.turn_in_place_threshold_rad
            else min(vehicle.linear_speed, target_distance)
        )
        if pose is not None and pose.quality == "degraded":
            linear_mps *= DEGRADED_LINEAR_SCALE
        if (
            target_cell is not None
            and local_map.is_unknown(*target_cell)
        ):
            linear_mps *= UNKNOWN_LINEAR_SCALE
        if safety is not None:
            decision = safety.evaluate(
                vehicle, grid, linear_mps, angular_rps, automatic=True
            )
            if decision.state in {"stopped", "fault"}:
                vehicle.stop()
                if (
                    decision.reason == "safety_edge"
                    and safety.observation.edge_point_vehicle_m is not None
                    and map_delta is not None
                    and _edge_evidence_is_mapped(safety, pose, local_map)
                ):
                    return
                self.status = "blocked"
                self.reason = decision.reason
                self.detail = None
                self._clear_pending_planning()
                return
            linear_mps, angular_rps = decision.linear_mps, decision.angular_rps
        vehicle.install_drive(linear_mps, angular_rps, now)

    def _advance_planning(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        changed_cells: tuple[MapCellUpdate, ...],
        expansion_budget: int,
    ) -> None:
        assert (
            self.goal is not None
            and self.requested_goal is not None
            and self._planner is not None
        )
        remaining = expansion_budget
        changes = changed_cells
        if self._planning_kind == "goal":
            before = self._planner.stats["expansions"]
            progress = self._planner.advance_plan(
                self._pose_cell(pose, local_map),
                self._goal_cell(local_map),
                changed_cells=changes,
                expansion_budget=remaining,
            )
            remaining -= self._planner.stats["expansions"] - before
            changes = ()
            if progress.status == "pending":
                return
            if progress.status == "ready":
                assert progress.path is not None
                self._set_path(progress.path)
                if self.goal_mode == "exact":
                    failure = self._execution_goal_failure()
                    if failure is None:
                        self._complete_planning()
                        return
                elif self._safe_execution_goal_remains_safe(
                    pose,
                    local_map,
                    require_observed=False,
                ):
                    confirmed = self._safe_execution_goal_remains_safe(
                        pose,
                        local_map,
                        require_observed=True,
                    )
                    self.goal_mode = (
                        "nearby_safe"
                        if confirmed
                        else "approaching_safe_stop"
                    )
                    self._complete_planning()
                    return
                else:
                    failure = "goal_blocked"
            else:
                failure = self._planner_failure()
                if failure == "start_blocked":
                    self._block_no_path(failure)
                    return
            if self._nearby_detail is None:
                self._nearby_detail = failure
            self._begin_safe_goal_search(local_map)
        if self._planning_kind == "candidate":
            self._advance_safe_candidate(
                pose,
                local_map,
                changes,
                remaining,
            )

    def _begin_safe_goal_search(self, local_map: ObservedGrid) -> None:
        self._planning_kind = "candidate"
        self.goal_mode = "approaching_safe_stop"
        self._safe_candidates = self._build_safe_candidates(local_map)
        self._safe_candidate_index = 0
        self._pending_candidate = None
        self._current_waypoint = None

    def _advance_safe_candidate(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        changed_cells: tuple[MapCellUpdate, ...],
        expansion_budget: int,
    ) -> None:
        assert self._planner is not None
        remaining = expansion_budget
        changes = changed_cells
        while True:
            if self._pending_candidate is None:
                self._pending_candidate = self._next_safe_candidate(
                    pose,
                    local_map,
                    allow_stale_geometry=bool(changes),
                )
                if self._pending_candidate is None:
                    self._block_no_path("nearby_safe_goal_unavailable")
                    return
                self.goal = self._pending_candidate[0]
                self.goal_mode = "approaching_safe_stop"
            if remaining <= 0:
                return
            point, goal_cell, requested_confirmed = self._pending_candidate
            before = self._planner.stats["expansions"]
            progress = self._planner.advance_plan(
                self._pose_cell(pose, local_map),
                goal_cell,
                changed_cells=changes,
                expansion_budget=remaining,
            )
            remaining -= self._planner.stats["expansions"] - before
            changes = ()
            generally_safe = self._candidate_is_safe(
                point,
                goal_cell,
                require_observed=False,
            )
            requested_safe = generally_safe and (
                not requested_confirmed
                or self._candidate_is_safe(
                    point,
                    goal_cell,
                    require_observed=True,
                )
            )
            if not requested_safe or progress.status == "unreachable":
                self._pending_candidate = None
                continue
            if progress.status == "pending":
                return
            assert progress.path is not None
            if self._planner.best_start_connection(
                (pose.x_m, pose.y_m),
                progress.path[0],
            ) is None:
                self._pending_candidate = None
                continue
            confirmed = self._candidate_is_safe(
                point,
                goal_cell,
                require_observed=True,
            )
            self.goal = point
            self.goal_mode = (
                "nearby_safe" if confirmed else "approaching_safe_stop"
            )
            self._set_path(progress.path)
            self._complete_planning()
            return

    def _next_safe_candidate(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        *,
        allow_stale_geometry: bool,
    ) -> SafeCandidate | None:
        assert self._planner is not None
        start = self._pose_cell(pose, local_map)
        while self._safe_candidate_index < len(self._safe_candidates):
            candidate = self._safe_candidates[self._safe_candidate_index]
            self._safe_candidate_index += 1
            point, goal_cell, confirmed = candidate
            if not self._planner.planning_budget_allows(start, goal_cell):
                continue
            if allow_stale_geometry or self._candidate_is_safe(
                point,
                goal_cell,
                require_observed=confirmed,
            ):
                return candidate
        return None

    def _build_safe_candidates(
        self,
        local_map: ObservedGrid,
    ) -> list[SafeCandidate]:
        assert self.requested_goal is not None
        resolution = local_map.resolution_m
        requested_x, requested_y = self.requested_goal
        radius = NEARBY_SAFE_BODY_DISTANCE_M + self._vehicle_radius_m
        sample_step = min(resolution, HARD_STOP_CLEARANCE_M)
        candidates = []
        for offset_x in _axis_offsets(radius, sample_step):
            for offset_y in _axis_offsets(radius, sample_step):
                distance_squared = offset_x**2 + offset_y**2
                if distance_squared > radius**2:
                    continue
                point = requested_x + offset_x, requested_y + offset_y
                goal_cell = (
                    math.floor(point[0] / resolution),
                    math.floor(point[1] / resolution),
                )
                candidates.append(
                    (distance_squared, point[1], point[0], point, goal_cell)
                )
        ordered = [
            (point, goal_cell)
            for _, _, _, point, goal_cell in sorted(candidates)
        ]
        return [
            (point, goal_cell, confirmed)
            for confirmed in (True, False)
            for point, goal_cell in ordered
        ]

    def _candidate_is_safe(
        self,
        point: tuple[float, float],
        goal_cell: tuple[int, int],
        *,
        require_observed: bool,
    ) -> bool:
        assert self._planner is not None
        cell_center = (
            (goal_cell[0] + 0.5) * self._path_resolution_m,
            (goal_cell[1] + 0.5) * self._path_resolution_m,
        )
        return self._planner.is_segment_passable(
            point,
            point,
            extra_clearance_m=HARD_STOP_CLEARANCE_M,
            require_observed=require_observed,
        ) and self._planner.is_segment_passable(
            cell_center,
            point,
            extra_clearance_m=HARD_STOP_CLEARANCE_M,
            require_observed=require_observed,
        )

    def _safe_execution_goal_remains_safe(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        *,
        require_observed: bool,
    ) -> bool:
        assert self.goal is not None and self._planner is not None
        goal_cell = self._goal_cell(local_map)
        return (
            bool(self._path)
            and self._path[-1] == goal_cell
            and self._candidate_is_safe(
                self.goal,
                goal_cell,
                require_observed=require_observed,
            )
            and self._planner.best_start_connection(
                (pose.x_m, pose.y_m),
                self._path[0],
            )
            is not None
        )

    def _execution_goal_failure(self) -> str | None:
        assert self.goal is not None and self._planner is not None
        if (
            self._planner.last_failure == "goal_blocked"
            or not self._planner.is_segment_passable(
                self.goal,
                self.goal,
                extra_clearance_m=HARD_STOP_CLEARANCE_M,
            )
        ):
            return "goal_blocked"
        if self._planner.last_failure in {
            "search_exhausted",
            "path_extraction",
        }:
            return "goal_unreachable"
        return None

    def _planner_failure(self) -> str:
        assert self._planner is not None
        if self._planner.last_failure == "goal_blocked":
            return "goal_blocked"
        if self._planner.last_failure in {"search_exhausted", "path_extraction"}:
            return "goal_unreachable"
        return self._planner.last_failure or "goal_unreachable"

    def _complete_planning(self) -> None:
        if (
            self._planning_map_changed
            and self._path != self._planning_previous_path
        ):
            self._replan_count += 1
        self.status = "active"
        self.reason = None
        self.detail = None
        self._current_waypoint = None
        self._clear_pending_planning()

    def _block_no_path(self, detail: str) -> None:
        self._set_path(None)
        self.status = "blocked"
        self.reason = "no_path"
        self.detail = detail
        self._current_waypoint = None
        self._clear_pending_planning()

    def _clear_pending_planning(self) -> None:
        self._planning_kind = None
        self._planning_map_changed = False
        self._planning_previous_path = list(self._path)
        self._safe_candidates = []
        self._safe_candidate_index = 0
        self._pending_candidate = None

    def _approach_distance_m(self) -> float | None:
        if self.requested_goal is None or self.goal is None:
            return None
        return max(
            0.0,
            math.dist(self.requested_goal, self.goal) - self._vehicle_radius_m,
        )

    def _pose_approach_distance_m(self, pose: PoseEstimate) -> float:
        assert self.requested_goal is not None
        return max(
            0.0,
            math.dist(self.requested_goal, (pose.x_m, pose.y_m))
            - self._vehicle_radius_m,
        )

    def _set_path(self, path: list[tuple[int, int]] | None) -> None:
        new_path = [] if path is None else path
        self.detail = (
            None
            if path is not None or self._planner is None
            else self._planner.last_failure
        )
        if new_path != self._path:
            self._path = new_path
            self._path_revision += 1

    def _goal_cell(self, local_map: ObservedGrid) -> tuple[int, int]:
        assert self.goal is not None
        return (
            math.floor(self.goal[0] / local_map.resolution_m),
            math.floor(self.goal[1] / local_map.resolution_m),
        )

    @staticmethod
    def _pose_cell(
        pose: PoseEstimate, local_map: ObservedGrid
    ) -> tuple[int, int]:
        return (
            math.floor(pose.x_m / local_map.resolution_m),
            math.floor(pose.y_m / local_map.resolution_m),
        )


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _axis_offsets(radius: float, max_step: float) -> tuple[float, ...]:
    intervals = max(1, math.ceil(2 * radius / max_step))
    return tuple(
        -radius + 2 * radius * index / intervals
        for index in range(intervals + 1)
    )


def _edge_evidence_is_mapped(
    safety: LocalSafetyRuntime,
    pose: PoseEstimate,
    local_map: ObservedGrid,
) -> bool:
    point = safety.observation.edge_point_vehicle_m
    if point is None:
        return False
    cosine, sine = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    x_m = pose.x_m + cosine * point[0] - sine * point[1]
    y_m = pose.y_m + sine * point[0] + cosine * point[1]
    return local_map.is_forbidden(
        math.floor(x_m / local_map.resolution_m),
        math.floor(y_m / local_map.resolution_m),
    )
