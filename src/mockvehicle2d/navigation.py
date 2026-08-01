"""Finite-view D* Lite go-to-goal controller."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import (
    AUTOMATIC_MINIMUM_CLEARANCE_M,
    LocalSafetyRuntime,
    MAX_TRANSLATION_STEP_M,
    SafetyAdvanceResult,
)

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
CANDIDATE_INSPECTIONS_PER_UPDATE = 256
MAX_CANDIDATE_INSPECTIONS_PER_MISSION = 256
SafeCandidate = tuple[tuple[float, float], tuple[int, int]]


class GotoController:
    """Plan from one estimated pose and observed map; never read simulator pose truth."""

    goal_tolerance_m = 0.1
    turn_in_place_threshold_rad = math.radians(20)

    def __init__(self) -> None:
        self.status = "idle"
        self.requested_goal: tuple[float, float] | None = None
        self.goal: tuple[float, float] | None = None
        self.reported_goal: tuple[float, float] | None = None
        self.goal_mode: str | None = None
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
        self._goal_access_cell: tuple[int, int] | None = None
        self._final_approach = False
        self._planning_kind: str | None = None
        self._planning_map_changed = False
        self._planning_previous_path: list[tuple[int, int]] = []
        self._safe_candidates: list[SafeCandidate] = []
        self._safe_candidate_index = 0
        self._pending_candidate: SafeCandidate | None = None
        self._candidate_inspections = 0
        self._skip_goal_connected_candidates = False
        self._waiting_safe_stop_goal: tuple[float, float] | None = None

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
        self.requested_goal = (x_m, y_m)
        self.goal = self.requested_goal
        self.reported_goal = self.goal if reported_goal is None else reported_goal
        self.goal_mode = "exact"
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
        self._goal_access_cell = None
        self._final_approach = False
        self._planning_kind = "goal"
        self._planning_map_changed = False
        self._planning_previous_path = []
        self._safe_candidates = []
        self._safe_candidate_index = 0
        self._pending_candidate = None
        self._candidate_inspections = 0
        self._skip_goal_connected_candidates = False
        self._waiting_safe_stop_goal = None
        self._planner = DStarLitePlanner(
            local_map,
            vehicle_radius_m=vehicle_radius_m,
            hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        )
        self._planner.validate_plan_request(
            self._pose_cell(pose, local_map),
            self._goal_cell(local_map),
        )

    def cancel(self, reason: str) -> None:
        if self.status in {"active", "blocked"}:
            self._final_approach = False
            self.status = "cancelled"
            self.reason = reason
            self.detail = None
            self._waiting_safe_stop_goal = None
            self._clear_pending_planning()

    def block(self, reason: str, detail: str | None = None) -> None:
        self._final_approach = False
        self.status = "blocked"
        self.reason = reason
        self.detail = detail
        self._waiting_safe_stop_goal = None
        self._clear_pending_planning()

    def block_for_localization_loss(self, pose: PoseEstimate) -> bool:
        if self.status != "active" or pose.quality != "lost":
            return False
        self.block("localization_lost")
        return True

    def finish_nearby_safe_stop(
        self,
        pose: PoseEstimate,
        detail: str | None = None,
    ) -> bool:
        if (
            self.status != "active"
            or self.requested_goal is None
            or pose.quality == "lost"
            or self._pose_approach_distance_m(pose)
            + _radial_uncertainty_m(pose)
            > NEARBY_SAFE_BODY_DISTANCE_M + 1e-9
        ):
            return False
        self.goal = pose.x_m, pose.y_m
        self._goal_access_cell = None
        self._final_approach = False
        self.goal_mode = "nearby_safe"
        self.status = "reached"
        self.reason = "nearby_safe_stop"
        self.detail = self._nearby_detail or detail
        self._current_waypoint = None
        self._waiting_safe_stop_goal = None
        self._clear_pending_planning()
        return True

    def unmapped_edge_evidence_cell(
        self,
        safety: LocalSafetyRuntime,
        pose: PoseEstimate,
        local_map: ObservedGrid,
    ) -> tuple[int, int] | None:
        if self.status != "active" or self._planner is None:
            return None
        cell = _edge_evidence_cell(safety, pose, local_map)
        if cell is None or local_map.is_forbidden(*cell):
            return None
        return cell

    def snapshot(self) -> dict[str, object]:
        goal = (
            None
            if self.reported_goal is None
            else {"x_m": self.reported_goal[0], "y_m": self.reported_goal[1]}
        )
        snapshot: dict[str, object] = {
            "status": self.status,
            "mode": "position",
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
            "final_approach": self._final_approach,
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
            "planner_stats": {
                **self._planner.stats,
                "candidate_inspections": self._candidate_inspections,
            },
        })
        return snapshot

    def update(
        self,
        *,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        max_linear_mps: float,
        max_angular_rps: float,
        advance_result: SafetyAdvanceResult | None = None,
        map_delta: LocalMapDelta | None = None,
        safety: LocalSafetyRuntime | None = None,
    ) -> tuple[float, float]:
        """Advance navigation state and return one desired body-velocity setpoint.

        This controller never writes the simulated actuator.  RobotController is
        the single owner that applies the returned setpoint through the safety
        runtime.
        """
        limits = (max_linear_mps, max_angular_rps)
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in limits)
            or not all(math.isfinite(value) and value > 0 for value in limits)
        ):
            raise ValueError("navigation velocity limits must be finite and positive")

        was_active = self.status == "active"
        if was_active and self._planner is None:
            self.block("local_state_unavailable")
            return 0.0, 0.0
        if was_active and self.block_for_localization_loss(pose):
            return 0.0, 0.0
        if advance_result is not None:
            collided = advance_result.collided
            safety_stop = advance_result.reason if advance_result.stopped else None
        else:
            collided = False
            safety_stop = None
        if not was_active:
            return 0.0, 0.0
        if collided:
            self.block("collision")
            return 0.0, 0.0
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
            self.block(safety_stop)
            return 0.0, 0.0

        assert self.goal is not None
        assert self._planner is not None
        x_m, y_m, yaw_rad = pose.x_m, pose.y_m, pose.yaw_rad
        if map_delta is not None and map_delta.peer_forbidden_cells is not None:
            self._planner.set_peer_forbidden_cells(
                map_delta.peer_forbidden_cells
            )
        changes = () if map_delta is None else map_delta.changed_cells
        if self._final_approach:
            if changes:
                self._planner.observe_changes(changes)
                self._planning_map_changed = True
            if not self._final_approach_segment_is_safe(pose):
                self._final_approach = False
                self._begin_safe_goal_search(
                    pose,
                    local_map,
                    skip_goal_connected=False,
                )
                return 0.0, 0.0
            changes = ()
        else:
            start_changed = (
                not self._path
                or self._path[0] != self._pose_cell(pose, local_map)
            )
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
            return 0.0, 0.0
        if not self._path:
            self._block_no_path(self.detail)
            return 0.0, 0.0

        if self._can_start_final_approach(pose, local_map):
            self._final_approach = True

        dx, dy = self.goal[0] - x_m, self.goal[1] - y_m
        distance = math.hypot(dx, dy)
        within_approach_limit = (
            self.goal_mode == "exact"
            or self._pose_approach_distance_m(pose)
            + _radial_uncertainty_m(pose)
            <= NEARBY_SAFE_BODY_DISTANCE_M + 1e-9
        )
        if distance <= self.goal_tolerance_m and within_approach_limit:
            if self.goal_mode == "approaching_safe_stop":
                if self._candidate_is_safe(
                    self.goal,
                    self._goal_cell(local_map),
                    require_observed=True,
                ):
                    self.goal_mode = "nearby_safe"
                elif self._waiting_safe_stop_goal != self.goal:
                    self._waiting_safe_stop_goal = self.goal
                    return 0.0, 0.0
                else:
                    self._block_no_path("nearby_safe_goal_unconfirmed")
                    return 0.0, 0.0
            self._final_approach = False
            self.status = "reached"
            if self.goal_mode == "nearby_safe":
                self.reason = "nearby_safe_stop"
                self.detail = self._nearby_detail
            else:
                self.reason = "goal_tolerance"
                self.detail = None
            self._waiting_safe_stop_goal = None
            self._clear_pending_planning()
            return 0.0, 0.0

        target_x, target_y = self.goal
        target_cell: tuple[int, int] | None = None
        if not self._final_approach and len(self._path) > 1:
            target_cell = self._path[1]
            target_x = (target_cell[0] + 0.5) * local_map.resolution_m
            target_y = (target_cell[1] + 0.5) * local_map.resolution_m
            if not self._planner.is_segment_passable(
                (x_m, y_m),
                (target_x, target_y),
                extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
            ) and (
                safety_stop is not None
                or not self._planner.is_segment_passable(
                    (x_m, y_m),
                    (target_x, target_y),
                )
            ):
                target_cell = self._planner.best_start_connection(
                    (x_m, y_m),
                    self._path[0],
                )
                if target_cell is None:
                    self._block_no_path("start_connection_unsafe")
                    return 0.0, 0.0
                target_x = (target_cell[0] + 0.5) * local_map.resolution_m
                target_y = (target_cell[1] + 0.5) * local_map.resolution_m
        self._current_waypoint = target_x, target_y
        target_distance = math.hypot(target_x - x_m, target_y - y_m)
        desired_yaw = math.atan2(target_y - y_m, target_x - x_m)
        heading_error = math.atan2(
            math.sin(desired_yaw - yaw_rad), math.cos(desired_yaw - yaw_rad)
        )
        angular_rps = max(
            -max_angular_rps,
            min(max_angular_rps, 2 * heading_error),
        )
        linear_mps = (
            0.0
            if abs(heading_error) > self.turn_in_place_threshold_rad
            else min(max_linear_mps, target_distance)
        )
        if pose is not None and pose.quality == "degraded":
            linear_mps *= DEGRADED_LINEAR_SCALE
        if (
            target_cell is not None
            and local_map.is_unknown(*target_cell)
        ):
            linear_mps *= UNKNOWN_LINEAR_SCALE
        return linear_mps, angular_rps

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
                start_position_m=(pose.x_m, pose.y_m),
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
                if failure in {"start_blocked", "expansion_limit"}:
                    self._block_no_path(failure)
                    return
            if self._nearby_detail is None:
                self._nearby_detail = failure
            self._begin_safe_goal_search(
                pose,
                local_map,
                skip_goal_connected=failure == "goal_unreachable",
            )
        if self._planning_kind == "candidate":
            self._advance_safe_candidate(
                pose,
                local_map,
                changes,
                remaining,
            )

    def _begin_safe_goal_search(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        *,
        skip_goal_connected: bool,
    ) -> None:
        self._final_approach = False
        self._planning_kind = "candidate"
        self.goal_mode = "approaching_safe_stop"
        self._goal_access_cell = None
        self._safe_candidates = self._build_safe_candidates(pose, local_map)
        self._skip_goal_connected_candidates = skip_goal_connected
        self._safe_candidate_index = 0
        self._pending_candidate = None
        self._current_waypoint = None
        self._waiting_safe_stop_goal = None

    def _advance_safe_candidate(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        changed_cells: tuple[MapCellUpdate, ...],
        expansion_budget: int,
    ) -> None:
        assert self._planner is not None
        remaining = expansion_budget
        inspections_remaining = CANDIDATE_INSPECTIONS_PER_UPDATE
        changes = changed_cells
        while inspections_remaining > 0:
            if self._pending_candidate is None:
                candidate, inspected = self._next_safe_candidate(
                    pose,
                    local_map,
                    allow_stale_geometry=bool(changes),
                    inspection_budget=inspections_remaining,
                )
                inspections_remaining -= inspected
                self._pending_candidate = candidate
                if candidate is None:
                    if self._safe_candidate_index == len(self._safe_candidates):
                        self._block_no_path("nearby_safe_goal_unavailable")
                    return
                if self.goal != self._pending_candidate[0]:
                    self._waiting_safe_stop_goal = None
                self.goal = self._pending_candidate[0]
                self.goal_mode = "approaching_safe_stop"
            if remaining <= 0:
                return
            point, goal_cell = self._pending_candidate
            before = self._planner.stats["expansions"]
            progress = self._planner.advance_plan(
                self._pose_cell(pose, local_map),
                goal_cell,
                changed_cells=changes,
                start_position_m=(pose.x_m, pose.y_m),
                expansion_budget=remaining,
            )
            remaining -= self._planner.stats["expansions"] - before
            changes = ()
            requested_safe = self._candidate_is_safe(
                point,
                goal_cell,
                require_observed=False,
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
            self._goal_access_cell = goal_cell
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
        inspection_budget: int,
    ) -> tuple[SafeCandidate | None, int]:
        assert self._planner is not None
        start = self._pose_cell(pose, local_map)
        inspected = 0
        while (
            self._safe_candidate_index < len(self._safe_candidates)
            and inspected < inspection_budget
            and self._candidate_inspections
            < MAX_CANDIDATE_INSPECTIONS_PER_MISSION
        ):
            candidate = self._safe_candidates[self._safe_candidate_index]
            self._safe_candidate_index += 1
            inspected += 1
            self._candidate_inspections += 1
            point, goal_cell = candidate
            if not self._planner.planning_budget_allows(start, goal_cell):
                continue
            if (
                self._skip_goal_connected_candidates
                and self.requested_goal is not None
                and self._planner.is_segment_passable(
                    point,
                    self.requested_goal,
                    extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
                )
            ):
                continue
            if allow_stale_geometry or self._candidate_is_safe(
                point,
                goal_cell,
                require_observed=False,
            ):
                return candidate, inspected
        if self._candidate_inspections >= MAX_CANDIDATE_INSPECTIONS_PER_MISSION:
            self._safe_candidate_index = len(self._safe_candidates)
        return None, inspected

    def _build_safe_candidates(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
    ) -> list[SafeCandidate]:
        assert self.requested_goal is not None
        resolution = local_map.resolution_m
        requested_x, requested_y = self.requested_goal
        candidate_body_distance = (
            NEARBY_SAFE_BODY_DISTANCE_M
            - MAX_TRANSLATION_STEP_M
            - _radial_uncertainty_m(pose)
        )
        if candidate_body_distance < 0:
            return []
        radius = candidate_body_distance + self._vehicle_radius_m
        candidates = []
        min_gx = math.floor((requested_x - radius) / resolution)
        max_gx = math.floor((requested_x + radius) / resolution)
        min_gy = math.floor((requested_y - radius) / resolution)
        max_gy = math.floor((requested_y + radius) / resolution)
        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                closest_x = max(gx * resolution, min(requested_x, (gx + 1) * resolution))
                closest_y = max(gy * resolution, min(requested_y, (gy + 1) * resolution))
                if math.hypot(closest_x - requested_x, closest_y - requested_y) > radius:
                    continue
                center_x = (gx + 0.5) * resolution
                center_y = (gy + 0.5) * resolution
                point_x = max(gx * resolution, min(pose.x_m, (gx + 1) * resolution))
                point_y = max(gy * resolution, min(pose.y_m, (gy + 1) * resolution))
                dx, dy = point_x - requested_x, point_y - requested_y
                point_distance = math.hypot(dx, dy)
                if point_distance <= radius:
                    point = point_x, point_y
                else:
                    scale = radius / point_distance
                    point = requested_x + dx * scale, requested_y + dy * scale
                if not (
                    gx * resolution - 1e-9 <= point[0] <= (gx + 1) * resolution + 1e-9
                    and gy * resolution - 1e-9
                    <= point[1]
                    <= (gy + 1) * resolution + 1e-9
                ):
                    continue
                candidates.append(
                    (
                        (point[0] - pose.x_m) ** 2
                        + (point[1] - pose.y_m) ** 2,
                        (point[0] - requested_x) ** 2
                        + (point[1] - requested_y) ** 2,
                        point[1],
                        point[0],
                        (center_x - point[0]) ** 2
                        + (center_y - point[1]) ** 2,
                        gy,
                        gx,
                        point,
                        (gx, gy),
                    )
                )
        return [
            (point, goal_cell)
            for _, _, _, _, _, _, _, point, goal_cell in sorted(candidates)
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
            extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
            require_observed=require_observed,
        ) and self._planner.is_segment_passable(
            cell_center,
            point,
            extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
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

    def _can_start_final_approach(
        self,
        pose: PoseEstimate,
        local_map: ObservedGrid,
    ) -> bool:
        return (
            not self._final_approach
            and self.goal_mode in {"nearby_safe", "approaching_safe_stop"}
            and len(self._path) == 1
            and self._path[0] == self._pose_cell(pose, local_map)
            and self._final_approach_segment_is_safe(pose)
        )

    def _final_approach_segment_is_safe(self, pose: PoseEstimate) -> bool:
        assert self.goal is not None and self._planner is not None
        return self._planner.is_segment_passable(
            (pose.x_m, pose.y_m),
            self.goal,
            extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        )

    def _execution_goal_failure(self) -> str | None:
        assert self.goal is not None and self._planner is not None
        if (
            self._planner.last_failure == "goal_blocked"
            or not self._planner.is_segment_passable(
                self.goal,
                self.goal,
                extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
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

    def _block_no_path(self, detail: str | None) -> None:
        self._set_path(None)
        self._current_waypoint = None
        self.block("no_path", detail)

    def _clear_pending_planning(self) -> None:
        self._planning_kind = None
        self._planning_map_changed = False
        self._planning_previous_path = list(self._path)
        self._safe_candidates = []
        self._safe_candidate_index = 0
        self._pending_candidate = None
        self._skip_goal_connected_candidates = False

    def _approach_distance_m(self) -> float | None:
        if self.requested_goal is None or self.goal is None:
            return None
        return self._point_approach_distance_m(self.goal)

    def _pose_approach_distance_m(self, pose: PoseEstimate) -> float:
        return self._point_approach_distance_m((pose.x_m, pose.y_m))

    def _point_approach_distance_m(self, point: tuple[float, float]) -> float:
        assert self.requested_goal is not None
        return max(
            0.0,
            math.dist(self.requested_goal, point) - self._vehicle_radius_m,
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
        if self._goal_access_cell is not None:
            return self._goal_access_cell
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

def _radial_uncertainty_m(pose: PoseEstimate) -> float:
    return math.sqrt(pose.covariance[0] + pose.covariance[1])


def _edge_evidence_is_mapped(
    safety: LocalSafetyRuntime,
    pose: PoseEstimate,
    local_map: ObservedGrid,
) -> bool:
    cell = _edge_evidence_cell(safety, pose, local_map)
    return cell is not None and local_map.is_forbidden(*cell)


def _edge_evidence_cell(
    safety: LocalSafetyRuntime,
    pose: PoseEstimate,
    local_map: ObservedGrid,
) -> tuple[int, int] | None:
    point = safety.observation.edge_point_vehicle_m
    if point is None:
        return None
    cosine, sine = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    x_m = pose.x_m + cosine * point[0] - sine * point[1]
    y_m = pose.y_m + sine * point[0] + cosine * point[1]
    return (
        math.floor(x_m / local_map.resolution_m),
        math.floor(y_m / local_map.resolution_m),
    )
