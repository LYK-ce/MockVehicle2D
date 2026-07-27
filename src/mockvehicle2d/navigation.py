"""Finite-view D* Lite go-to-goal controller."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.vehicle import Vehicle

if TYPE_CHECKING:
    from mockvehicle2d.local_state import LocalMapDelta, ObservedGrid, PoseEstimate


DEGRADED_LINEAR_SCALE = 0.5
UNKNOWN_LINEAR_SCALE = 0.4
MAX_REPORTED_PATH_CELLS = 64


class GotoController:
    """Plan from one estimated pose and observed map; never read simulator pose truth."""

    goal_tolerance_m = 0.1
    yaw_tolerance_rad = math.radians(2)
    turn_in_place_threshold_rad = math.radians(20)

    def __init__(self) -> None:
        self.status = "idle"
        self.mode = "position"
        self.goal: tuple[float, float] | None = None
        self.reported_goal: tuple[float, float] | None = None
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
        self.goal = (x_m, y_m)
        self.reported_goal = self.goal if reported_goal is None else reported_goal
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
        self._planner = DStarLitePlanner(
            local_map, vehicle_radius_m=vehicle_radius_m
        )
        self._set_path(
            self._planner.plan(
                self._pose_cell(pose, local_map),
                self._goal_cell(local_map),
            )
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
        self.goal = None
        self.reported_goal = None
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

    def cancel(self, reason: str) -> None:
        if self.status == "active":
            self.status = "cancelled"
            self.reason = reason
            self.detail = None

    def block_for_localization_loss(
        self, vehicle: Vehicle, pose: PoseEstimate, now: float | None = None
    ) -> bool:
        if self.status != "active" or pose.quality != "lost":
            return False
        vehicle.stop(now)
        self.status = "blocked"
        self.reason = "localization_lost"
        self.detail = None
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
            "reason": self.reason,
            "detail": self.detail,
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
        old_path = self._path
        new_path = self._planner.plan(
            self._pose_cell(pose, local_map),
            self._goal_cell(local_map),
            changed_cells=() if map_delta is None else map_delta.changed_cells,
        )
        self._set_path(new_path)
        if map_delta is not None and map_delta.changed_cells and self._path != old_path:
            self._replan_count += 1

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
            vehicle.stop()
            return

        if self.mode == "rotation":
            assert pose is not None and self.yaw_goal_rad is not None
            yaw_error = _wrap_angle(self.yaw_goal_rad - pose.yaw_rad)
            if abs(yaw_error) <= self.yaw_tolerance_rad:
                self.status = "reached"
                self.reason = "yaw_tolerance"
                self.detail = None
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
        if start_changed or (
            map_delta is not None and map_delta.changed_cells
        ):
            self.replan(pose, map_delta, local_map)
        if not self._path:
            self.status = "blocked"
            self.reason = "no_path"
            vehicle.stop()
            return

        dx, dy = self.goal[0] - x_m, self.goal[1] - y_m
        distance = math.hypot(dx, dy)
        if distance <= self.goal_tolerance_m:
            self.status = "reached"
            self.reason = "goal_tolerance"
            self.detail = None
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
                return
            linear_mps, angular_rps = decision.linear_mps, decision.angular_rps
        vehicle.install_drive(linear_mps, angular_rps, now)

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
