"""End-to-end finite-view navigation through RobotController."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    AutoState,
    GotoMission,
    ModeAction,
    ModeCommand,
    RobotController,
)
from mockvehicle2d.local_state import (
    AnchorSpec,
    AnchoredLocalState,
    OdometryConfig,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.server import VehicleRuntime
from mockvehicle2d.vehicle import Vehicle


def make_runtime(
    grid: MapGrid,
    *,
    x_m: float = 2.5,
    y_m: float = 5.5,
    radius_m: float = 0.5,
    map_resolution_m: float = 1.0,
) -> VehicleRuntime:
    vehicle = Vehicle(
        x_m,
        y_m,
        radius=radius_m,
        command_timeout=1.0,
        now=0.0,
    )
    anchor = AnchorSpec("runtime-anchor", vehicle.x, vehicle.y, 0.0)
    return VehicleRuntime(
        [],
        grid,
        vehicle,
        RobotController(mission_capacity=4),
        LocalSafetyRuntime(),
        AnchoredLocalState(
            anchor,
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            odometry_config=OdometryConfig(),
            timestamp=0.0,
            map_resolution_m=map_resolution_m,
        ),
    )


def start_missions(runtime: VehicleRuntime, *missions: GotoMission) -> None:
    mode, _ = runtime.handle_command(
        ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        monotonic_now=runtime.vehicle.last_update,
    )
    pushed, _ = runtime.handle_command(
        AutoCommand(2, AutoAction.PUSH, tuple(missions)),
        monotonic_now=runtime.vehicle.last_update,
    )
    assert mode.accepted and pushed.accepted


class TestControllerNavigation(unittest.TestCase):
    def test_unknown_next_cell_is_traversable_but_speed_limited(self) -> None:
        anchor = AnchorSpec("unknown-anchor", 10.0, 10.0, 0.0)
        local_map = ObservedGrid(anchor)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        grid = MapGrid.from_wall_set(64, 64, set())
        vehicle = Vehicle(10.0, 10.0, now=0.0)
        controller = RobotController()
        safety = LocalSafetyRuntime()
        controller.handle(
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.handle(
            AutoCommand(
                2,
                AutoAction.PUSH,
                (GotoMission("unknown", "global_map", 14.0, 10.0, 2),),
            ),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.tick(
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=0.0,
        )
        linear_mps, _ = vehicle.body_velocities()
        self.assertGreater(linear_mps, 0.0)
        self.assertLessEqual(linear_mps, vehicle.linear_speed * 0.4)
        self.assertEqual(controller.navigation.status, "active")

    def test_hidden_route_obstacle_is_discovered_and_incrementally_replanned(self) -> None:
        runtime = make_runtime(
            MapGrid.from_wall_set(32, 12, {(16, 5)})
        )
        start_missions(
            runtime,
            GotoMission("detour", "global_map", 26.5, 5.5, 2),
        )

        terminal_events = []
        first_replan_x = None
        for tick in range(1, 900):
            frame = runtime.update(tick / 6, tick / 6)
            terminal_events.extend(frame.events)
            if (
                runtime.controller.navigation.snapshot()["replan_count"]
                and first_replan_x is None
            ):
                first_replan_x = runtime.vehicle.x
            if runtime.controller.auto_state is AutoState.IDLE:
                break

        self.assertFalse(runtime.vehicle.collision)
        self.assertIsNotNone(first_replan_x)
        self.assertLess(first_replan_x, 16 - runtime.vehicle.radius)
        self.assertEqual(
            runtime.controller.navigation.status,
            "reached",
            runtime.controller.snapshot(),
        )
        self.assertGreaterEqual(
            runtime.controller.navigation.snapshot()["replan_count"],
            1,
        )
        self.assertIn("reached", [event.status for event in terminal_events])

    def test_obstacle_goal_finishes_at_confirmed_safe_stop_within_one_metre(self) -> None:
        grid = MapGrid.from_wall_set(16, 12, {(8, 5)})
        runtime = make_runtime(grid, map_resolution_m=0.5)
        requested = (8.5, 5.5)
        start_missions(
            runtime,
            GotoMission("blocked-goal", "global_map", *requested, 2),
        )

        reached = None
        for tick in range(1, 900):
            frame = runtime.update(tick / 6, tick / 6)
            reached = next(
                (event for event in frame.events if event.status == "reached"),
                reached,
            )
            if reached is not None:
                break

        self.assertIsNotNone(reached, runtime.controller.snapshot())
        self.assertFalse(runtime.vehicle.collision)
        self.assertEqual(runtime.controller.navigation.goal_mode, "nearby_safe")
        self.assertEqual(runtime.controller.navigation.reason, "nearby_safe_stop")
        self.assertLessEqual(
            runtime.controller.navigation.snapshot()["approach_distance_m"],
            1.0,
        )

    def test_planning_work_is_sliced_and_out_of_range_search_terminates(self) -> None:
        anchor = AnchorSpec("budget-anchor", 10.0, 10.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        local_map = ObservedGrid(anchor)
        grid = MapGrid.from_wall_set(512, 512, set())
        vehicle = Vehicle(10.0, 10.0, now=0.0)
        safety = LocalSafetyRuntime()
        controller = RobotController()
        controller.handle(
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.handle(
            AutoCommand(
                2,
                AutoAction.PUSH,
                (GotoMission("sliced", "global_map", 14.0, 10.0, 2),),
            ),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        expansion_counts = []
        with patch(
            "mockvehicle2d.navigation.PLANNING_EXPANSIONS_PER_UPDATE",
            1,
        ):
            previous = 0
            for _ in range(256):
                controller.tick(
                    vehicle=vehicle,
                    grid=grid,
                    safety=safety,
                    anchor=anchor,
                    pose=pose,
                    local_map=local_map,
                    map_delta=None,
                    advance_result=SafetyAdvanceResult(),
                    now=0.0,
                )
                current = controller.navigation.snapshot()["planner_stats"][
                    "expansions"
                ]
                expansion_counts.append(current - previous)
                previous = current
                if vehicle.body_velocities() != (0.0, 0.0):
                    break
        self.assertTrue(expansion_counts)
        self.assertLessEqual(max(expansion_counts), 1)
        self.assertGreater(
            vehicle.body_velocities()[0],
            0.0,
            controller.snapshot(),
        )

        controller.handle(
            AutoCommand(3, AutoAction.CANCEL_ALL),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.handle(
            AutoCommand(
                4,
                AutoAction.PUSH,
                (GotoMission("too-far", "global_map", 400.0, 10.0, 4),),
            ),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        inspection_deltas = []
        previous = 0
        for _ in range(4):
            controller.tick(
                vehicle=vehicle,
                grid=grid,
                safety=safety,
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                map_delta=None,
                advance_result=SafetyAdvanceResult(),
                now=0.0,
            )
            current = controller.navigation.snapshot()["planner_stats"][
                "candidate_inspections"
            ]
            inspection_deltas.append(current - previous)
            previous = current
            if controller.auto_state is AutoState.BLOCKED:
                break
        self.assertLessEqual(max(inspection_deltas), 256)
        self.assertEqual(controller.auto_state, AutoState.BLOCKED)
        self.assertEqual(vehicle.body_velocities(), (0.0, 0.0))
        too_far_events = [
            event
            for event in controller.drain_events()
            if event.mission.mission_id == "too-far"
        ]
        self.assertEqual(
            [(event.status, event.reason, event.detail) for event in too_far_events],
            [
                ("queued", None, None),
                ("blocked", "invalid_goal", "goal exceeds maximum distance"),
            ],
        )

    def test_active_navigation_fails_closed_when_localization_is_lost(self) -> None:
        anchor = AnchorSpec("lost-anchor", 10.0, 10.0, 0.0)
        local_map = ObservedGrid(anchor)
        nominal = PoseEstimate(
            anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        lost = PoseEstimate(
            anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "lost",
            0.1,
            1,
        )
        grid = MapGrid.from_wall_set(64, 64, set())
        vehicle = Vehicle(10.0, 10.0, now=0.0)
        safety = LocalSafetyRuntime()
        controller = RobotController()
        controller.handle(
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.handle(
            AutoCommand(
                2,
                AutoAction.PUSH,
                (GotoMission("lost", "global_map", 14.0, 10.0, 2),),
            ),
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            now=0.0,
        )
        controller.tick(
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=nominal,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=0.0,
        )
        self.assertNotEqual(vehicle.body_velocities(), (0.0, 0.0))

        controller.tick(
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=lost,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=0.0,
        )
        self.assertEqual(controller.auto_state, AutoState.BLOCKED)
        self.assertEqual(controller.navigation.reason, "localization_lost")
        self.assertEqual(vehicle.body_velocities(), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
