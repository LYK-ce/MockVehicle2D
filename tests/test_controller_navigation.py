"""End-to-end finite-view navigation through RobotController."""

from pathlib import Path
import math
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
    FREE,
    LocalMapDelta,
    MapCellUpdate,
    OCCUPIED,
    OdometryConfig,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import (
    LocalSafetyRuntime,
    SafetyAdvanceResult,
    SafetyDecision,
    SafetyObservation,
)
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
    mode = runtime.handle_command(
        ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        monotonic_now=runtime.vehicle.last_update,
    )
    pushed = runtime.handle_command(
        AutoCommand(2, AutoAction.PUSH, tuple(missions)),
        monotonic_now=runtime.vehicle.last_update,
    )
    assert mode.accepted and pushed.accepted


class TestControllerNavigation(unittest.TestCase):
    def test_nearby_safe_candidates_prefer_the_current_side(self) -> None:
        local_map = ObservedGrid(
            AnchorSpec("candidate-anchor", 0.0, 0.0, 0.0),
            resolution_m=0.5,
        )
        pose = PoseEstimate(
            "candidate-anchor",
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        navigation = RobotController().navigation
        navigation.start(4.0, 0.0, local_map=local_map, pose=pose)

        first = navigation._build_safe_candidates(pose, local_map)
        second = navigation._build_safe_candidates(pose, local_map)

        self.assertEqual(first, second)
        self.assertEqual(
            len(first),
            len({access_cell for _, access_cell in first}),
        )
        self.assertLessEqual(len(first), 64)
        self.assertTrue(
            all(
                navigation._point_approach_distance_m(point) <= 0.95 + 1e-9
                for point, _ in first
            )
        )
        self.assertEqual(
            {
                access_cell
                for point, access_cell in first
                if math.isclose(point[0], 2.55) and math.isclose(point[1], 0.0)
            },
            {(5, -1), (5, 0)},
        )
        self.assertEqual(
            math.dist((pose.x_m, pose.y_m), first[0][0]),
            min(
                math.dist((pose.x_m, pose.y_m), point)
                for point, _ in first
            ),
        )

    def test_nearby_completion_accounts_for_position_uncertainty(self) -> None:
        local_map = ObservedGrid(
            AnchorSpec("uncertain-anchor", 0.0, 0.0, 0.0),
            resolution_m=0.5,
        )
        uncertain_pose = PoseEstimate(
            "uncertain-anchor",
            0.6,
            0.0,
            0.0,
            (0.04, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        navigation = RobotController().navigation
        navigation.start(2.0, 0.0, local_map=local_map, pose=uncertain_pose)

        candidates = navigation._build_safe_candidates(
            uncertain_pose,
            local_map,
        )

        self.assertTrue(candidates)
        self.assertTrue(
            all(
                navigation._point_approach_distance_m(point)
                <= 0.75 + 1e-9
                for point, _ in candidates
            )
        )
        self.assertFalse(navigation.finish_nearby_safe_stop(uncertain_pose))
        certain_pose = PoseEstimate(
            "uncertain-anchor",
            0.6,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            1,
        )
        self.assertTrue(navigation.finish_nearby_safe_stop(certain_pose))

    def test_unconfirmed_safe_stop_blocks_after_one_observation_tick(self) -> None:
        local_map = ObservedGrid(
            AnchorSpec("unconfirmed-anchor", 0.0, 0.0, 0.0),
            resolution_m=1.0,
        )
        pose = PoseEstimate(
            "unconfirmed-anchor",
            1.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        navigation = RobotController().navigation
        navigation.start(2.5, 0.5, local_map=local_map, pose=pose)
        navigation._clear_pending_planning()
        navigation.goal = (pose.x_m, pose.y_m)
        navigation.goal_mode = "approaching_safe_stop"
        navigation._goal_access_cell = (1, 0)
        navigation._set_path([(1, 0)])

        first = navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=1.0,
            max_angular_rps=1.0,
        )
        second = navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=1.0,
            max_angular_rps=1.0,
        )

        self.assertEqual(first, (0.0, 0.0))
        self.assertEqual(second, (0.0, 0.0))
        self.assertEqual(navigation.status, "blocked")
        self.assertEqual(navigation.reason, "no_path")
        self.assertEqual(navigation.detail, "nearby_safe_goal_unconfirmed")

    def test_observation_confirms_safe_stop_on_the_next_tick(self) -> None:
        local_map = ObservedGrid(
            AnchorSpec("confirmed-anchor", 0.0, 0.0, 0.0),
            resolution_m=1.0,
        )
        pose = PoseEstimate(
            "confirmed-anchor",
            1.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        navigation = RobotController().navigation
        navigation.start(2.5, 0.5, local_map=local_map, pose=pose)
        navigation._clear_pending_planning()
        navigation.goal = (pose.x_m, pose.y_m)
        navigation.goal_mode = "approaching_safe_stop"
        navigation._goal_access_cell = (1, 0)
        navigation._set_path([(1, 0)])
        navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=1.0,
            max_angular_rps=1.0,
        )
        observed = LocalMapDelta(
            tuple(
                MapCellUpdate(gx, gy, FREE)
                for gx in range(-1, 4)
                for gy in range(-2, 3)
            )
        )

        desired = navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=1.0,
            max_angular_rps=1.0,
            map_delta=observed,
        )

        self.assertEqual(desired, (0.0, 0.0))
        self.assertEqual(navigation.status, "reached")
        self.assertEqual(navigation.goal_mode, "nearby_safe")
        self.assertEqual(navigation.reason, "nearby_safe_stop")

    def test_new_obstacle_exits_final_approach_and_replans(self) -> None:
        local_map = ObservedGrid(
            AnchorSpec("final-approach-anchor", 0.0, 0.0, 0.0),
            resolution_m=1.0,
        )
        pose = PoseEstimate(
            "final-approach-anchor",
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        navigation = RobotController().navigation
        navigation.start(3.5, 0.5, local_map=local_map, pose=pose)
        unsafe_endpoint = 1.5, 0.5
        navigation._clear_pending_planning()
        navigation.goal = unsafe_endpoint
        navigation.goal_mode = "nearby_safe"
        navigation._goal_access_cell = (0, 0)
        navigation._set_path([(0, 0)])
        navigation._final_approach = True

        desired = navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=1.0,
            max_angular_rps=1.0,
            map_delta=LocalMapDelta((MapCellUpdate(1, 0, OCCUPIED),)),
        )

        self.assertFalse(navigation.snapshot()["final_approach"])
        self.assertEqual(desired, (0.0, 0.0))
        self.assertTrue(navigation.snapshot()["planning"])

    def test_unknown_dead_end_turns_around_exits_and_replans(self) -> None:
        walls = {
            *((x, y) for x in range(3, 16) for y in (3, 7)),
            *((15, y) for y in range(4, 7)),
        }
        runtime = make_runtime(
            MapGrid.from_wall_set(22, 12, walls),
            map_resolution_m=0.5,
        )
        start_missions(
            runtime,
            GotoMission("dead-end", "global_map", 18.5, 5.5, 2),
        )

        furthest_x = runtime.vehicle.x
        entry_yaw = runtime.vehicle.yaw
        reversed_heading = False
        exited_after_reversal = False
        for tick in range(1, 500):
            runtime.update(tick / 4, tick / 4)
            furthest_x = max(furthest_x, runtime.vehicle.x)
            yaw_delta = abs(
                math.atan2(
                    math.sin(runtime.vehicle.yaw - entry_yaw),
                    math.cos(runtime.vehicle.yaw - entry_yaw),
                )
            )
            reversed_heading |= yaw_delta > math.radians(150)
            exited_after_reversal |= (
                reversed_heading and runtime.vehicle.x < furthest_x - 0.5
            )
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break

        self.assertFalse(runtime.vehicle.collision)
        self.assertGreater(furthest_x, 2.75)
        self.assertTrue(reversed_heading, runtime.controller.snapshot())
        self.assertTrue(exited_after_reversal, runtime.controller.snapshot())
        self.assertEqual(
            runtime.controller.navigation.status,
            "reached",
            runtime.controller.snapshot(),
        )

    def test_sealed_exit_becomes_blocked_within_bounded_ticks(self) -> None:
        walls = {
            *((x, y) for x in range(3, 10) for y in (3, 9)),
            *((x, y) for x in (3, 9) for y in range(4, 9)),
        }
        runtime = make_runtime(
            MapGrid.from_wall_set(16, 14, walls),
            x_m=6.5,
            y_m=6.5,
            map_resolution_m=0.5,
        )
        start_missions(
            runtime,
            GotoMission("sealed", "global_map", 12.5, 6.5, 2),
        )

        for tick in range(1, 80):
            runtime.update(tick / 4, tick / 4)
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break

        self.assertFalse(runtime.vehicle.collision)
        self.assertLessEqual(tick, 20)
        self.assertEqual(
            runtime.controller.auto_state,
            AutoState.BLOCKED,
            runtime.controller.snapshot(),
        )
        self.assertEqual(runtime.controller.navigation.reason, "no_path")

    def test_multi_obstacle_layout_terminates_without_active_stop_loop(self) -> None:
        walls = {
            (5, 2),
            (5, 4),
            (5, 5),
            (6, 3),
            (6, 4),
            (6, 10),
            (7, 6),
            (7, 10),
            (8, 2),
            (9, 5),
            (9, 8),
            (9, 9),
            (10, 6),
            (11, 6),
            (11, 7),
            (12, 3),
            (12, 5),
            (13, 1),
            (13, 10),
        }
        runtime = make_runtime(
            MapGrid.from_wall_set(20, 12, walls),
            map_resolution_m=0.5,
        )
        start_missions(
            runtime,
            GotoMission("multi-obstacle", "global_map", 14.5, 5.5, 2),
        )

        start = runtime.vehicle.x, runtime.vehicle.y
        max_displacement = 0.0
        for tick in range(1, 500):
            runtime.update(tick / 6, tick / 6)
            max_displacement = max(
                max_displacement,
                math.dist(start, (runtime.vehicle.x, runtime.vehicle.y)),
            )
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break

        self.assertFalse(runtime.vehicle.collision)
        self.assertGreater(max_displacement, 1.0)
        self.assertIsNot(
            runtime.controller.auto_state,
            AutoState.ACTIVE,
            runtime.controller.snapshot(),
        )
        self.assertIn(runtime.controller.navigation.status, {"blocked", "reached"})
        if runtime.controller.navigation.status == "blocked":
            self.assertEqual(runtime.controller.navigation.reason, "no_path")

    def test_fresh_edge_stop_is_absorbed_then_replanned(self) -> None:
        class OneShotEdgeSafety(LocalSafetyRuntime):
            def __init__(self) -> None:
                super().__init__()
                self.fired = False

            def evaluate(
                self,
                vehicle,
                grid,
                desired_linear_mps,
                desired_angular_rps,
                *,
                automatic,
                scan_points=None,
                scan_healthy=True,
            ) -> SafetyDecision:
                if automatic and desired_linear_mps and not self.fired:
                    self.fired = True
                    self.observation = SafetyObservation(
                        edge_clearance_m=0.25,
                        edge_point_vehicle_m=(1.5, 0.0),
                    )
                    self.decision = SafetyDecision(
                        0.0,
                        desired_angular_rps,
                        "stopped",
                        "safety_edge",
                    )
                    return self.decision
                return super().evaluate(
                    vehicle,
                    grid,
                    desired_linear_mps,
                    desired_angular_rps,
                    automatic=automatic,
                    scan_points=scan_points,
                    scan_healthy=scan_healthy,
                )

        runtime = make_runtime(
            MapGrid.from_wall_set(20, 12, set()),
            map_resolution_m=0.5,
        )
        runtime.safety = OneShotEdgeSafety()
        start_missions(
            runtime,
            GotoMission("fresh-edge", "global_map", 8.5, 5.5, 2),
        )

        deferred = False
        for tick in range(1, 300):
            runtime.update(tick / 6, tick / 6)
            deferred |= (
                runtime.controller.auto_state is AutoState.ACTIVE
                and runtime.safety.decision.state == "stopped"
            )
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break

        self.assertTrue(deferred)
        self.assertFalse(runtime.vehicle.collision)
        self.assertEqual(runtime.controller.auto_state, AutoState.IDLE)
        self.assertEqual(runtime.controller.navigation.status, "reached")

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

        event_cursor = runtime.controller.latest_event_seq
        terminal_events = []
        first_replan_x = None
        for tick in range(1, 900):
            runtime.update(tick / 6, tick / 6)
            new_events = runtime.controller.events_after(event_cursor)
            if new_events:
                event_cursor = new_events[-1].event_seq
                terminal_events.extend(new_events)
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
        self.assertIn(
            "reached",
            [event.status for event in terminal_events],
        )

    def test_narrow_unsafe_corridor_replans_through_open_detour(self) -> None:
        walls = {
            (x, y)
            for x in range(6, 11)
            for y in (4, 6)
        }
        runtime = make_runtime(
            MapGrid.from_wall_set(20, 12, walls),
            map_resolution_m=0.5,
        )
        start_missions(
            runtime,
            GotoMission("narrow-detour", "global_map", 14.5, 5.5, 2),
        )

        max_lateral_deviation = 0.0
        for tick in range(1, 1200):
            runtime.update(tick / 6, tick / 6)
            max_lateral_deviation = max(
                max_lateral_deviation,
                abs(runtime.vehicle.y - 5.5),
            )
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break

        self.assertFalse(runtime.vehicle.collision)
        self.assertEqual(
            runtime.controller.navigation.status,
            "reached",
            runtime.controller.snapshot(),
        )
        self.assertGreater(max_lateral_deviation, 1.0)

    def test_obstacle_goal_blocks_when_one_metre_and_clearance_are_incompatible(self) -> None:
        grid = MapGrid.from_wall_set(16, 12, {(8, 5)})
        runtime = make_runtime(grid, map_resolution_m=0.5)
        requested = (8.5, 5.5)
        start_missions(
            runtime,
            GotoMission("blocked-goal", "global_map", *requested, 2),
        )

        # A radius-0.5 vehicle needs sqrt(2) * 0.5 + 0.8 = 1.507 m
        # centre distance around this cell corner, so the 1 m body-distance
        # terminal bound and 0.30 m clearance have no intersection.
        for tick in range(1, 900):
            runtime.update(tick / 6, tick / 6)
            if runtime.controller.auto_state is not AutoState.ACTIVE:
                break
        else:
            self.fail(runtime.controller.snapshot())

        self.assertFalse(runtime.vehicle.collision)
        self.assertEqual(runtime.controller.auto_state, AutoState.BLOCKED)
        self.assertEqual(runtime.controller.navigation.status, "blocked")
        self.assertEqual(runtime.controller.navigation.reason, "no_path")
        self.assertEqual(
            runtime.controller.navigation.detail,
            "nearby_safe_goal_unavailable",
        )
        self.assertEqual(runtime.vehicle.body_velocities(), (0.0, 0.0))

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
            for event in controller.events_after(0)
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
