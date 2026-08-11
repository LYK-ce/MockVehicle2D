"""Headless fixed-tick episode execution."""

import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pytest

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    CoverageMission,
    GotoMission,
    ModeAction,
    ModeCommand,
    OpMode,
    PatrolMission,
)
from mockvehicle2d.episode import (
    EpisodeResult,
    MIN_PROGRESS_TRANSLATION_M,
    _DeterministicPeerStateExchange,
    _update_no_progress,
    _vehicle_has_unfinished_work,
    _vehicles_stopped,
    run_episode,
)
from mockvehicle2d.fleet import (
    AnchorPose,
    FleetRuntime,
    FleetScenario,
    FleetVehicleSpec,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import P2PSettings
from mockvehicle2d.protocol import parse_command
from mockvehicle2d.safety import AUTOMATIC_MINIMUM_CLEARANCE_M


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUR_VEHICLE_CROSSING_GOALS = (
    ("mock_vehicle_01", (15.0, 10.0)),
    ("mock_vehicle_02", (5.0, 10.0)),
    ("mock_vehicle_03", (10.0, 15.0)),
    ("mock_vehicle_04", (10.0, 5.0)),
)
FOUR_VEHICLE_PATROL_ROUTES = (
    ("mock_vehicle_01", ((12.0, 9.0), (7.0, 9.0))),
    ("mock_vehicle_02", ((8.0, 11.0), (13.0, 11.0))),
    ("mock_vehicle_03", ((9.0, 12.0), (9.0, 7.0))),
    ("mock_vehicle_04", ((11.0, 8.0), (11.0, 13.0))),
)
FOUR_VEHICLE_MERGE_PATROL_ROUTES = (
    ("mock_vehicle_01", ((12.0, 10.0), (7.0, 9.0))),
    ("mock_vehicle_02", ((8.0, 10.0), (13.0, 11.0))),
    ("mock_vehicle_03", ((10.0, 12.0), (9.0, 7.0))),
    ("mock_vehicle_04", ((10.0, 8.0), (11.0, 13.0))),
)
FOUR_VEHICLE_DISJOINT_PATROL_ROUTES = (
    ("mock_vehicle_01", ((6.5, 8.5), (7.0, 9.0))),
    ("mock_vehicle_02", ((13.5, 11.5), (13.0, 11.0))),
    ("mock_vehicle_03", ((8.5, 6.5), (9.0, 7.0))),
    ("mock_vehicle_04", ((11.5, 13.5), (11.0, 13.0))),
)
FOUR_VEHICLE_COVERAGE_STRIPES = (
    ("mock_vehicle_01", (6.0, 8.0, 7.8, 12.0)),
    ("mock_vehicle_03", (8.2, 8.0, 10.0, 12.0)),
    ("mock_vehicle_04", (10.4, 8.0, 12.2, 12.0)),
    ("mock_vehicle_02", (12.6, 8.0, 14.4, 12.0)),
)
FOUR_VEHICLE_COVERAGE_QUADRANTS = (
    ("mock_vehicle_01", (6.0, 6.0, 9.8, 9.8)),
    ("mock_vehicle_02", (10.2, 10.2, 14.0, 14.0)),
    ("mock_vehicle_03", (10.2, 6.0, 14.0, 9.8)),
    ("mock_vehicle_04", (6.0, 10.2, 9.8, 14.0)),
)
FOUR_VEHICLE_CYCLE_SPECS = (
    ("vehicle_1", (8.0, 8.0, 0.0), (12.0, 8.0)),
    ("vehicle_2", (12.0, 8.0, 1.5707963267948966), (12.0, 12.0)),
    ("vehicle_3", (12.0, 12.0, 3.141592653589793), (8.0, 12.0)),
    ("vehicle_4", (8.0, 12.0, -1.5707963267948966), (8.0, 8.0)),
)


def scenario(*, p2p: P2PSettings | None = None) -> FleetScenario:
    return FleetScenario(
        "episode_test",
        (
            FleetVehicleSpec(
                "vehicle_1",
                19090,
                "spawn_1",
                AnchorPose(5.0, 5.0, 0.0),
                None if p2p is None else 20090,
            ),
        ),
        100,
        p2p,
    )


def mission(x_m: float) -> GotoMission:
    return GotoMission("goto-1", "global_map", x_m, 5.0, 2)


def run_four_vehicle_crossing(scenario: FleetScenario) -> EpisodeResult:
    return run_episode(
        scenario,
        {
            vehicle_id: (
                GotoMission(
                    f"goto-{vehicle_id[-2:]}",
                    "global_map",
                    *goal,
                    2,
                ),
            )
            for vehicle_id, goal in FOUR_VEHICLE_CROSSING_GOALS
        },
        max_simulation_s=90.0,
        grid=MapGrid.from_wall_set(20, 20, set()),
    )


def run_four_vehicle_merge_patrol(
    scenario: FleetScenario,
    *,
    cycles: int = 2,
) -> EpisodeResult:
    return run_episode(
        scenario,
        {
            vehicle_id: (
                PatrolMission(
                    f"merge-patrol-{vehicle_id[-2:]}",
                    "global_map",
                    waypoints,
                    cycles,
                    2,
                ),
            )
            for vehicle_id, waypoints in FOUR_VEHICLE_MERGE_PATROL_ROUTES
        },
        max_simulation_s=150.0,
        grid=MapGrid.from_wall_set(20, 20, set()),
        linear_speed=1.0,
    )


def run_four_vehicle_quadrant_coverage(
    scenario: FleetScenario,
    *,
    lane_spacing_m: float = 1.9,
) -> EpisodeResult:
    return run_episode(
        scenario,
        {
            vehicle_id: (
                CoverageMission(
                    f"coverage-quadrant-{vehicle_id[-2:]}",
                    "global_map",
                    *area,
                    lane_spacing_m,
                    2,
                ),
            )
            for vehicle_id, area in FOUR_VEHICLE_COVERAGE_QUADRANTS
        },
        max_simulation_s=160.0,
        grid=MapGrid.from_wall_set(20, 20, set()),
        linear_speed=1.0,
    )


def grouped_coverage_missions(
    vehicle_ids,
    area,
    *,
    lane_spacing_m=2.0,
    coordination_id="fleet-alpha",
):
    min_x_m, min_y_m, max_x_m, max_y_m = area
    missions = {}
    for vehicle_id in vehicle_ids:
        command = parse_command(
            json.dumps(
                {
                    "type": "auto",
                    "seq": 2,
                    "action": "push",
                    "missions": [
                        {
                            "mission_id": f"coverage-{vehicle_id}",
                            "type": "coverage",
                            "frame_id": "global_map",
                            "area": {
                                "min_x_m": min_x_m,
                                "min_y_m": min_y_m,
                                "max_x_m": max_x_m,
                                "max_y_m": max_y_m,
                            },
                            "lane_spacing_m": lane_spacing_m,
                            "coordination_id": coordination_id,
                        }
                    ],
                }
            ),
            linear_limit_mps=1.0,
            angular_limit_rps=math.pi,
            mission_batch_limit=16,
        )
        missions[vehicle_id] = (command.missions[0],)
    return missions


def run_coverage_episode_with_truth(
    scenario,
    missions,
    *,
    max_simulation_s,
    grid,
):
    truth = {vehicle_id: [] for vehicle_id in missions}
    observed_fleet = []
    real_tick = FleetRuntime.tick

    def record_truth(fleet, timestamp):
        result = real_tick(fleet, timestamp)
        observed_fleet[:] = [fleet]
        for vehicle_id, pose in fleet.world.truth_snapshot().items():
            truth[vehicle_id].append(pose[:2])
        return result

    with patch.object(FleetRuntime, "tick", new=record_truth):
        result = run_episode(
            scenario,
            missions,
            max_simulation_s=max_simulation_s,
            grid=grid,
            linear_speed=1.0,
        )
    return result, truth, observed_fleet[0]


class TestEpisodeRunner(unittest.TestCase):
    def test_zero_speed_idle_vacate_session_is_not_episode_stopped(self) -> None:
        fleet = FleetRuntime.create(
            scenario(),
            grid=MapGrid.from_wall_set(20, 20, set()),
        )
        controller = fleet.nodes["vehicle_1"].controller
        controller.mode = OpMode.AUTO
        controller._idle_vacate_origin_pose = (5.0, 5.0, 0.0)

        self.assertEqual(
            fleet.world.vehicle("vehicle_1").target_velocities(),
            (0.0, 0.0),
        )
        self.assertFalse(_vehicles_stopped(fleet, ("vehicle_1",)))

        controller._clear_yield()
        self.assertTrue(_vehicles_stopped(fleet, ("vehicle_1",)))

    def assert_four_vehicle_completed(
        self,
        result: EpisodeResult,
        mission_types: tuple[str, str, str, str],
        *,
        max_no_progress_s: float,
    ) -> None:
        self.assertTrue(result.success, result.as_dict())
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(len(result.vehicles), 4)
        self.assertGreaterEqual(
            result.minimum_inter_vehicle_clearance_m,
            AUTOMATIC_MINIMUM_CLEARANCE_M,
        )
        self.assertTrue(
            all(len(vehicle["missions"]) == 1 for vehicle in result.vehicles)
        )
        self.assertEqual(
            [vehicle["missions"][0]["type"] for vehicle in result.vehicles],
            list(mission_types),
        )
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and vehicle["blocked_reason"] is None
                and vehicle["missions"][0]["status"] == "reached"
                and vehicle["longest_no_progress_duration_s"]
                <= max_no_progress_s
                for vehicle in result.vehicles
            ),
            result.as_dict(),
        )

    def assert_grouped_coverage_completed(
        self,
        outcome,
        routes,
        partition_bounds,
        *,
        axis,
    ) -> None:
        result, truth, fleet = outcome
        self.assertTrue(result.success, result.as_dict())
        self.assertGreaterEqual(
            result.minimum_inter_vehicle_clearance_m,
            AUTOMATIC_MINIMUM_CLEARANCE_M,
        )
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and vehicle["missions"][0]["status"] == "reached"
                and vehicle["final_safety"]["state"] == "clear"
                for vehicle in result.vehicles
            ),
            result.as_dict(),
        )
        self.assertTrue(
            all(
                node.controller.auto_state.value == "idle"
                and not node.controller.is_automatic_motion_active
                for node in fleet.nodes.values()
            )
        )

        endpoint_owners = {}
        for vehicle_id, route in routes.items():
            for endpoint in set(route):
                endpoint_owners.setdefault(endpoint, set()).add(vehicle_id)
        for endpoint, owners in endpoint_owners.items():
            self.assertLessEqual(
                min(
                    math.dist(point, endpoint)
                    for vehicle_id in owners
                    for point in truth[vehicle_id]
                ),
                0.6,
            )

        vehicle_ids = sorted(routes)
        for index, vehicle_id in enumerate(vehicle_ids):
            coordinates = [point[axis] for point in truth[vehicle_id]]
            lower, upper = partition_bounds[vehicle_id]
            if index:
                self.assertGreaterEqual(min(coordinates), lower - 0.75)
            if index + 1 < len(vehicle_ids):
                self.assertLessEqual(max(coordinates), upper + 0.75)

    def test_completion_is_stable_across_realtime_factors(self) -> None:
        results = [
            run_episode(
                scenario(),
                {"vehicle_1": (mission(5.6),)},
                max_simulation_s=10.0,
                grid=MapGrid.from_wall_set(20, 20, set()),
                realtime_factor=factor,
            )
            for factor in (1.0, 5.0)
        ]

        self.assertEqual(results[0].to_json(), results[1].to_json())
        payload = results[0].as_dict()
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["termination_reason"], "completed")
        self.assertIsNone(payload["minimum_inter_vehicle_clearance_m"])
        self.assertGreater(payload["tick_count"], 0)
        self.assertGreater(payload["vehicles"][0]["path_length_m"], 0.0)
        self.assertEqual(payload["vehicles"][0]["missions"][0]["status"], "reached")
        self.assertEqual(
            json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
            results[0].to_json(),
        )

    def test_initial_clearance_and_vehicle_order_are_deterministic(self) -> None:
        specs = (
            FleetVehicleSpec(
                "vehicle_1",
                19090,
                "spawn_1",
                AnchorPose(5.0, 5.0, 3.141592653589793),
            ),
            FleetVehicleSpec(
                "vehicle_2",
                19091,
                "spawn_2",
                AnchorPose(8.0, 5.0, 0.0),
            ),
        )
        missions = {
            "vehicle_1": (mission(4.4),),
            "vehicle_2": (mission(8.6),),
        }
        results = [
            run_episode(
                FleetScenario("clearance_test", order, 100),
                missions,
                max_simulation_s=0.05,
                grid=MapGrid.from_wall_set(20, 20, set()),
            )
            for order in (specs, tuple(reversed(specs)))
        ]

        self.assertEqual(results[0].to_json(), results[1].to_json())
        self.assertEqual(results[0].tick_count, 0)
        self.assertEqual(results[0].minimum_inter_vehicle_clearance_m, 2.0)

    def test_no_progress_resets_on_translation_and_keeps_current_tail(self) -> None:
        current = longest = 0
        for translation_m in (
            0.0,
            0.0,
            MIN_PROGRESS_TRANSLATION_M,
            0.0,
            0.0,
            0.0,
        ):
            current, longest = _update_no_progress(
                current,
                longest,
                translation_m,
                work_active=True,
            )

        self.assertEqual((current, longest), (3, 3))

    def test_no_progress_excludes_terminal_idle_but_tracks_pending_work(self) -> None:
        statuses = {
            ("early", "first"): "reached",
            ("other", "only"): "not_started",
        }
        self.assertFalse(_vehicle_has_unfinished_work(statuses, "early"))
        self.assertTrue(_vehicle_has_unfinished_work(statuses, "other"))
        self.assertEqual(
            _update_no_progress(4, 4, 0.0, work_active=False),
            (0, 4),
        )

        statuses[("early", "second")] = "not_started"
        self.assertTrue(_vehicle_has_unfinished_work(statuses, "early"))
        self.assertEqual(
            _update_no_progress(4, 4, 0.0, work_active=True),
            (5, 5),
        )

    def test_two_vehicle_crossing_example_reports_interaction_metrics(self) -> None:
        from mockvehicle2d import episode as episode_module

        crossing = FleetScenario.load(
            REPO_ROOT / "examples" / "two_vehicle_crossing_episode.json"
        )
        missions = {
            "mock_vehicle_01": (
                GotoMission("goto-1", "global_map", 11.0, 11.0, 2),
            ),
            "mock_vehicle_02": (
                GotoMission("goto-2", "global_map", 9.0, 11.0, 2),
            ),
        }
        observed_stopping = []
        observed_control_stops = []
        real_stopped = episode_module._vehicles_stopped
        real_tick = FleetRuntime.tick

        def record_stopping_state(fleet, vehicle_ids):
            observed_stopping.append(
                tuple(
                    (
                        fleet.world.vehicle(vehicle_id).target_velocities(),
                        fleet.world.vehicle(vehicle_id).body_velocities(),
                    )
                    for vehicle_id in vehicle_ids
                )
            )
            return real_stopped(fleet, vehicle_ids)

        def record_control_stops(fleet, now):
            result = real_tick(fleet, now)
            observed_control_stops.append(fleet.control_stop_transitions)
            return result

        with patch(
            "mockvehicle2d.episode._vehicles_stopped",
            side_effect=record_stopping_state,
        ), patch.object(FleetRuntime, "tick", new=record_control_stops):
            results = [
                run_episode(
                    crossing,
                    missions,
                    max_simulation_s=30.0,
                    grid=MapGrid.from_wall_set(24, 24, set()),
                    realtime_factor=factor,
                )
                for factor in (1.0, 5.0)
            ]

        self.assertEqual(results[0].to_json(), results[1].to_json())
        result = results[0]
        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "completed")
        clearance = result.minimum_inter_vehicle_clearance_m
        self.assertIsNotNone(clearance)
        assert clearance is not None
        self.assertGreaterEqual(clearance, AUTOMATIC_MINIMUM_CLEARANCE_M)
        self.assertEqual(len(result.vehicles), 2)
        self.assertEqual(
            [vehicle["blocked_reason"] for vehicle in result.vehicles],
            [None, None],
        )
        self.assertEqual(
            [vehicle["missions"][0]["status"] for vehicle in result.vehicles],
            ["reached", "reached"],
        )
        self.assertTrue(
            any(
                transition
                for transition in observed_control_stops
            )
        )
        self.assertTrue(
            all(
                target == executed == (0.0, 0.0)
                for target, executed in observed_stopping[-1]
            )
        )
        for vehicle in result.vehicles:
            self.assertGreater(vehicle["longest_no_progress_duration_s"], 0.0)
            self.assertLessEqual(
                vehicle["longest_no_progress_duration_s"],
                result.simulation_duration_s,
            )

    def test_four_vehicle_crossing_is_deterministic_and_order_independent(self) -> None:
        crossing = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_crossing_episode.json"
        )
        reordered = FleetScenario(
            crossing.scenario_id,
            tuple(reversed(crossing.vehicles)),
            crossing.tick_ms,
        )
        results = (
            run_four_vehicle_crossing(crossing),
            run_four_vehicle_crossing(crossing),
            run_four_vehicle_crossing(reordered),
        )

        self.assertEqual(len({result.to_json() for result in results}), 1)
        for result in results:
            self.assertTrue(result.success, result.as_dict())
            self.assertEqual(result.termination_reason, "completed")
            self.assertGreaterEqual(
                result.minimum_inter_vehicle_clearance_m,
                AUTOMATIC_MINIMUM_CLEARANCE_M,
            )
            self.assertEqual(len(result.vehicles), 4)
            self.assertTrue(
                all(
                    not vehicle["collision_occurred"]
                    and not vehicle["blocked"]
                    and vehicle["blocked_reason"] is None
                    and vehicle["missions"][0]["status"] == "reached"
                    for vehicle in result.vehicles
                )
            )

    def test_four_vehicle_crossing_is_stable_across_tick_sizes(self) -> None:
        crossing = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_crossing_episode.json"
        )

        for tick_ms in (50, 250):
            with self.subTest(tick_ms=tick_ms):
                result = run_four_vehicle_crossing(
                    FleetScenario(
                        crossing.scenario_id,
                        crossing.vehicles,
                        tick_ms,
                    )
                )
                self.assertTrue(result.success, result.as_dict())
                self.assertEqual(result.termination_reason, "completed")
                self.assertGreaterEqual(
                    result.minimum_inter_vehicle_clearance_m,
                    AUTOMATIC_MINIMUM_CLEARANCE_M,
                )
                self.assertTrue(
                    all(
                        not vehicle["collision_occurred"]
                        and not vehicle["blocked"]
                        and vehicle["missions"][0]["status"] == "reached"
                        for vehicle in result.vehicles
                    )
                )

    def test_disjoint_goto_is_unchanged_for_one_two_and_four_vehicles(self) -> None:
        specs = tuple(
            FleetVehicleSpec(
                f"vehicle_{index}",
                19089 + index,
                f"spawn_{index}",
                AnchorPose(4.0, 4.0 * index, 0.0),
            )
            for index in range(1, 5)
        )
        missions = {
            f"vehicle_{index}": (
                GotoMission(
                    f"disjoint-goto-{index}",
                    "global_map",
                    7.0,
                    4.0 * index,
                    2,
                ),
            )
            for index in range(1, 5)
        }
        results = tuple(
            run_episode(
                FleetScenario("disjoint_goto", specs[:vehicle_count], 100),
                {
                    vehicle_id: vehicle_missions
                    for vehicle_id, vehicle_missions in missions.items()
                    if int(vehicle_id[-1]) <= vehicle_count
                },
                max_simulation_s=20.0,
                grid=MapGrid.from_wall_set(24, 24, set()),
                linear_speed=1.0,
            )
            for vehicle_count in (1, 2, 4)
        )

        baseline = results[0].vehicles[0]
        self.assertEqual(
            [result.tick_count for result in results],
            [results[0].tick_count] * len(results),
        )
        for result in results:
            self.assertTrue(result.success, result.as_dict())
            self.assertEqual(result.termination_reason, "completed")
            if len(result.vehicles) > 1:
                self.assertGreaterEqual(
                    result.minimum_inter_vehicle_clearance_m,
                    AUTOMATIC_MINIMUM_CLEARANCE_M,
                )
            for vehicle in result.vehicles:
                self.assertEqual(
                    (
                        vehicle["path_length_m"],
                        vehicle["longest_no_progress_duration_s"],
                    ),
                    (
                        baseline["path_length_m"],
                        baseline["longest_no_progress_duration_s"],
                    ),
                )
                self.assertFalse(vehicle["collision_occurred"])
                self.assertFalse(vehicle["blocked"])
                self.assertEqual(vehicle["missions"][0]["status"], "reached")

    def test_four_vehicle_cycle_keeps_each_goto_on_its_commanded_vehicle(
        self,
    ) -> None:
        specs = tuple(
            FleetVehicleSpec(
                vehicle_id,
                19090 + index,
                f"spawn_{index + 1}",
                AnchorPose(*start),
            )
            for index, (vehicle_id, start, _) in enumerate(
                FOUR_VEHICLE_CYCLE_SPECS
            )
        )
        missions = {
            vehicle_id: (
                GotoMission(
                    f"cycle-goto-{vehicle_id[-1]}",
                    "global_map",
                    *goal,
                    2,
                ),
            )
            for vehicle_id, _, goal in FOUR_VEHICLE_CYCLE_SPECS
        }
        results = tuple(
            run_episode(
                FleetScenario("cycle_goto", ordered_specs, 100),
                missions,
                max_simulation_s=30.0,
                grid=MapGrid.from_wall_set(20, 20, set()),
                linear_speed=1.0,
            )
            for ordered_specs in (specs, tuple(reversed(specs)))
        )

        self.assertEqual(results[0].to_json(), results[1].to_json())
        result = results[0]
        self.assert_four_vehicle_completed(
            result,
            ("goto",) * 4,
            max_no_progress_s=5.0,
        )
        self.assertEqual(
            [vehicle["missions"][0]["mission_id"] for vehicle in result.vehicles],
            [f"cycle-goto-{index}" for index in range(1, 5)],
        )

    def test_four_vehicle_adjacent_goto_endpoints_finish_without_reassignment(
        self,
    ) -> None:
        goals = (
            (12.0, 8.05),
            (12.0, 9.4),
            (12.0, 10.75),
            (12.0, 12.1),
        )
        specs = tuple(
            FleetVehicleSpec(
                f"vehicle_{index}",
                19089 + index,
                f"spawn_{index}",
                AnchorPose(4.0, 5.0 + 2.5 * index, 0.0),
            )
            for index in range(1, 5)
        )
        result = run_episode(
            FleetScenario("adjacent_goto", specs, 100),
            {
                f"vehicle_{index}": (
                    GotoMission(
                        f"adjacent-goto-{index}",
                        "global_map",
                        *goals[index - 1],
                        2,
                    ),
                )
                for index in range(1, 5)
            },
            max_simulation_s=40.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assert_four_vehicle_completed(
            result,
            ("goto",) * 4,
            max_no_progress_s=10.0,
        )
        self.assertEqual(
            [vehicle["missions"][0]["mission_id"] for vehicle in result.vehicles],
            [f"adjacent-goto-{index}" for index in range(1, 5)],
        )

    def test_parked_goal_vehicle_remains_dynamic_while_peer_routes_around_it(
        self,
    ) -> None:
        parked = FleetScenario(
            "parked_goal",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(10.0, 10.0, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(5.0, 10.0, 0.0),
                ),
            ),
            100,
        )
        result = run_episode(
            parked,
            {
                "vehicle_1": (
                    GotoMission("parked-goto", "global_map", 10.0, 10.0, 2),
                ),
                "vehicle_2": (
                    GotoMission("passing-goto", "global_map", 15.0, 10.0, 2),
                ),
            },
            max_simulation_s=40.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assertTrue(result.success, result.as_dict())
        self.assertEqual(result.termination_reason, "completed")
        self.assertGreaterEqual(
            result.minimum_inter_vehicle_clearance_m,
            AUTOMATIC_MINIMUM_CLEARANCE_M,
        )
        self.assertEqual(result.vehicles[0]["path_length_m"], 0.0)
        self.assertGreater(result.vehicles[1]["path_length_m"], 10.0)
        self.assertEqual(
            [vehicle["missions"][0]["status"] for vehicle in result.vehicles],
            ["reached", "reached"],
        )

    def test_four_vehicle_patrol_repeatedly_clears_shared_crossing(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_episode(
            matrix,
            {
                vehicle_id: (
                    PatrolMission(
                        f"patrol-{vehicle_id[-2:]}",
                        "global_map",
                        waypoints,
                        2,
                        2,
                    ),
                )
                for vehicle_id, waypoints in FOUR_VEHICLE_PATROL_ROUTES
            },
            max_simulation_s=120.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assert_four_vehicle_completed(
            result,
            ("patrol",) * 4,
            max_no_progress_s=30.0,
        )

    def test_four_vehicle_patrol_clears_opposing_merge_routes(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_four_vehicle_merge_patrol(matrix)

        self.assert_four_vehicle_completed(
            result,
            ("patrol",) * 4,
            max_no_progress_s=45.0,
        )

    @pytest.mark.extended
    def test_four_vehicle_merge_patrol_extended_matrix(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        reordered = FleetScenario(
            matrix.scenario_id,
            tuple(reversed(matrix.vehicles)),
            matrix.tick_ms,
        )
        deterministic = tuple(
            run_four_vehicle_merge_patrol(scenario, cycles=1)
            for scenario in (matrix, matrix, reordered)
        )
        tick_results = tuple(
            run_four_vehicle_merge_patrol(
                FleetScenario(matrix.scenario_id, matrix.vehicles, tick_ms),
                cycles=1,
            )
            for tick_ms in (50, 250)
        )

        self.assertEqual(len({result.to_json() for result in deterministic}), 1)
        for result in (*deterministic, *tick_results):
            self.assert_four_vehicle_completed(
                result,
                ("patrol",) * 4,
                max_no_progress_s=45.0,
            )

    def test_four_vehicle_disjoint_patrol_has_no_false_coordination_failure(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_episode(
            matrix,
            {
                vehicle_id: (
                    PatrolMission(
                        f"disjoint-patrol-{vehicle_id[-2:]}",
                        "global_map",
                        waypoints,
                        2,
                        2,
                    ),
                )
                for vehicle_id, waypoints in FOUR_VEHICLE_DISJOINT_PATROL_ROUTES
            },
            max_simulation_s=30.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assert_four_vehicle_completed(
            result,
            ("patrol",) * 4,
            max_no_progress_s=10.0,
        )

    def test_four_vehicle_coverage_completes_adjacent_static_stripes(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_episode(
            matrix,
            {
                vehicle_id: (
                    CoverageMission(
                        f"coverage-stripe-{vehicle_id[-2:]}",
                        "global_map",
                        *area,
                        0.9,
                        2,
                    ),
                )
                for vehicle_id, area in FOUR_VEHICLE_COVERAGE_STRIPES
            },
            max_simulation_s=150.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assert_four_vehicle_completed(
            result,
            ("coverage",) * 4,
            max_no_progress_s=40.0,
        )

    def test_four_vehicle_coverage_clears_shared_quadrant_ingress(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_four_vehicle_quadrant_coverage(matrix)

        self.assert_four_vehicle_completed(
            result,
            ("coverage",) * 4,
            max_no_progress_s=45.0,
        )

    def test_two_vehicles_partition_one_coordinated_coverage_area(self) -> None:
        specs = (
            FleetVehicleSpec(
                "vehicle_a",
                19090,
                "spawn_a",
                AnchorPose(3.0, 3.0, 0.0),
            ),
            FleetVehicleSpec(
                "vehicle_b",
                19091,
                "spawn_b",
                AnchorPose(13.0, 3.0, math.pi),
            ),
        )
        area = 4.0, 3.0, 12.0, 7.0
        missions = grouped_coverage_missions(
            ("vehicle_a", "vehicle_b"),
            area,
        )
        expected_routes = {
            "vehicle_a": (
                (4.0, 3.0),
                (8.0, 3.0),
                (8.0, 5.0),
                (4.0, 5.0),
                (4.0, 7.0),
                (8.0, 7.0),
            ),
            "vehicle_b": (
                (8.0, 3.0),
                (12.0, 3.0),
                (12.0, 5.0),
                (8.0, 5.0),
                (8.0, 7.0),
                (12.0, 7.0),
            ),
        }
        for vehicle_id, (mission,) in missions.items():
            self.assertEqual(
                mission.effective_subgoals(
                    vehicle_id,
                    tuple(sorted(set(missions) - {vehicle_id})),
                ),
                expected_routes[vehicle_id],
            )
        runs = tuple(
            run_coverage_episode_with_truth(
                FleetScenario("grouped_coverage", vehicles, 100),
                missions,
                max_simulation_s=120.0,
                grid=MapGrid.from_wall_set(20, 12, set()),
            )
            for vehicles in (specs, tuple(reversed(specs)))
        )

        self.assertEqual(runs[0][0].to_json(), runs[1][0].to_json())
        partition_bounds = {
            "vehicle_a": (4.0, 8.0),
            "vehicle_b": (8.0, 12.0),
        }
        for outcome in runs:
            self.assert_grouped_coverage_completed(
                outcome,
                expected_routes,
                partition_bounds,
                axis=0,
            )

    @pytest.mark.extended
    def test_four_vehicles_partition_one_coordinated_coverage_area(self) -> None:
        vehicle_ids = ("vehicle_a", "vehicle_b", "vehicle_c", "vehicle_d")
        specs = tuple(
            FleetVehicleSpec(
                vehicle_id,
                19100 + index,
                f"spawn_{vehicle_id[-1]}",
                AnchorPose(2.0 + 4.0 * index, 3.0, math.pi / 2),
            )
            for index, vehicle_id in enumerate(vehicle_ids)
        )
        area = 2.0, 4.0, 18.0, 8.0
        missions = grouped_coverage_missions(vehicle_ids, area)
        expected_routes = {}
        partition_bounds = {}
        for index, vehicle_id in enumerate(vehicle_ids):
            min_x_m = 2.0 + 4.0 * index
            max_x_m = min_x_m + 4.0
            partition_bounds[vehicle_id] = min_x_m, max_x_m
            expected_routes[vehicle_id] = (
                (min_x_m, 4.0),
                (max_x_m, 4.0),
                (max_x_m, 6.0),
                (min_x_m, 6.0),
                (min_x_m, 8.0),
                (max_x_m, 8.0),
            )
            self.assertEqual(
                missions[vehicle_id][0].effective_subgoals(
                    vehicle_id,
                    tuple(sorted(set(vehicle_ids) - {vehicle_id})),
                ),
                expected_routes[vehicle_id],
            )
        ordered_bounds = [partition_bounds[vehicle_id] for vehicle_id in vehicle_ids]
        self.assertEqual(ordered_bounds[0][0], area[0])
        self.assertEqual(ordered_bounds[-1][1], area[2])
        self.assertTrue(
            all(
                first[1] == second[0]
                for first, second in zip(ordered_bounds, ordered_bounds[1:])
            )
        )

        canonical = run_coverage_episode_with_truth(
            FleetScenario("grouped_coverage_four", specs, 100),
            missions,
            max_simulation_s=120.0,
            grid=MapGrid.from_wall_set(22, 12, set()),
        )
        reversed_declaration = run_coverage_episode_with_truth(
            FleetScenario(
                "grouped_coverage_four",
                tuple(reversed(specs)),
                100,
            ),
            missions,
            max_simulation_s=120.0,
            grid=MapGrid.from_wall_set(22, 12, set()),
        )

        self.assertEqual(canonical[0].to_json(), reversed_declaration[0].to_json())
        for outcome in (canonical, reversed_declaration):
            self.assert_grouped_coverage_completed(
                outcome,
                expected_routes,
                partition_bounds,
                axis=0,
            )

    def test_single_vehicle_legacy_coverage_keeps_the_full_area(self) -> None:
        command = parse_command(
            json.dumps(
                {
                    "type": "auto",
                    "seq": 2,
                    "action": "push",
                    "missions": [
                        {
                            "mission_id": "legacy-coverage",
                            "type": "coverage",
                            "frame_id": "global_map",
                            "area": {
                                "min_x_m": 5.0,
                                "min_y_m": 5.0,
                                "max_x_m": 7.0,
                                "max_y_m": 6.0,
                            },
                            "lane_spacing_m": 1.0,
                        }
                    ],
                }
            ),
            linear_limit_mps=1.0,
            angular_limit_rps=math.pi,
            mission_batch_limit=16,
        )
        coverage = command.missions[0]

        self.assertNotIn("coordination_id", coverage.as_dict())
        self.assertEqual(
            coverage.effective_subgoals("vehicle_1", ()),
            (
                (5.0, 5.0),
                (7.0, 5.0),
                (7.0, 6.0),
                (5.0, 6.0),
            ),
        )

        result = run_episode(
            scenario(),
            {"vehicle_1": (coverage,)},
            max_simulation_s=30.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assertTrue(result.success, result.as_dict())
        self.assertFalse(result.vehicles[0]["collision_occurred"])
        self.assertFalse(result.vehicles[0]["blocked"])
        self.assertEqual(result.vehicles[0]["missions"][0]["status"], "reached")
        self.assertEqual(result.vehicles[0]["final_safety"]["state"], "clear")

    @pytest.mark.extended
    def test_four_vehicle_quadrant_coverage_extended_matrix(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        reordered = FleetScenario(
            matrix.scenario_id,
            tuple(reversed(matrix.vehicles)),
            matrix.tick_ms,
        )
        deterministic = tuple(
            run_four_vehicle_quadrant_coverage(
                scenario,
                lane_spacing_m=3.8,
            )
            for scenario in (matrix, matrix, reordered)
        )
        tick_results = tuple(
            run_four_vehicle_quadrant_coverage(
                FleetScenario(matrix.scenario_id, matrix.vehicles, tick_ms),
                lane_spacing_m=3.8,
            )
            for tick_ms in (50, 250)
        )

        self.assertEqual(len({result.to_json() for result in deterministic}), 1)
        for result in (*deterministic, *tick_results):
            self.assert_four_vehicle_completed(
                result,
                ("coverage",) * 4,
                max_no_progress_s=45.0,
            )

    def test_four_vehicle_mixed_missions_share_coordination(self) -> None:
        matrix = FleetScenario.load(
            REPO_ROOT / "tests" / "fixtures" / "four_vehicle_mission_matrix.json"
        )
        result = run_episode(
            matrix,
            {
                "mock_vehicle_01": (
                    GotoMission("mixed-goto-01", "global_map", 14.5, 9.0, 2),
                ),
                "mock_vehicle_02": (
                    PatrolMission(
                        "mixed-patrol-02",
                        "global_map",
                        ((6.0, 11.0), (13.0, 11.0)),
                        1,
                        2,
                    ),
                ),
                "mock_vehicle_03": (
                    CoverageMission(
                        "mixed-coverage-03",
                        "global_map",
                        8.5,
                        7.5,
                        11.5,
                        9.5,
                        1.0,
                        2,
                    ),
                ),
                "mock_vehicle_04": (
                    CoverageMission(
                        "mixed-coverage-04",
                        "global_map",
                        8.5,
                        10.5,
                        11.5,
                        12.5,
                        1.0,
                        2,
                    ),
                ),
            },
            max_simulation_s=120.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
            linear_speed=1.0,
        )

        self.assert_four_vehicle_completed(
            result,
            ("goto", "patrol", "coverage", "coverage"),
            # This metric includes the Goto vehicle's idle tail while the longer
            # Coverage missions finish; shared-crossing wait is bounded separately.
            max_no_progress_s=45.0,
        )

    def test_completed_episode_drains_residual_motion_before_returning(self) -> None:
        from mockvehicle2d import episode as episode_module

        observed = []
        real_stopped = episode_module._vehicles_stopped

        def record_stopping_state(fleet, vehicle_ids):
            observed.append(
                tuple(
                    (
                        fleet.world.vehicle(vehicle_id).target_velocities(),
                        fleet.world.vehicle(vehicle_id).body_velocities(),
                    )
                    for vehicle_id in vehicle_ids
                )
            )
            return real_stopped(fleet, vehicle_ids)

        with patch(
            "mockvehicle2d.episode._vehicles_stopped",
            side_effect=record_stopping_state,
        ):
            result = run_episode(
                scenario(),
                {"vehicle_1": (mission(5.6),)},
                max_simulation_s=10.0,
                grid=MapGrid.from_wall_set(20, 20, set()),
            )

        self.assertTrue(result.success)
        self.assertTrue(
            any(target == (0.0, 0.0) and executed != (0.0, 0.0)
                for state in observed for target, executed in state)
        )
        self.assertTrue(
            all(target == executed == (0.0, 0.0)
                for target, executed in observed[-1])
        )

    def test_crossing_coordination_is_stable_across_tick_sizes(self) -> None:
        crossing = FleetScenario.load(
            REPO_ROOT / "examples" / "two_vehicle_crossing_episode.json"
        )
        missions = {
            "mock_vehicle_01": (
                GotoMission("goto-1", "global_map", 11.0, 11.0, 2),
            ),
            "mock_vehicle_02": (
                GotoMission("goto-2", "global_map", 9.0, 11.0, 2),
            ),
        }

        for tick_ms in (50, 250):
            with self.subTest(tick_ms=tick_ms):
                result = run_episode(
                    FleetScenario(
                        f"crossing_{tick_ms}",
                        crossing.vehicles,
                        tick_ms,
                    ),
                    missions,
                    max_simulation_s=30.0,
                    grid=MapGrid.from_wall_set(24, 24, set()),
                )
                self.assertTrue(result.success, result.as_dict())
                self.assertFalse(
                    any(vehicle["collision_occurred"] for vehicle in result.vehicles)
                )
                self.assertGreaterEqual(
                    result.minimum_inter_vehicle_clearance_m,
                    AUTOMATIC_MINIMUM_CLEARANCE_M,
                )

    def test_coordination_is_shared_by_patrol_missions(self) -> None:
        crossing = FleetScenario.load(
            REPO_ROOT / "examples" / "two_vehicle_crossing_episode.json"
        )
        result = run_episode(
            crossing,
            {
                "mock_vehicle_01": (
                    GotoMission("goto-1", "global_map", 11.0, 11.0, 2),
                ),
                "mock_vehicle_02": (
                    PatrolMission(
                        "patrol-2",
                        "global_map",
                        ((9.0, 11.0), (9.0, 10.5)),
                        1,
                        2,
                    ),
                ),
            },
            max_simulation_s=40.0,
            grid=MapGrid.from_wall_set(24, 24, set()),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.vehicles[1]["missions"][0]["type"], "patrol")
        self.assertEqual(result.vehicles[1]["missions"][0]["status"], "reached")
        self.assertGreaterEqual(
            result.minimum_inter_vehicle_clearance_m,
            AUTOMATIC_MINIMUM_CLEARANCE_M,
        )

    def test_peer_state_expiry_keeps_an_active_yielder_stopped(self) -> None:
        crossing = FleetScenario.load(
            REPO_ROOT / "examples" / "two_vehicle_crossing_episode.json"
        )
        fleet = FleetRuntime.create(
            crossing,
            grid=MapGrid.from_wall_set(24, 24, set()),
            in_process_peer_states=True,
        )
        exchange = _DeterministicPeerStateExchange(fleet)
        for index, (vehicle_id, goal) in enumerate(
            (
                ("mock_vehicle_01", (11.0, 11.0)),
                ("mock_vehicle_02", (9.0, 11.0)),
            ),
            1,
        ):
            self.assertTrue(
                fleet.handle_command(
                    vehicle_id,
                    ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
                ).accepted
            )
            self.assertTrue(
                fleet.handle_command(
                    vehicle_id,
                    AutoCommand(
                        2,
                        AutoAction.PUSH,
                        (GotoMission(f"goto-{index}", "global_map", *goal, 2),),
                    ),
                ).accepted
            )

        for _ in range(80):
            fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))
            exchange.advance()
            if fleet.nodes["mock_vehicle_02"].controller.is_yielding:
                break
        else:
            self.fail("lower-priority vehicle never yielded")

        for _ in range(6):
            fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))

        node = fleet.nodes["mock_vehicle_02"]
        self.assertEqual(node.map_sync.peer_vehicle_states(), ())
        self.assertTrue(node.controller.is_yielding)
        self.assertEqual(node.controller.auto_state.value, "active")
        self.assertEqual(
            fleet.world.vehicle("mock_vehicle_02").target_velocities(),
            (0.0, 0.0),
        )
        self.assertTrue(
            fleet.handle_command(
                "mock_vehicle_02",
                ModeCommand(3, ModeAction.SWITCH_TO_MANUAL),
            ).accepted
        )
        self.assertFalse(node.controller.is_yielding)

    def test_timeout_uses_simulation_time(self) -> None:
        result = run_episode(
            scenario(),
            {"vehicle_1": (mission(15.0),)},
            max_simulation_s=0.25,
            grid=MapGrid.from_wall_set(20, 20, set()),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "timeout")
        self.assertEqual(result.tick_count, 2)
        self.assertEqual(result.simulation_duration_s, 0.2)

    def test_rejects_empty_work_and_nondeterministic_p2p(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one mission"):
            run_episode(scenario(), {}, max_simulation_s=1.0)
        settings = P2PSettings(Path("sidecar"), Path("runtime"))
        with self.assertRaisesRegex(ValueError, "deterministic communication"):
            run_episode(
                scenario(p2p=settings),
                {"vehicle_1": (mission(5.6),)},
                max_simulation_s=1.0,
            )

    def test_cli_builds_deterministic_goto_and_prints_json(self) -> None:
        from mockvehicle2d.cli import main as cli

        result = unittest.mock.Mock()
        result.to_json.return_value = '{"success":true}'
        arguments = [
            "mockvehicle2d",
            "episode",
            "--scenario",
            str(REPO_ROOT / "examples" / "single_vehicle_episode.json"),
            "--max-simulation-s",
            "10",
            "--goto",
            "mock_vehicle_01,11,10",
            "--linear-acceleration-mps2",
            "2",
            "--linear-deceleration-mps2",
            "3",
            "--angular-acceleration-rps2",
            "4",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch("mockvehicle2d.episode.run_episode", return_value=result) as run,
            patch("builtins.print") as output,
        ):
            cli.main()

        submitted = run.call_args.args[1]["mock_vehicle_01"][0]
        self.assertEqual(submitted.mission_id, "episode-goto-0001")
        self.assertEqual((submitted.x_m, submitted.y_m), (11.0, 10.0))
        self.assertEqual(run.call_args.kwargs["linear_acceleration_mps2"], 2.0)
        self.assertEqual(run.call_args.kwargs["linear_deceleration_mps2"], 3.0)
        self.assertEqual(run.call_args.kwargs["angular_acceleration_rps2"], 4.0)
        output.assert_called_once_with('{"success":true}')


if __name__ == "__main__":
    unittest.main()
