"""Headless fixed-tick episode execution."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    GotoMission,
    ModeAction,
    ModeCommand,
    PatrolMission,
)
from mockvehicle2d.episode import (
    MIN_PROGRESS_TRANSLATION_M,
    _DeterministicPeerStateExchange,
    _update_no_progress,
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
from mockvehicle2d.safety import AUTOMATIC_MINIMUM_CLEARANCE_M


REPO_ROOT = Path(__file__).resolve().parents[1]


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


class TestEpisodeRunner(unittest.TestCase):
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
            )

        self.assertEqual((current, longest), (3, 3))

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
        real_stopped = episode_module._vehicles_stopped

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

        with patch(
            "mockvehicle2d.episode._vehicles_stopped",
            side_effect=record_stopping_state,
        ):
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
                target == (0.0, 0.0) and executed != (0.0, 0.0)
                for state in observed_stopping
                for target, executed in state
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
