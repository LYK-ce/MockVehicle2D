"""Headless fixed-tick episode execution."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from mockvehicle2d.controller import GotoMission
from mockvehicle2d.episode import run_episode
from mockvehicle2d.fleet import AnchorPose, FleetScenario, FleetVehicleSpec
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import P2PSettings


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
        self.assertTrue(payload["success"])
        self.assertEqual(payload["termination_reason"], "completed")
        self.assertGreater(payload["tick_count"], 0)
        self.assertGreater(payload["vehicles"][0]["path_length_m"], 0.0)
        self.assertEqual(payload["vehicles"][0]["missions"][0]["status"], "reached")
        self.assertEqual(
            json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
            results[0].to_json(),
        )

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
        output.assert_called_once_with('{"success":true}')


if __name__ == "__main__":
    unittest.main()
