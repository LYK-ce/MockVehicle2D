from pathlib import Path
import unittest

from mockvehicle2d.fleet import FleetRuntime, FleetScenario


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTwoVehicleScenario(unittest.TestCase):
    def test_example_has_safe_unique_endpoints_and_runtime(self) -> None:
        scenario = FleetScenario.load(REPO_ROOT / "examples" / "two_vehicle_scenario.json")

        self.assertEqual(scenario.scenario_id, "two_vehicle_exploration")
        self.assertEqual(
            [vehicle.operator_port for vehicle in scenario.vehicles], [19090, 19091]
        )
        self.assertEqual(
            [vehicle.p2p_port for vehicle in scenario.vehicles], [20090, 20091]
        )
        self.assertIsNotNone(scenario.p2p)
        assert scenario.p2p is not None
        self.assertEqual(str(scenario.p2p.runtime_dir), ".runtime/two_vehicle_exploration")
        self.assertEqual(
            set(FleetRuntime.create(scenario).nodes),
            {"mock_vehicle_01", "mock_vehicle_02"},
        )


if __name__ == "__main__":
    unittest.main()
