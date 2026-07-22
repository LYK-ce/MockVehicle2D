"""Safety sensing and pure decision checks."""

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.collision import is_circle_passable, is_swept_circle_passable, raycast
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import (
    HARD_STOP_CLEARANCE_M,
    SLOW_ZONE_CLEARANCE_M,
    SafetyGovernor,
    SafetyObservation,
    nearest_edge_clearance,
    nearest_obstacle_clearance,
)
from mockvehicle2d.scan import LaserPoint, ScanConfig, scan_grid


class MapStateSafetyTest(unittest.TestCase):
    def test_void_is_not_a_wall_but_is_not_passable_or_ground(self) -> None:
        grid = MapGrid(3, 3)
        grid.set_cell(1, 1, 2)

        self.assertFalse(grid.is_wall(1, 1))
        self.assertTrue(grid.is_void(1, 1))
        self.assertFalse(grid.has_ground(1, 1))
        self.assertFalse(grid.is_passable(1, 1))
        self.assertFalse(grid.has_ground(-1, 1))

        for invalid in (-1, 3, True, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                grid.set_cell(0, 0, invalid)

    def test_collision_treats_void_as_non_passable(self) -> None:
        grid = MapGrid(6, 4)
        grid.set_cell(3, 1, 2)

        self.assertFalse(is_circle_passable(grid, 3.5, 1.5, 0.4))
        self.assertFalse(is_swept_circle_passable(grid, 1.5, 1.5, 4.5, 1.5, 0.4))
        self.assertTrue(raycast(grid, 1, 1, 4, 1).hit)
        self.assertTrue(raycast(grid, 1, 2, 8, 2).hit)

    def test_horizontal_tmini_only_returns_walls(self) -> None:
        config = ScanConfig(0.0, 0.0, 1.0, 0.1, 0.05, 8.0)
        grid = MapGrid(8, 3)
        grid.set_cell(3, 1, 2)
        self.assertEqual(scan_grid(grid, 1.5, 1.5, 0.0, config)[0].range, 0.0)

        grid.set_cell(5, 1, 1)
        self.assertAlmostEqual(scan_grid(grid, 1.5, 1.5, 0.0, config)[0].range, 3.5)


class SafetySensingTest(unittest.TestCase):
    def test_obstacle_clearance_selects_travel_sector_and_ignores_no_return(self) -> None:
        points = [
            LaserPoint(0.0, 0.0, 0.0),
            LaserPoint(math.radians(20), 1.4, 1.0),
            LaserPoint(math.pi / 2, 0.5, 1.0),
            LaserPoint(math.pi, 0.9, 1.0),
        ]

        self.assertAlmostEqual(nearest_obstacle_clearance(points, 0.5, 0.4), 1.0)
        self.assertAlmostEqual(nearest_obstacle_clearance(points, -0.5, 0.4), 0.5)
        self.assertIsNone(nearest_obstacle_clearance(points, 0.0, 0.4))

    def test_edge_clearance_detects_void_out_of_bounds_and_vehicle_width(self) -> None:
        grid = MapGrid(8, 6)
        grid.set_cell(5, 3, 2)

        clearance = nearest_edge_clearance(
            grid, 3.5, 2.5, 0.0, 0.5, vehicle_radius=0.6, lookahead_m=3.0
        )
        self.assertIsNotNone(clearance)
        self.assertAlmostEqual(clearance, 1.15, delta=0.05)

        reverse_clearance = nearest_edge_clearance(
            MapGrid(8, 6), 1.5, 2.5, 0.0, -0.5, vehicle_radius=0.5, lookahead_m=2.0
        )
        self.assertIsNotNone(reverse_clearance)
        self.assertAlmostEqual(reverse_clearance, 1.0, delta=0.05)

        self.assertIsNone(
            nearest_edge_clearance(
                MapGrid(8, 6), 3.5, 2.5, 0.0, 0.5, vehicle_radius=0.4, lookahead_m=2.0
            )
        )


class SafetyGovernorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.governor = SafetyGovernor()

    def test_automatic_slow_zone_scales_translation(self) -> None:
        midpoint = (HARD_STOP_CLEARANCE_M + SLOW_ZONE_CLEARANCE_M) / 2
        decision = self.governor.limit(0.5, 0.2, SafetyObservation(obstacle_clearance_m=midpoint), True)

        self.assertEqual(decision.state, "limited")
        self.assertEqual(decision.reason, "safety_obstacle")
        self.assertAlmostEqual(decision.linear_mps, 0.25)
        self.assertEqual(decision.angular_rps, 0.2)

    def test_hard_edge_stops_manual_translation_but_allows_rotation(self) -> None:
        decision = self.governor.limit(
            -0.5,
            -0.3,
            SafetyObservation(edge_clearance_m=HARD_STOP_CLEARANCE_M / 2),
            False,
        )

        self.assertEqual(decision.state, "stopped")
        self.assertEqual(decision.reason, "safety_edge")
        self.assertEqual(decision.linear_mps, 0.0)
        self.assertEqual(decision.angular_rps, -0.3)

    def test_manual_ignores_slow_zone(self) -> None:
        decision = self.governor.limit(
            0.3,
            0.1,
            SafetyObservation(obstacle_clearance_m=SLOW_ZONE_CLEARANCE_M / 2),
            False,
        )
        self.assertEqual((decision.linear_mps, decision.angular_rps), (0.3, 0.1))
        self.assertEqual((decision.state, decision.reason), ("clear", None))

    def test_sensor_fault_stops_translation_and_pure_rotation_remains_available(self) -> None:
        decision = self.governor.limit(0.4, 0.3, SafetyObservation(healthy=False), True)
        self.assertEqual((decision.linear_mps, decision.angular_rps), (0.0, 0.3))
        self.assertEqual((decision.state, decision.reason), ("fault", "safety_sensor_fault"))

        rotation = self.governor.limit(
            0.0,
            -0.4,
            SafetyObservation(obstacle_clearance_m=0.0),
            True,
        )
        self.assertEqual((rotation.linear_mps, rotation.angular_rps), (0.0, -0.4))
        self.assertEqual(rotation.state, "stopped")

        clear = self.governor.limit(0.4, 0.0, SafetyObservation(), True)
        self.assertEqual((clear.state, clear.reason), ("clear", None))


if __name__ == "__main__":
    unittest.main()
