"""Focused standard-library checks for deterministic YDLidar-style grid scans."""

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.scan import ScanConfig, scan_grid


class GridScanTest(unittest.TestCase):
    def test_first_wall_is_a_return_at_its_boundary(self) -> None:
        grid = MapGrid.from_wall_set(8, 4, {(4, 1)})
        point = scan_grid(grid, 1.5, 1.5, 0.0, ScanConfig(0.0, 0.0, 1.0, 0.1, 0.05, 8.0))[0]
        self.assertIsNotNone(point.range)
        self.assertAlmostEqual(point.range, 2.5)
        self.assertEqual(point.intensity, 1.0)

    def test_axis_aligned_boundary_ray_hits_forward_wall(self) -> None:
        grid = MapGrid.from_wall_set(16, 16, {(11, 10)})
        point = scan_grid(grid, 10.0, 10.0, 0.0, ScanConfig(0.0, 0.0, 1.0, 0.1, 0.05, 12.0))[0]
        self.assertIsNotNone(point.range)
        self.assertTrue(math.isfinite(point.range))
        self.assertAlmostEqual(point.range, 1.0)

    def test_no_return_is_not_a_max_range_obstacle(self) -> None:
        grid = MapGrid.from_wall_set(5, 5, set())
        point = scan_grid(grid, 2.5, 2.5, 0.0, ScanConfig(0.0, 0.0, 1.0, 0.1, 0.05, 8.0))[0]
        self.assertIsNone(point.range)
        self.assertEqual(point.intensity, 0.0)

    def test_sweep_and_yaw_rotate_local_rays(self) -> None:
        config = ScanConfig(0.0, math.pi / 2, math.pi / 2, 0.2, 0.05, 8.0)
        grid = MapGrid.from_wall_set(8, 8, {(4, 1), (1, 4)})
        points = scan_grid(grid, 1.5, 1.5, 0.0, config)
        self.assertEqual([round(point.angle, 6) for point in points], [0.0, round(math.pi / 2, 6)])
        self.assertTrue(all(point.range is not None and abs(point.range - 2.5) < 1e-9 for point in points))
        yaw_point = scan_grid(grid, 1.5, 1.5, math.pi / 2, ScanConfig(0.0, 0.0, 1.0, 0.1, 0.05, 8.0))[0]
        self.assertAlmostEqual(yaw_point.range, 2.5)

    def test_max_range_and_metadata_are_explicit(self) -> None:
        grid = MapGrid.from_wall_set(8, 4, {(4, 1)})
        config = ScanConfig(0.0, math.pi / 2, math.pi / 2, 0.2, 0.05, 2.0)
        point = scan_grid(grid, 1.5, 1.5, 0.0, config)[0]
        self.assertIsNone(point.range)
        self.assertEqual(config.as_dict()["point_count"], 2)
        self.assertEqual(config.as_dict()["no_return"], {"range": None, "intensity": 0.0})

    def test_metadata_max_angle_matches_last_sample(self) -> None:
        config = ScanConfig(0.0, 1.0, 0.6, 0.1, 0.05, 8.0)
        metadata = config.as_dict()
        points = scan_grid(MapGrid.from_wall_set(2, 2, set()), 0.5, 0.5, 0.0, config)
        self.assertEqual(metadata["point_count"], 2)
        self.assertAlmostEqual(metadata["max_angle"], 0.6)
        self.assertAlmostEqual(metadata["max_angle"], points[-1].angle)
        default_metadata = ScanConfig().as_dict()
        self.assertEqual(default_metadata["point_count"], 360)
        self.assertAlmostEqual(default_metadata["max_angle"], ScanConfig().max_angle)


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GridScanTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
