"""Tests for the full-truth A* debugging helper."""

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.map_grid import MapGrid, VOID
from mockvehicle2d.pathfinding import a_star_search
from mockvehicle2d.pathfinding.a_star import _inflate_blocked


# ── helpers ──────────────────────────────────────────────────

def _empty_grid(w: int = 10, h: int = 10) -> MapGrid:
    return MapGrid(w, h)


def _grid_from_walls(w: int, h: int, walls: set[tuple[int, int]]) -> MapGrid:
    return MapGrid.from_wall_set(w, h, walls)


# ── A* search tests ──────────────────────────────────────────

class TestAStar(unittest.TestCase):

    def test_straight_line_empty(self):
        """AC1: Start→goal on empty grid returns straight-ish shortest path."""
        grid = _empty_grid(10, 10)
        path = a_star_search(grid, (0, 0), (5, 5))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 5))
        # With diagonal moves the length should be ~6 (incl. start)
        self.assertLessEqual(len(path), 7)

    def test_same_start_goal(self):
        grid = _empty_grid(10, 10)
        path = a_star_search(grid, (3, 3), (3, 3))
        self.assertIsNotNone(path)
        self.assertEqual(path, [(3, 3)])

    def test_no_path_walled_goal(self):
        """AC3: Goal surrounded by walls → None."""
        walls = {(4, 3), (4, 4), (4, 5), (5, 3), (5, 5), (6, 3), (6, 4), (6, 5)}
        grid = _grid_from_walls(10, 10, walls)
        path = a_star_search(grid, (0, 0), (5, 4), vehicle_radius=0)
        self.assertIsNone(path)

    def test_path_avoids_walls(self):
        """AC2: Path does not pass through walls."""
        walls = {(5, y) for y in range(3, 8)}  # vertical wall at x=5
        grid = _grid_from_walls(10, 10, walls)
        path = a_star_search(grid, (2, 5), (8, 5), vehicle_radius=0)
        self.assertIsNotNone(path)
        for x, y in path:
            self.assertFalse(grid.is_wall(x, y))

    def test_path_goes_around_wall(self):
        """Path must detour around a blocking wall."""
        walls = {(5, y) for y in range(0, 8)}  # wall blocks direct route
        grid = _grid_from_walls(10, 10, walls)
        path = a_star_search(grid, (2, 5), (8, 5), vehicle_radius=0)
        self.assertIsNotNone(path)
        # Path must cross x=5 at y >= 8 (above the wall)
        # Actually the path goes from left to right, need to find where it crosses
        crossed = False
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            if (x1 < 5 and x2 >= 5) or (x1 > 5 and x2 <= 5):
                crossed = True
                # Check crossing is above wall
                self.assertGreaterEqual(min(y1, y2), 8,
                    f"path crossed wall at y={min(y1, y2)}")
        self.assertTrue(crossed, "path never crossed the x=5 line")

    def test_diagonal_corner_cutting(self):
        """AC4: Diagonal move blocked when corner cells are walls."""
        # . W
        # F .    (F=free, W=wall, start at bottom-left, goal at top-right)
        walls = {(1, 0)}
        grid = _grid_from_walls(3, 3, walls)
        path = a_star_search(grid, (0, 0), (1, 1), vehicle_radius=0)
        self.assertIsNotNone(path)
        # Should not go directly (0,0)→(1,1) because (1,0) is wall
        self.assertNotIn((1, 1), path[1:2], "should not cut corner diagonally")
        # Path should go (0,0)→(0,1)→(1,1) or (0,0)→(1,0) blocked → detour
        if len(path) > 2:
            # Check path length suggests detour
            pass  # corner-cutting prevention verified by exclusion above

    def test_diagonal_allowed_when_corners_free(self):
        """Diagonal is allowed when both cardinal neighbours are free."""
        grid = _empty_grid(3, 3)
        path = a_star_search(grid, (0, 0), (1, 1), vehicle_radius=0)
        self.assertIsNotNone(path)
        # Can be (0,0)→(1,1) directly
        self.assertLessEqual(len(path), 3)

    def test_inflation_keeps_distance(self):
        """AC5: With vehicle_radius=0.5, path stays 1 cell away from walls."""
        # Single wall in middle, try to pass next to it
        walls = {(5, 5)}
        grid = _grid_from_walls(10, 10, walls)
        path = a_star_search(grid, (0, 0), (9, 9), vehicle_radius=0.5)
        self.assertIsNotNone(path)
        blocked = _inflate_blocked(grid)
        for x, y in path:
            self.assertNotIn((x, y), blocked,
                f"path waypoint {x},{y} in inflated blocked zone")

    def test_start_inflated_blocked(self):
        """Start on inflated cell → None."""
        walls = {(1, 1)}
        grid = _grid_from_walls(10, 10, walls)
        # (1, 0) is adjacent to wall (1,1), so inflated
        path = a_star_search(grid, (1, 0), (9, 9), vehicle_radius=0.5)
        self.assertIsNone(path)

    def test_void_cells_are_blocked_when_inflation_is_enabled(self):
        grid = _empty_grid(5, 5)
        for y in range(grid.height):
            grid.set_cell(2, y, VOID)

        path = a_star_search(grid, (0, 2), (4, 2), vehicle_radius=0.5)

        self.assertIsNone(path)

    def test_radius_is_converted_to_resolution_aware_inflation_cells(self):
        grid = _grid_from_walls(8, 8, {(2, 3)})

        path = a_star_search(
            grid,
            (0, 3),
            (0, 7),
            vehicle_radius=1.1,
            resolution_m=1.0,
        )

        self.assertIsNone(path)
        self.assertIn((0, 3), _inflate_blocked(grid, inflation_cells=2))

    def test_out_of_bounds_raises(self):
        grid = _empty_grid(10, 10)
        with self.assertRaises(ValueError):
            a_star_search(grid, (-1, 0), (5, 5))
        with self.assertRaises(ValueError):
            a_star_search(grid, (0, 0), (10, 5))


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAStar))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
