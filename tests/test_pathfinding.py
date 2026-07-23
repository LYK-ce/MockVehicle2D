"""
test_pathfinding.py — Tests for A* search and WaypointFollower.
"""

import math
import unittest

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.pathfinding import a_star_search, WaypointFollower
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

    def test_out_of_bounds_raises(self):
        grid = _empty_grid(10, 10)
        with self.assertRaises(ValueError):
            a_star_search(grid, (-1, 0), (5, 5))
        with self.assertRaises(ValueError):
            a_star_search(grid, (0, 0), (10, 5))


# ── WaypointFollower tests ───────────────────────────────────

class TestWaypointFollower(unittest.TestCase):

    def test_forward_when_facing_target(self):
        """AC6: Vehicle facing the next waypoint → cmd=forward."""
        path = [(0, 0), (5, 0)]
        follower = WaypointFollower(path)
        cmd, done = follower.next_cmd(0, 0, 0.0)  # at start, facing +x
        self.assertEqual(cmd, "forward")
        self.assertFalse(done)

    def test_spin_left_when_target_is_left(self):
        path = [(0, 0), (0, 5)]  # target is +y (down in screen coords)
        follower = WaypointFollower(path)
        # At (0,0), facing +x (0°), target is at +y (90° clockwise)
        # delta = 90° positive → spin_left
        cmd, done = follower.next_cmd(0, 0, 0.0)
        self.assertEqual(cmd, "spin_left")

    def test_spin_right_when_target_is_right(self):
        path = [(0, 0), (0, -5)]  # target is -y
        follower = WaypointFollower(path)
        cmd, done = follower.next_cmd(0, 0, 0.0)
        self.assertEqual(cmd, "spin_right")

    def test_arrival_at_goal(self):
        """AC6: Within arrival_distance → stop + reached_goal=True."""
        path = [(0, 0), (5, 0)]
        follower = WaypointFollower(path, arrival_distance=0.5)
        cmd, done = follower.next_cmd(4.8, 0, 0.0)
        self.assertEqual(cmd, "stop")
        self.assertTrue(done)

    def test_waypoint_advance(self):
        """Vehicle advances to next waypoint when close enough."""
        path = [(0, 0), (3, 0), (6, 0)]
        follower = WaypointFollower(path, waypoint_distance=0.5)
        # Close to first waypoint → should target second
        cmd, done = follower.next_cmd(2.7, 0, 0.0)
        self.assertFalse(done)
        self.assertEqual(follower.current_target, (6, 0))

    def test_reset(self):
        path1 = [(0, 0), (5, 0)]
        path2 = [(0, 0), (0, 5)]
        follower = WaypointFollower(path1)
        follower.reset(path2)
        self.assertEqual(follower.goal, (0, 5))

    def test_path_too_short_raises(self):
        with self.assertRaises(ValueError):
            WaypointFollower([(0, 0)])

    def test_facing_opposite_direction(self):
        """Vehicle facing away from target: should turn around (spin_left or spin_right)."""
        path = [(0, 0), (5, 0)]
        follower = WaypointFollower(path)
        cmd, done = follower.next_cmd(0, 0, math.pi)  # facing -x
        self.assertIn(cmd, ("spin_left", "spin_right"))
        self.assertFalse(done)


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAStar))
    suite.addTests(loader.loadTestsFromTestCase(TestWaypointFollower))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
