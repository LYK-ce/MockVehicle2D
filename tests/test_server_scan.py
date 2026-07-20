"""Wire-format check for the WebSocket scan frame."""

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.test_collision import main as collision_main
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.scan import scan_message


class ScanMessageTest(unittest.TestCase):
    def test_existing_collision_suite_still_passes(self) -> None:
        self.assertEqual(collision_main(), 0)

    def test_scan_message_contains_laserscan_metadata_and_points(self) -> None:
        grid = MapGrid.from_wall_set(8, 4, {(4, 1)})
        message = scan_message(grid, 1.5, 1.5, 0.0, 1717800000.124)
        self.assertEqual(message["type"], "scan")
        self.assertEqual(message["frame_id"], "laser")
        self.assertEqual(message["config"]["no_return"], {"range": None, "intensity": 0.0})
        self.assertEqual(len(message["points"]), message["config"]["point_count"])
        forward = next(point for point in message["points"] if abs(point["angle"]) < 1e-9)
        self.assertAlmostEqual(forward["range"], 2.5)
        self.assertEqual(forward["intensity"], 1.0)


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScanMessageTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
