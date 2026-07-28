"""Anchored odometry and finite-view local-map contracts."""

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.local_state import (
    FREE,
    FORBIDDEN,
    OCCUPIED,
    UNKNOWN,
    AnchorSpec,
    AnchoredLocalState,
    AnchoredOdometry,
    OdometryConfig,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.scan import LaserPoint, ScanConfig


class AnchorAndOdometryTest(unittest.TestCase):
    def test_anchor_round_trip_includes_heading(self) -> None:
        anchor = AnchorSpec("vehicle", 100.0, 50.0, math.pi / 2, 0.2, 0.05)
        global_pose = anchor.anchor_to_global(2.0, -1.0, 0.25)
        self.assertAlmostEqual(global_pose[0], 101.0)
        self.assertAlmostEqual(global_pose[1], 52.0)
        local_pose = anchor.global_to_anchor(*global_pose)
        for actual, expected in zip(local_pose, (2.0, -1.0, 0.25), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_odometry_is_relative_to_the_known_birth_pose(self) -> None:
        odometry = AnchoredOdometry(
            AnchorSpec("vehicle", 100.0, 200.0, 0.0),
            truth_x_m=10.0,
            truth_y_m=20.0,
            truth_yaw_rad=0.0,
            timestamp=1.0,
        )
        pose = odometry.update(11.0, 19.0, 0.2, timestamp=2.0)
        self.assertAlmostEqual(pose.x_m, 1.0)
        self.assertAlmostEqual(pose.y_m, -1.0)
        self.assertAlmostEqual(pose.yaw_rad, 0.2)

    def test_fixed_noise_seed_is_reproducible(self) -> None:
        config = OdometryConfig(0.1, 0.05, 42)
        anchor = AnchorSpec("vehicle", 0.0, 0.0, 0.0)
        first = AnchoredOdometry(
            anchor,
            truth_x_m=0.0,
            truth_y_m=0.0,
            truth_yaw_rad=0.0,
            config=config,
            timestamp=0.0,
        )
        second = AnchoredOdometry(
            anchor,
            truth_x_m=0.0,
            truth_y_m=0.0,
            truth_yaw_rad=0.0,
            config=config,
            timestamp=0.0,
        )
        self.assertEqual(
            first.update(1.0, 1.0, 0.2, timestamp=1.0),
            second.update(1.0, 1.0, 0.2, timestamp=1.0),
        )


class ObservedGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = AnchorSpec("vehicle", 0.0, 0.0, 0.0)
        self.grid = ObservedGrid(self.anchor, resolution_m=1.0)
        self.pose = PoseEstimate(
            self.anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            0,
        )
        self.config = ScanConfig(
            min_angle=0.0,
            max_angle=0.0,
            angle_increment=1.0,
            scan_time=0.1,
            min_range=0.1,
            max_range=5.0,
            range_sample_rate_hz=1,
            scan_rate_hz=10,
        )

    def test_hit_marks_ray_and_endpoint_without_revealing_occlusion(self) -> None:
        delta = self.grid.integrate_scan(
            (LaserPoint(0.0, 2.5, 1.0),),
            self.pose,
            1.0,
            self.config,
        )
        self.assertTrue(delta.changed_cells)
        self.assertEqual(self.grid.get_cell(3, 0), OCCUPIED)
        self.assertEqual(self.grid.get_cell(4, 0), UNKNOWN)

    def test_no_return_marks_only_observed_free_ray(self) -> None:
        self.grid.integrate_scan(
            (LaserPoint(0.0, 0.0, 0.0),),
            self.pose,
            1.0,
            self.config,
        )
        self.assertEqual(self.grid.get_cell(4, 0), FREE)
        self.assertEqual(self.grid.get_cell(6, 0), UNKNOWN)

    def test_forbidden_edge_evidence_wins_over_horizontal_free_space(self) -> None:
        self.grid.integrate_scan(
            (LaserPoint(0.0, 0.0, 0.0),),
            self.pose,
            1.0,
            self.config,
            forbidden_points_vehicle_m=((2.0, 0.0),),
        )
        self.assertEqual(self.grid.get_cell(2, 0), FORBIDDEN)

    def test_revision_changes_only_when_map_content_changes(self) -> None:
        scan = (LaserPoint(0.0, 2.5, 1.0),)
        first = self.grid.integrate_scan(
            scan,
            self.pose,
            1.0,
            self.config,
        )
        second = self.grid.integrate_scan(
            scan,
            self.pose,
            2.0,
            self.config,
        )
        self.assertGreater(first.revision, 0)
        self.assertEqual(second.revision, first.revision)
        self.assertEqual(second.changed_cells, ())

    def test_lost_localization_does_not_write_the_map(self) -> None:
        state = AnchoredLocalState(
            self.anchor,
            truth_x_m=0.5,
            truth_y_m=0.5,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )
        state.set_localization_quality("lost", timestamp=1.0)
        delta = state.integrate_scan(
            (LaserPoint(0.0, 2.5, 1.0),),
            1.0,
            self.config,
        )
        self.assertIsNone(delta)
        self.assertEqual(state.local_map.revision, 0)


if __name__ == "__main__":
    unittest.main()
