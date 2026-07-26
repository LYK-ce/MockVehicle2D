"""Finite-view scan matching checks; truth is used only as test oracle."""

import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.local_state import (
    AnchorSpec,
    AnchoredLocalState,
    CorrelativeScanMatcher,
    ObservedGrid,
    PoseEstimate,
    ScanMatchConfig,
)
from mockvehicle2d.scan import LaserPoint, ScanConfig


ANCHOR = AnchorSpec("scan-test", 0.0, 0.0, 0.0)
SCAN_CONFIG = ScanConfig(
    min_angle=-math.pi,
    max_angle=math.pi,
    angle_increment=math.pi / 16,
    max_range=20.0,
)
LANDMARK_SCAN = (
    LaserPoint(-1.20, 3.1, 1.0),
    LaserPoint(-0.65, 5.2, 1.0),
    LaserPoint(-0.15, 7.4, 1.0),
    LaserPoint(0.35, 4.3, 1.0),
    LaserPoint(0.80, 6.6, 1.0),
    LaserPoint(1.35, 3.8, 1.0),
    LaserPoint(2.25, 5.7, 1.0),
)


def pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> PoseEstimate:
    return PoseEstimate("scan-test", x, y, yaw, (0.1, 0.1, 0.02), "nominal", 1.0, 1)


def matcher() -> CorrelativeScanMatcher:
    return CorrelativeScanMatcher(
        ScanMatchConfig(
            xy_window_m=0.6,
            xy_step_m=0.1,
            yaw_window_rad=math.radians(8),
            yaw_step_rad=math.radians(1),
            sample_stride=1,
            min_support=5,
            min_score=0.55,
            min_margin=0.005,
        )
    )


def learned_grid() -> ObservedGrid:
    grid = ObservedGrid(ANCHOR, resolution_m=0.25)
    grid.integrate_scan(LANDMARK_SCAN, pose(), 1.0, SCAN_CONFIG)
    return grid


def pose_error(result, predicted: PoseEstimate) -> float:
    corrected = (
        predicted.x_m + result.correction_x_m,
        predicted.y_m + result.correction_y_m,
        predicted.yaw_rad + result.correction_yaw_rad,
    )
    return math.hypot(corrected[0], corrected[1]) + abs(corrected[2])


def test_scan_match_reduces_translation_and_rotation_drift() -> None:
    predicted = pose(0.4, -0.3, math.radians(5))
    result = matcher().match(LANDMARK_SCAN, predicted, learned_grid())

    assert result.accepted
    assert result.support >= 5
    assert pose_error(result, predicted) < (
        math.hypot(predicted.x_m, predicted.y_m) + abs(predicted.yaw_rad)
    )


def test_scan_match_is_deterministic() -> None:
    predicted = pose(0.3, 0.2, math.radians(-4))
    first = matcher().match(LANDMARK_SCAN, predicted, learned_grid())
    second = matcher().match(LANDMARK_SCAN, predicted, learned_grid())
    assert first == second


@pytest.mark.parametrize(
    ("origin_x", "wall_x"),
    ((0.5, 3.0), (3.5, 1.0)),
)
def test_repeated_boundary_scan_does_not_create_half_cell_drift(
    origin_x: float, wall_x: float
) -> None:
    """A hit on either side of a cell boundary must not pull toward its centre."""
    origin_y = 0.5
    endpoints = ((wall_x, y) for y in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5))
    points = tuple(
        LaserPoint(
            math.atan2(y - origin_y, wall_x - origin_x),
            math.hypot(wall_x - origin_x, y - origin_y),
            1.0,
        )
        for wall_x, y in endpoints
    )
    grid = ObservedGrid(ANCHOR, resolution_m=1.0)
    initial = pose(origin_x, origin_y)
    grid.integrate_scan(points, initial, 1.0, SCAN_CONFIG)

    result = CorrelativeScanMatcher(
        ScanMatchConfig(
            xy_window_m=0.5,
            xy_step_m=0.5,
            yaw_window_rad=0.0,
            yaw_step_rad=1.0,
            sample_stride=1,
            min_support=5,
            min_score=0.5,
            min_margin=0.005,
        )
    ).match(points, initial, grid)

    assert result.correction_x_m == 0.0
    assert result.correction_y_m == 0.0


def test_default_scan_match_has_a_deterministic_work_budget() -> None:
    points = tuple(
        LaserPoint(
            tmini_angle,
            4.0 + (index % 7) * 0.25,
            1.0,
        )
        for index, tmini_angle in enumerate(
            index * (2 * math.pi / 667) for index in range(667)
        )
    )
    grid = ObservedGrid(ANCHOR, resolution_m=0.25)
    grid.integrate_scan(points, pose(), 1.0, SCAN_CONFIG)
    scan_matcher = CorrelativeScanMatcher()

    result = scan_matcher.match(points, pose(), grid)

    assert 0 < result.work_units <= scan_matcher.config.max_work_units


def test_scan_match_budget_holds_when_one_scan_exceeds_it() -> None:
    config = ScanMatchConfig(
        xy_window_m=0.5,
        xy_step_m=0.1,
        yaw_window_rad=math.radians(5),
        yaw_step_rad=math.radians(1),
        sample_stride=1,
        min_support=6,
        min_score=0.0,
        min_margin=0.0,
        max_work_units=6,
    )
    points = tuple(
        LaserPoint(index * (2 * math.pi / 667), 4.0, 1.0)
        for index in range(667)
    )
    grid = ObservedGrid(ANCHOR, resolution_m=0.25)
    grid.integrate_scan(points, pose(), 1.0, SCAN_CONFIG)

    result = CorrelativeScanMatcher(config).match(points, pose(), grid)

    assert result.work_units <= config.max_work_units


def test_scan_match_budget_must_allow_minimum_support() -> None:
    with pytest.raises(ValueError):
        ScanMatchConfig(min_support=7, max_work_units=6)


@pytest.mark.parametrize(
    ("grid", "points", "reason"),
    (
        (ObservedGrid(ANCHOR, resolution_m=0.25), LANDMARK_SCAN, "empty_map"),
        (learned_grid(), (LaserPoint(0.0, 18.0, 1.0),), "insufficient_support"),
        (
            learned_grid(),
            tuple(LaserPoint(point.angle, point.range + 8.0, point.intensity) for point in LANDMARK_SCAN),
            "insufficient_support",
        ),
    ),
)
def test_scan_match_rejects_empty_weak_and_outlier_inputs(
    grid: ObservedGrid, points: tuple[LaserPoint, ...], reason: str
) -> None:
    result = matcher().match(points, pose(0.2), grid)
    assert not result.accepted
    assert result.reason == reason
    assert result.correction_x_m == result.correction_y_m == result.correction_yaw_rad == 0.0


def test_ambiguous_match_is_rejected_by_margin() -> None:
    grid = ObservedGrid(ANCHOR, resolution_m=1.0)
    repeated = tuple(LaserPoint(0.0, 4.2, 1.0) for _ in range(8))
    grid.integrate_scan(repeated, pose(), 1.0, SCAN_CONFIG)
    result = matcher().match(repeated, pose(0.2), grid)
    assert not result.accepted
    assert result.reason in {"insufficient_support", "ambiguous"}


def test_accepted_correction_persists_into_following_odometry() -> None:
    state = AnchoredLocalState(
        ANCHOR,
        truth_x_m=0.0,
        truth_y_m=0.0,
        truth_yaw_rad=0.0,
        scan_match_config=matcher().config,
        timestamp=0.0,
        map_resolution_m=0.25,
    )
    state.local_map.integrate_scan(
        LANDMARK_SCAN, state.pose, 0.0, SCAN_CONFIG
    )
    state.odometry.apply_correction(0.4, -0.3, math.radians(5), timestamp=1.0)

    state.match_and_integrate_scan(LANDMARK_SCAN, 2.0, SCAN_CONFIG)
    corrected_error = math.hypot(state.pose.x_m, state.pose.y_m) + abs(
        state.pose.yaw_rad
    )
    assert state.last_scan_match is not None and state.last_scan_match.accepted
    assert corrected_error < math.hypot(0.4, 0.3) + math.radians(5)

    state.update_from_truth(1.0, 0.0, 0.0, timestamp=3.0)
    assert abs(state.pose.x_m - 1.0) < 0.4
