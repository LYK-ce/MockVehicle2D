"""Deterministic actuator, motion, watchdog, and collision checks."""

import math

import pytest

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.vehicle import Vehicle


def vehicle(**kwargs) -> Vehicle:
    return Vehicle(5.0, 5.0, now=0.0, **kwargs)


def test_straight_reverse_and_arc_motion_use_installed_si_setpoints() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    straight = vehicle()
    straight.install_drive(0.25, 0.0, 0.0)
    straight.advance(grid, 0.4)
    assert straight.x == pytest.approx(5.06875)
    assert straight.y == pytest.approx(5.0)

    straight.install_drive(-0.25, 0.0, 0.4)
    straight.advance(grid, 0.8)
    assert straight.x == pytest.approx(5.08875)
    assert straight.body_velocities() == pytest.approx((-0.15, 0.0))

    arc = vehicle()
    arc.install_drive(0.4, 0.5, 0.0)
    arc.advance(grid, 0.5)
    assert arc.x > 5.0
    assert arc.y > 5.0
    assert arc.yaw == pytest.approx(0.210211264227)


def test_pure_rotation_is_normalized_without_translation() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    rotating = vehicle(angular_speed=1_000_000.0, command_timeout=2.0)
    rotating.install_drive(0.0, 1_000_000.0, 0.0)

    rotating.advance(grid, 1.0)

    assert (rotating.x, rotating.y) == (5.0, 5.0)
    assert rotating.yaw == pytest.approx(math.pi / 2)


def test_watchdog_integrates_only_until_deadline() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(command_timeout=1.0)
    moving.install_drive(0.5, 0.0, 0.0)

    moving.advance(grid, 1.0)

    assert moving.x == pytest.approx(5.375)
    assert moving.command == "stop"
    assert moving.target_velocities() == (0.0, 0.0)
    assert moving.body_velocities() == (0.5, 0.0)
    assert moving.command_deadline is None

    moving.advance(grid, 2.0)
    assert moving.x == pytest.approx(5.5)
    assert moving.body_velocities() == (0.0, 0.0)


def test_install_drive_validates_limits_and_clock_before_mutating_state() -> None:
    moving = vehicle()
    moving.install_drive(0.25, 0.1, 0.0)
    before = (moving.command, moving.body_velocities(), moving.command_deadline)

    for linear, angular, now in (
        (0.51, 0.0, 0.0),
        (0.0, math.pi, 0.0),
        (math.nan, 0.0, 0.0),
        (True, 0.0, 0.0),
        (0.1, 0.0, 0.1),
    ):
        with pytest.raises(ValueError):
            moving.install_drive(linear, angular, now)
        assert (moving.command, moving.body_velocities(), moving.command_deadline) == before


@pytest.mark.parametrize(
    "parameter",
    (
        "linear_acceleration_mps2",
        "linear_deceleration_mps2",
        "angular_acceleration_rps2",
    ),
)
def test_acceleration_limits_must_be_finite_and_positive(parameter: str) -> None:
    for value in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            vehicle(**{parameter: value})


def test_stop_with_time_discards_unintegrated_motion() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(command_timeout=5.0)
    moving.install_drive(0.5, 0.2, 0.0)

    moving.stop(1.0)
    moving.advance(grid, 2.0)

    assert (moving.x, moving.y, moving.yaw) == (5.0, 5.0, 0.0)
    assert moving.body_velocities() == (0.0, 0.0)


def test_acceleration_braking_and_stop_target_are_bounded() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(command_timeout=5.0)
    moving.install_drive(0.5, 0.0, 0.0)

    moving.advance(grid, 0.25)
    assert moving.body_velocities() == pytest.approx((0.25, 0.0))
    assert moving.x == pytest.approx(5.03125)
    moving.advance(grid, 0.5)
    assert moving.body_velocities() == pytest.approx((0.5, 0.0))

    stopped_at = moving.x
    moving.stop()
    assert moving.target_velocities() == (0.0, 0.0)
    assert moving.body_velocities() == pytest.approx((0.5, 0.0))
    moving.advance(grid, 1.0)
    assert moving.body_velocities() == (0.0, 0.0)
    assert moving.x - stopped_at == pytest.approx(0.125)


def test_reversal_brakes_through_zero_before_accelerating_backwards() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(command_timeout=5.0)
    moving.install_drive(0.5, 0.0, 0.0)
    moving.advance(grid, 0.5)
    moving.install_drive(-0.5, 0.0, 0.5)

    moving.advance(grid, 0.75)
    assert moving.body_velocities()[0] == pytest.approx(0.25)
    moving.advance(grid, 1.0)
    assert moving.body_velocities()[0] == pytest.approx(0.0)
    moving.advance(grid, 1.25)
    assert moving.body_velocities()[0] == pytest.approx(-0.25)


def test_reversal_with_zero_net_distance_retains_the_physical_path() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(command_timeout=5.0)
    moving.install_drive(0.5, 0.0, 0.0)
    moving.advance(grid, 0.5)
    start_x = moving.x
    moving.install_drive(-0.5, 0.0, 0.5)
    trajectory = []

    moving.advance(grid, 1.5, trajectory=trajectory)

    assert moving.x == pytest.approx(start_x)
    assert max(point[1] for point in trajectory) == pytest.approx(start_x + 0.125)
    assert moving.body_velocities()[0] == pytest.approx(-0.5)


def test_low_speed_reversal_detects_wall_crossed_between_matching_endpoints() -> None:
    grid = MapGrid.from_wall_set(20, 20, {(3, 5)})
    moving = Vehicle(
        2.492,
        5.5,
        linear_acceleration_mps2=1.0,
        linear_deceleration_mps2=1.0,
        radius=0.5,
        command_timeout=5.0,
        now=0.0,
    )
    moving.install_drive(0.1, 0.0, 0.0)
    moving.advance(grid, 0.1)
    assert moving.x == pytest.approx(2.497)
    moving.install_drive(-0.1, 0.0, 0.1)

    collided = moving.advance(grid, 0.3)

    assert collided
    assert moving.collision


def test_low_speed_angular_reversal_retains_the_physical_orientation_path() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    rotating = vehicle(
        angular_acceleration_rps2=1.0,
        command_timeout=5.0,
    )
    rotating.install_drive(0.0, 0.1, 0.0)
    rotating.advance(grid, 0.1)
    start_yaw = rotating.yaw
    rotating.install_drive(0.0, -0.1, 0.1)
    trajectory = []

    rotating.advance(grid, 0.3, trajectory=trajectory)

    assert rotating.yaw == pytest.approx(start_yaw)
    assert max(point[3] for point in trajectory) == pytest.approx(start_yaw + 0.005)
    assert rotating.body_velocities() == pytest.approx((0.0, -0.1))


def test_linear_and_angular_breakpoints_share_one_strict_timeline() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    coarse = vehicle(
        linear_speed=1.0,
        angular_speed=1.0,
        linear_acceleration_mps2=1.0,
        linear_deceleration_mps2=1.0,
        angular_acceleration_rps2=1.0,
        command_timeout=5.0,
    )
    coarse.install_drive(0.1, 0.2, 0.0)
    coarse.advance(grid, 0.2)
    coarse.install_drive(-0.1, -0.2, 0.2)
    trajectory = []

    coarse.advance(grid, 0.6, trajectory=trajectory)

    reference = vehicle(
        linear_speed=1.0,
        angular_speed=1.0,
        linear_acceleration_mps2=1.0,
        linear_deceleration_mps2=1.0,
        angular_acceleration_rps2=1.0,
        command_timeout=5.0,
    )
    reference.install_drive(0.1, 0.2, 0.0)
    reference.advance(grid, 0.2)
    reference.install_drive(-0.1, -0.2, 0.2)
    for step in range(201, 601):
        reference.advance(grid, step / 1000)

    assert [point[0] for point in trajectory] == pytest.approx(
        [0.2, 0.3, 0.4, 0.6]
    )
    assert all(
        current[0] < following[0]
        for current, following in zip(trajectory, trajectory[1:])
    )
    assert trajectory[-1][0] == 0.6
    assert (coarse.x, coarse.y, coarse.yaw) == pytest.approx(
        (reference.x, reference.y, reference.yaw),
        abs=1e-4,
    )


def test_nearly_equal_motion_breakpoints_do_not_duplicate_timestamps() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle(
        linear_speed=1.0,
        angular_speed=1.0,
        linear_acceleration_mps2=1.0,
        angular_acceleration_rps2=1.0,
        command_timeout=5.0,
    )
    moving.install_drive(0.1, 0.1000000000005, 0.0)
    trajectory = []

    moving.advance(grid, 0.2, trajectory=trajectory)

    assert [point[0] for point in trajectory] == pytest.approx([0.0, 0.1, 0.2])
    assert all(
        current[0] < following[0]
        for current, following in zip(trajectory, trajectory[1:])
    )


def test_angular_velocity_uses_the_configured_ramp() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    rotating = vehicle(
        angular_speed=2.0,
        angular_acceleration_rps2=1.0,
        command_timeout=5.0,
    )
    rotating.install_drive(0.0, 2.0, 0.0)

    rotating.advance(grid, 0.5)

    assert rotating.body_velocities() == pytest.approx((0.0, 0.5))
    assert rotating.yaw == pytest.approx(0.125)


def test_straight_ramp_and_stopping_distance_are_tick_size_stable() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    results = []
    for tick_s in (0.05, 0.1, 0.25):
        moving = vehicle(command_timeout=5.0)
        moving.install_drive(0.5, 0.0, 0.0)
        now = 0.0
        while now < 0.5 - 1e-12:
            now = min(0.5, now + tick_s)
            moving.advance(grid, now)
        moving.stop()
        while now < 1.0 - 1e-12:
            now = min(1.0, now + tick_s)
            moving.advance(grid, now)
        results.append((moving.x, moving.body_velocities()))

    for position, velocities in results:
        assert position == pytest.approx(5.25)
        assert velocities == (0.0, 0.0)


def test_accelerating_arc_timestamps_and_geometry_match_fine_time_steps() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    coarse = vehicle(
        linear_speed=1.0,
        angular_speed=1.0,
        linear_acceleration_mps2=1.0,
        angular_acceleration_rps2=4.0,
        command_timeout=2.0,
    )
    coarse.install_drive(1.0, 1.0, 0.0)
    trajectory = []
    coarse.advance(grid, 1.0, trajectory=trajectory)

    reference = vehicle(
        linear_speed=1.0,
        angular_speed=1.0,
        linear_acceleration_mps2=1.0,
        angular_acceleration_rps2=4.0,
        command_timeout=2.0,
    )
    reference.install_drive(1.0, 1.0, 0.0)
    for step in range(1, 1001):
        reference.advance(grid, step / 1000)

    assert trajectory[0] == (0.0, 5.0, 5.0, 0.0)
    assert trajectory[-1] == (1.0, coarse.x, coarse.y, coarse.yaw)
    assert all(
        current[0] < following[0]
        for current, following in zip(trajectory, trajectory[1:])
    )
    for current, following in zip(trajectory, trajectory[1:]):
        assert math.dist(current[1:3], following[1:3]) <= 0.25 + 1e-12
        yaw_step = math.atan2(
            math.sin(following[3] - current[3]),
            math.cos(following[3] - current[3]),
        )
        assert abs(yaw_step) <= math.pi / 18 + 1e-12
    assert (coarse.x, coarse.y, coarse.yaw) == pytest.approx(
        (reference.x, reference.y, reference.yaw),
        abs=2e-3,
    )


def test_safety_reduced_velocity_cannot_reverse_or_exceed_active_setpoint() -> None:
    grid = MapGrid.from_wall_set(20, 20, set())
    moving = vehicle()
    moving.install_drive(0.5, 0.2, 0.0)
    moving.advance(grid, 0.5, limited_velocities=(0.25, 0.1))
    assert moving.x > 5.0
    assert moving.x < 5.25

    for limited in ((0.6, 0.1), (-0.1, 0.1), (0.1, -0.1)):
        with pytest.raises(ValueError):
            moving.advance(grid, 0.6, limited_velocities=limited)


def test_collision_stops_at_last_safe_pose_and_clears_setpoint() -> None:
    grid = MapGrid.from_wall_set(20, 20, {(4, y) for y in range(20)})
    moving = Vehicle(
        2.5,
        5.5,
        linear_speed=2.0,
        command_timeout=2.0,
        now=0.0,
    )
    moving.install_drive(2.0, 0.4, 0.0)

    collided = moving.advance(grid, 2.0)

    assert collided
    assert moving.collision
    assert 2.5 < moving.x <= 3.5
    assert moving.y > 5.5
    assert moving.command == "stop"
    assert moving.body_velocities() == (0.0, 0.0)
    assert moving.target_velocities() == (0.0, 0.0)
