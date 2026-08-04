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
