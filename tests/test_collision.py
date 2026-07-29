"""Direct checks for the continuous circular-vehicle collision primitive."""

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid, VOID


def test_sweep_rejects_wall_crossing_but_accepts_clear_motion() -> None:
    grid = MapGrid.from_wall_set(10, 10, {(5, 5)})

    assert is_swept_circle_passable(grid, 2.5, 2.5, 4.5, 2.5, 0.5)
    assert not is_swept_circle_passable(grid, 4.0, 5.5, 6.5, 5.5, 0.5)


def test_exact_tangency_is_safe_but_any_penetration_is_blocked() -> None:
    grid = MapGrid.from_wall_set(10, 10, {(5, 5)})

    assert is_swept_circle_passable(grid, 4.5, 4.0, 4.5, 7.0, 0.5)
    assert not is_swept_circle_passable(grid, 4.51, 4.0, 4.51, 7.0, 0.5)


def test_zero_length_sweep_checks_void_and_world_boundary() -> None:
    grid = MapGrid(4, 4)
    grid.set_cell(2, 1, VOID)

    assert not is_swept_circle_passable(grid, 2.5, 1.5, 2.5, 1.5, 0.4)
    assert is_swept_circle_passable(grid, 0.5, 2.0, 0.5, 2.0, 0.5)
    assert not is_swept_circle_passable(grid, 0.49, 2.0, 0.49, 2.0, 0.5)
