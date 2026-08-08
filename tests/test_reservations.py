import math

from mockvehicle2d.coordination import (
    ReservationTable,
    TimedCell,
    prioritized_sipp,
)


def test_prioritized_sipp_delays_for_vertex_edge_and_crossing_reservations() -> None:
    table = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    table.add(
        "peer",
        (
            TimedCell((1, -1), 0.0, 0.0),
            TimedCell((1, 1), 2.0, 4.0),
        ),
        base_time_s=0.0,
        radius_m=0.2,
        time_margin_s=0.0,
        goal_hold=True,
    )

    plan = prioritized_sipp(
        ((0, 0), (2, 0)),
        table,
        now_s=0.0,
        horizon_s=6.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    )

    assert plan is not None
    assert plan[0].leave_offset_s >= 2.0
    assert plan[-1].cell == (2, 0)


def test_reservation_table_covers_goal_hold_and_opposite_edge_swap() -> None:
    table = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    table.add(
        "peer",
        (
            TimedCell((1, 0), 0.0, 0.0),
            TimedCell((0, 0), 1.0, 2.0),
        ),
        base_time_s=0.0,
        radius_m=0.2,
        time_margin_s=0.0,
        goal_hold=True,
    )

    assert table.edge_conflict_end((0, 0), (1, 0), 0.0, 1.0) == 1.0
    assert table.cell_conflict_end((0, 0), 3.0, 3.5) == math.inf


def test_no_reservations_preserve_immediate_unconflicted_motion() -> None:
    plan = prioritized_sipp(
        ((0, 0), (1, 0), (2, 0)),
        ReservationTable(0.5, own_radius_m=0.2, clearance_m=0.1),
        now_s=10.0,
        horizon_s=4.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    )

    assert plan is not None
    assert plan[0] == TimedCell((0, 0), 0.0, 0.0)
    assert plan[-1].cell == (2, 0)


def test_reservation_declaration_order_does_not_change_the_schedule() -> None:
    trajectories = {
        "peer_b": (
            TimedCell((1, -1), 0.0, 0.0),
            TimedCell((1, 1), 1.0, 2.0),
        ),
        "peer_a": (
            TimedCell((2, -1), 0.0, 0.0),
            TimedCell((2, 1), 2.0, 3.0),
        ),
    }

    def scheduled(order: tuple[str, ...]) -> tuple[TimedCell, ...] | None:
        table = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
        for source in order:
            table.add(
                source,
                trajectories[source],
                base_time_s=0.0,
                radius_m=0.2,
                time_margin_s=0.0,
                goal_hold=False,
            )
        return prioritized_sipp(
            ((0, 0), (3, 0)),
            table,
            now_s=0.0,
            horizon_s=6.0,
            linear_speed_mps=1.0,
            angular_speed_rps=math.pi,
            initial_yaw_rad=0.0,
            time_margin_s=0.0,
        )

    assert scheduled(("peer_a", "peer_b")) == scheduled(("peer_b", "peer_a"))
