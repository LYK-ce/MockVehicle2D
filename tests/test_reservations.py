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


def test_sipp_reschedules_predecessor_when_an_intermediate_turn_is_unsafe() -> None:
    table = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    table.add(
        "peer",
        (TimedCell((1, 0), 1.2, 1.4),),
        base_time_s=0.0,
        radius_m=0.2,
        time_margin_s=0.0,
        goal_hold=False,
    )

    plan = prioritized_sipp(
        ((0, 0), (1, 0), (1, 1)),
        table,
        now_s=0.0,
        horizon_s=4.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    )

    assert plan is not None
    assert plan[1].enter_offset_s >= 1.4
    assert table.cell_conflict_end(
        plan[1].cell,
        plan[1].enter_offset_s,
        plan[1].leave_offset_s,
    ) is None


def test_sipp_backtracks_when_edge_wait_outlives_intermediate_safe_interval() -> None:
    table = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    for source, cell, enter, leave in (
        ("middle_peer", (1, 0), 1.5, 2.0),
        ("goal_peer", (2, 0), 1.5, 2.5),
    ):
        table.add(
            source,
            (TimedCell(cell, enter, leave),),
            base_time_s=0.0,
            radius_m=0.2,
            time_margin_s=0.0,
            goal_hold=False,
        )

    plan = prioritized_sipp(
        ((0, 0), (1, 0), (2, 0)),
        table,
        now_s=0.0,
        horizon_s=5.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    )

    assert plan is not None
    assert plan[1].enter_offset_s >= 2.0
    assert plan[-1].cell == (2, 0)
    assert plan[-1].leave_offset_s == 5.0
    assert all(
        table.cell_conflict_end(
            timed.cell,
            timed.enter_offset_s,
            timed.leave_offset_s,
        ) is None
        for timed in plan
    )
    assert all(
        table.edge_conflict_end(
            first.cell,
            second.cell,
            first.leave_offset_s,
            second.enter_offset_s,
        ) is None
        for first, second in zip(plan, plan[1:])
    )


def test_sipp_delays_final_goal_hold_or_reports_no_safe_plan() -> None:
    finite = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    finite.add(
        "peer",
        (TimedCell((1, 0), 0.5, 2.0),),
        base_time_s=0.0,
        radius_m=0.2,
        time_margin_s=0.0,
        goal_hold=False,
    )
    plan = prioritized_sipp(
        ((0, 0), (1, 0)),
        finite,
        now_s=0.0,
        horizon_s=4.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    )

    assert plan is not None
    assert plan[-1].cell == (1, 0)
    assert plan[-1].enter_offset_s >= 2.0
    assert plan[-1].leave_offset_s == 4.0

    permanent = ReservationTable(1.0, own_radius_m=0.2, clearance_m=0.1)
    permanent.add(
        "peer",
        (TimedCell((1, 0), 0.5, 2.0),),
        base_time_s=0.0,
        radius_m=0.2,
        time_margin_s=0.0,
        goal_hold=True,
    )
    assert prioritized_sipp(
        ((0, 0), (1, 0)),
        permanent,
        now_s=0.0,
        horizon_s=4.0,
        linear_speed_mps=1.0,
        angular_speed_rps=math.pi,
        initial_yaw_rad=0.0,
        time_margin_s=0.0,
    ) is None
