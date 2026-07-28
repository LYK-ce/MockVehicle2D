"""RobotController is the only motion and mission authority."""

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    AutoState,
    GotoMission,
    ManualAction,
    ManualCommand,
    ModeAction,
    ModeCommand,
    OpMode,
    RobotController,
)
from mockvehicle2d.local_state import (
    AnchorSpec,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.vehicle import Vehicle


class Harness:
    def __init__(self, *, capacity: int = 2, command_timeout: float = 0.5) -> None:
        self.anchor = AnchorSpec("controller-test", 10.0, 10.0, 0.0)
        self.local_map = ObservedGrid(self.anchor)
        self.pose = PoseEstimate(
            self.anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0,
        )
        self.grid = MapGrid.from_wall_set(64, 64, set())
        self.vehicle = Vehicle(
            10.0,
            10.0,
            command_timeout=command_timeout,
            now=0.0,
        )
        self.safety = LocalSafetyRuntime()
        self.controller = RobotController(mission_capacity=capacity)

    def mode(self, seq: int, action: ModeAction):
        return self.controller.handle(
            ModeCommand(seq, action),
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            now=self.vehicle.last_update,
        )

    def auto(
        self,
        seq: int,
        action: AutoAction,
        missions: tuple[GotoMission, ...] = (),
    ):
        return self.controller.handle(
            AutoCommand(seq, action, missions),
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            now=self.vehicle.last_update,
        )

    def manual(
        self,
        seq: int,
        action: ManualAction,
        linear_mps: float = 0.0,
        angular_rps: float = 0.0,
    ):
        return self.controller.handle(
            ManualCommand(seq, action, linear_mps, angular_rps),
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            now=self.vehicle.last_update,
        )

    def tick(
        self,
        now: float,
        *,
        pose: PoseEstimate | None = None,
        advance: SafetyAdvanceResult | None = None,
    ) -> None:
        result = (
            self.safety.advance(
                self.vehicle,
                self.grid,
                now,
                automatic=self.controller.is_automatic_motion_active,
            )
            if advance is None
            else advance
        )
        self.controller.tick(
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            anchor=self.anchor,
            pose=self.pose if pose is None else pose,
            local_map=self.local_map,
            map_delta=None,
            advance_result=result,
            now=now,
        )


def mission(
    mission_id: str,
    x_m: float,
    y_m: float = 10.0,
    seq: int = 1,
) -> GotoMission:
    return GotoMission(mission_id, "global_map", x_m, y_m, seq)


def case_defaults_to_manual_with_no_hidden_task_authority() -> None:
    harness = Harness()
    snapshot = harness.controller.snapshot()

    assert harness.controller.mode is OpMode.MANUAL
    assert harness.controller.auto_state is AutoState.IDLE
    assert snapshot["active_mission"] is None
    assert snapshot["mission_queue"]["size"] == 0


def case_wrong_mode_commands_are_rejected_without_changing_motion() -> None:
    harness = Harness()

    rejected_auto = harness.auto(1, AutoAction.PUSH, (mission("m1", 12.0),))
    assert not rejected_auto.accepted
    assert rejected_auto.reason == "wrong_mode"

    assert harness.mode(2, ModeAction.SWITCH_TO_AUTO).accepted
    rejected_manual = harness.controller.handle(
        ManualCommand(3, ManualAction.DRIVE, 0.2, 0.1),
        vehicle=harness.vehicle,
        grid=harness.grid,
        safety=harness.safety,
        now=0.0,
    )
    assert not rejected_manual.accepted
    assert rejected_manual.reason == "wrong_mode"
    assert harness.vehicle.body_velocities() == (0.0, 0.0)


def case_manual_takeover_preserves_missions_and_requires_explicit_resume() -> None:
    harness = Harness(capacity=3)
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    harness.auto(
        2,
        AutoAction.PUSH,
        (mission("first", 14.0), mission("second", 18.0)),
    )
    harness.tick(0.0)
    assert harness.controller.active_mission.mission_id == "first"
    assert harness.controller.auto_state is AutoState.ACTIVE
    harness.controller.drain_events()

    harness.mode(3, ModeAction.SWITCH_TO_MANUAL)
    assert harness.controller.mode is OpMode.MANUAL
    assert harness.controller.auto_state is AutoState.PAUSED
    assert harness.controller.active_mission.mission_id == "first"
    assert harness.controller.snapshot()["mission_queue"]["mission_ids"] == ["second"]
    assert harness.vehicle.body_velocities() == (0.0, 0.0)
    paused = harness.controller.drain_events()
    assert [(event.mission.mission_id, event.status) for event in paused] == [
        ("first", "paused")
    ]

    harness.mode(4, ModeAction.SWITCH_TO_AUTO)
    assert harness.controller.drain_events() == ()
    harness.tick(0.1)
    assert harness.controller.auto_state is AutoState.PAUSED
    assert harness.vehicle.body_velocities() == (0.0, 0.0)

    harness.auto(5, AutoAction.RESUME)
    harness.tick(0.2)
    assert harness.controller.auto_state is AutoState.ACTIVE
    assert harness.controller.navigation.status == "active"
    active_mission = harness.controller.active_mission
    harness.mode(6, ModeAction.SWITCH_TO_AUTO)
    assert harness.controller.auto_state is AutoState.ACTIVE
    assert harness.controller.active_mission is active_mission


def case_push_is_atomic_bounded_and_mission_ids_are_idempotent() -> None:
    harness = Harness(capacity=2)
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    accepted = harness.auto(
        2,
        AutoAction.PUSH,
        (mission("one", 11.0), mission("two", 12.0)),
    )
    assert accepted.accepted
    harness.controller.drain_events()

    duplicate = harness.auto(
        3,
        AutoAction.PUSH,
        (mission("one", 11.0),),
    )
    assert duplicate.accepted
    assert duplicate.reason is None
    assert harness.controller.drain_events() == ()

    conflict = harness.auto(
        4,
        AutoAction.PUSH,
        (mission("one", 99.0),),
    )
    assert not conflict.accepted
    assert conflict.reason == "mission_id_conflict"

    full = harness.auto(
        5,
        AutoAction.PUSH,
        (mission("three", 13.0),),
    )
    assert not full.accepted
    assert full.reason == "mission_queue_full"
    assert harness.controller.snapshot()["mission_queue"]["mission_ids"] == [
        "one",
        "two",
    ]


def case_cancel_all_cancels_active_and_pending_missions() -> None:
    harness = Harness(capacity=3)
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    harness.auto(
        2,
        AutoAction.PUSH,
        (mission("one", 14.0), mission("two", 18.0)),
    )
    harness.controller.drain_events()
    harness.tick(0.0)
    harness.controller.drain_events()

    result = harness.auto(3, AutoAction.CANCEL_ALL)
    events = harness.controller.drain_events()
    assert result.accepted
    assert [event.mission.mission_id for event in events] == ["one", "two"]
    assert {event.status for event in events} == {"cancelled"}
    assert harness.controller.active_mission is None
    assert harness.controller.auto_state is AutoState.IDLE
    assert harness.controller.snapshot()["mission_queue"]["size"] == 0


def case_blocked_mission_retains_queue_and_does_not_skip() -> None:
    harness = Harness(capacity=3)
    lost = PoseEstimate(
        harness.anchor.anchor_id,
        0.0,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        "lost",
        0.0,
        1,
    )
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    harness.auto(
        2,
        AutoAction.PUSH,
        (mission("blocked", 14.0), mission("waiting", 18.0)),
    )
    harness.tick(0.0, pose=lost)

    assert harness.controller.auto_state is AutoState.BLOCKED
    assert harness.controller.active_mission.mission_id == "blocked"
    assert harness.controller.snapshot()["mission_queue"]["mission_ids"] == [
        "waiting"
    ]
    assert harness.controller.navigation.snapshot()["requested_goal"] == {
        "frame_id": "anchor_map",
        "x_m": 4.0,
        "y_m": 0.0,
    }
    assert harness.vehicle.body_velocities() == (0.0, 0.0)


def case_zero_distance_missions_complete_in_queue_order() -> None:
    harness = Harness(capacity=3)
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    harness.auto(
        2,
        AutoAction.PUSH,
        (mission("one", 10.0), mission("two", 10.0)),
    )
    harness.controller.drain_events()

    harness.tick(0.0)
    first = harness.controller.drain_events()
    harness.tick(0.1)
    second = harness.controller.drain_events()

    assert [(event.mission.mission_id, event.status) for event in first] == [
        ("one", "active"),
        ("one", "reached"),
    ]
    assert [(event.mission.mission_id, event.status) for event in second] == [
        ("two", "active"),
        ("two", "reached"),
    ]
    assert harness.controller.auto_state is AutoState.IDLE


def case_manual_setpoint_refreshes_watchdog_then_expires_without_resume() -> None:
    harness = Harness(command_timeout=0.5)
    result = harness.manual(1, ManualAction.DRIVE, 0.2, 0.1)
    assert result.accepted
    assert harness.vehicle.body_velocities() == (0.2, 0.1)

    harness.tick(0.25)
    assert harness.vehicle.body_velocities() == (0.2, 0.1)
    assert math.isclose(harness.vehicle.command_deadline, 0.75)

    harness.tick(0.5)
    assert harness.vehicle.body_velocities() == (0.0, 0.0)
    harness.tick(1.0)
    assert harness.vehicle.body_velocities() == (0.0, 0.0)


def case_manual_and_auto_are_both_gated_by_safety() -> None:
    harness = Harness()
    harness.grid = MapGrid.from_wall_set(64, 64, {(11, y) for y in range(64)})
    harness.vehicle.x = 10.3

    rejected = harness.manual(1, ManualAction.DRIVE, 0.5, 0.0)
    assert not rejected.accepted
    assert rejected.reason == "safety_obstacle"
    assert harness.vehicle.body_velocities() == (0.0, 0.0)

    harness.mode(2, ModeAction.SWITCH_TO_AUTO)
    harness.auto(3, AutoAction.PUSH, (mission("safe", 14.0),))
    harness.tick(0.0)
    assert harness.vehicle.body_velocities() == (0.0, 0.0)
    assert harness.safety.snapshot()["reason"] == "safety_obstacle"


def case_disconnect_pauses_auto_and_stops_manual() -> None:
    harness = Harness()
    harness.manual(1, ManualAction.DRIVE, 0.2, 0.0)
    harness.controller.disconnect(harness.vehicle)
    assert harness.vehicle.body_velocities() == (0.0, 0.0)

    harness.mode(2, ModeAction.SWITCH_TO_AUTO)
    harness.auto(3, AutoAction.PUSH, (mission("one", 14.0),))
    harness.tick(0.0)
    harness.controller.disconnect(harness.vehicle)
    assert harness.controller.auto_state is AutoState.PAUSED
    assert harness.controller.active_mission.mission_id == "one"
    assert harness.vehicle.body_velocities() == (0.0, 0.0)


def case_stop_motion_is_global_task_preserving_and_idempotent() -> None:
    harness = Harness(capacity=3)
    harness.manual(1, ManualAction.DRIVE, 0.2, 0.1)

    assert harness.mode(2, ModeAction.STOP_MOTION).accepted
    assert harness.controller.mode is OpMode.MANUAL
    assert harness.vehicle.body_velocities() == (0.0, 0.0)
    assert not harness.controller.snapshot()["manual_setpoint_active"]
    assert harness.mode(3, ModeAction.STOP_MOTION).accepted
    assert harness.controller.drain_events() == ()

    harness.mode(4, ModeAction.SWITCH_TO_AUTO)
    harness.auto(
        5,
        AutoAction.PUSH,
        (mission("active", 14.0), mission("queued", 18.0)),
    )
    harness.tick(0.0)
    harness.controller.drain_events()

    assert harness.mode(6, ModeAction.STOP_MOTION).accepted
    assert harness.controller.mode is OpMode.AUTO
    assert harness.controller.auto_state is AutoState.PAUSED
    assert harness.controller.active_mission.mission_id == "active"
    assert harness.controller.snapshot()["mission_queue"]["mission_ids"] == [
        "queued"
    ]
    assert harness.vehicle.body_velocities() == (0.0, 0.0)
    paused = harness.controller.drain_events()
    assert [(event.status, event.reason) for event in paused] == [
        ("paused", "stop_motion")
    ]

    assert harness.mode(7, ModeAction.STOP_MOTION).accepted
    assert harness.controller.drain_events() == ()


def case_resume_is_idempotent_while_auto_is_active() -> None:
    harness = Harness()
    harness.mode(1, ModeAction.SWITCH_TO_AUTO)
    harness.auto(2, AutoAction.PUSH, (mission("active", 14.0),))
    harness.tick(0.0)
    harness.controller.drain_events()
    before = (
        harness.controller.active_mission,
        harness.controller.navigation.snapshot(),
        harness.vehicle.body_velocities(),
    )

    assert harness.auto(3, AutoAction.RESUME).accepted
    after = (
        harness.controller.active_mission,
        harness.controller.navigation.snapshot(),
        harness.vehicle.body_velocities(),
    )
    assert after == before
    assert harness.controller.drain_events() == ()


class TestRobotController(unittest.TestCase):
    test_defaults = staticmethod(case_defaults_to_manual_with_no_hidden_task_authority)
    test_wrong_mode = staticmethod(
        case_wrong_mode_commands_are_rejected_without_changing_motion
    )
    test_manual_takeover = staticmethod(
        case_manual_takeover_preserves_missions_and_requires_explicit_resume
    )
    test_atomic_queue = staticmethod(
        case_push_is_atomic_bounded_and_mission_ids_are_idempotent
    )
    test_cancel_all = staticmethod(case_cancel_all_cancels_active_and_pending_missions)
    test_blocked_queue = staticmethod(
        case_blocked_mission_retains_queue_and_does_not_skip
    )
    test_queue_order = staticmethod(case_zero_distance_missions_complete_in_queue_order)
    test_manual_watchdog = staticmethod(
        case_manual_setpoint_refreshes_watchdog_then_expires_without_resume
    )
    test_safety_gate = staticmethod(case_manual_and_auto_are_both_gated_by_safety)
    test_disconnect = staticmethod(case_disconnect_pauses_auto_and_stops_manual)
    test_stop_motion = staticmethod(
        case_stop_motion_is_global_task_preserving_and_idempotent
    )
    test_active_resume = staticmethod(case_resume_is_idempotent_while_auto_is_active)


if __name__ == "__main__":
    unittest.main()
