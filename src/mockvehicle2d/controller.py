"""Single authority for manual motion and autonomous missions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import TYPE_CHECKING
import uuid

from mockvehicle2d.navigation import GotoController

if TYPE_CHECKING:
    from mockvehicle2d.local_state import (
        AnchorSpec,
        LocalMapDelta,
        ObservedGrid,
        PoseEstimate,
    )
    from mockvehicle2d.map_grid import MapGrid
    from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
    from mockvehicle2d.vehicle import Vehicle


MISSION_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
MISSION_FRAME = "global_map"


class OpMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class AutoState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"


class ModeAction(str, Enum):
    SWITCH_TO_MANUAL = "switch_to_manual"
    SWITCH_TO_AUTO = "switch_to_auto"
    STOP_MOTION = "stop_motion"


class ManualAction(str, Enum):
    DRIVE = "drive"
    STOP = "stop"


class AutoAction(str, Enum):
    PUSH = "push"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL_ALL = "cancel_all"


@dataclass(frozen=True)
class GotoMission:
    mission_id: str
    frame_id: str
    x_m: float
    y_m: float
    submitted_seq: int

    def __post_init__(self) -> None:
        if not MISSION_ID_PATTERN.fullmatch(self.mission_id):
            raise ValueError("invalid mission_id")
        if self.frame_id != MISSION_FRAME:
            raise ValueError(f"frame_id must be {MISSION_FRAME}")
        if (
            isinstance(self.x_m, bool)
            or isinstance(self.y_m, bool)
            or not math.isfinite(self.x_m)
            or not math.isfinite(self.y_m)
        ):
            raise ValueError("mission coordinates must be finite")
        if (
            isinstance(self.submitted_seq, bool)
            or not isinstance(self.submitted_seq, int)
            or not 0 <= self.submitted_seq <= 2**64 - 1
        ):
            raise ValueError("submitted_seq must be an unsigned 64-bit integer")

    @property
    def fingerprint(self) -> tuple[str, float, float]:
        return self.frame_id, self.x_m, self.y_m

    def as_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "type": "goto",
            "frame_id": self.frame_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "submitted_seq": self.submitted_seq,
        }


@dataclass(frozen=True)
class ModeCommand:
    seq: int
    action: ModeAction


@dataclass(frozen=True)
class ManualCommand:
    seq: int
    action: ManualAction
    linear_mps: float = 0.0
    angular_rps: float = 0.0


@dataclass(frozen=True)
class AutoCommand:
    seq: int
    action: AutoAction
    missions: tuple[GotoMission, ...] = ()


Command = ModeCommand | ManualCommand | AutoCommand


@dataclass(frozen=True)
class ControllerEvent:
    event_seq: int
    event_epoch: str
    mission: GotoMission
    status: str
    reason: str | None = None
    detail: str | None = None
    navigation: dict[str, object] | None = None

    def as_dict(self, timestamp: float) -> dict[str, object]:
        message: dict[str, object] = {
            "type": "mission_update",
            "event_seq": self.event_seq,
            "event_epoch": self.event_epoch,
            "timestamp_s": timestamp,
            "mission_id": self.mission.mission_id,
            "submitted_seq": self.mission.submitted_seq,
            "status": self.status,
            "goal": {
                "frame_id": self.mission.frame_id,
                "x_m": self.mission.x_m,
                "y_m": self.mission.y_m,
            },
        }
        if self.reason is not None:
            message["reason"] = self.reason
        if self.detail is not None:
            message["detail"] = self.detail
        if self.navigation is not None:
            message["navigation"] = self.navigation
        return message


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    reason: str | None = None


class RobotController:
    """Own mode, mission lifecycle, navigation and the only actuator output."""

    def __init__(
        self,
        navigation: GotoController | None = None,
        *,
        mission_capacity: int = 16,
    ) -> None:
        if isinstance(mission_capacity, bool) or not isinstance(mission_capacity, int):
            raise ValueError("mission_capacity must be an integer")
        if mission_capacity <= 0:
            raise ValueError("mission_capacity must be positive")
        self.mode = OpMode.MANUAL
        self.auto_state = AutoState.IDLE
        self.navigation = navigation or GotoController()
        self.mission_capacity = mission_capacity
        self.active_mission: GotoMission | None = None
        self._pending: deque[GotoMission] = deque()
        # ponytail: process-lifetime ledgers fit the simulator; add persistence and
        # explicit retention only when long-running deployment volume requires it.
        self._mission_history: dict[str, tuple[str, float, float]] = {}
        self._events: list[ControllerEvent] = []
        self.event_epoch = uuid.uuid4().hex
        self._manual_setpoint: tuple[float, float] | None = None
        self._manual_deadline: float | None = None
        self._needs_start = False

    @property
    def is_automatic_motion_active(self) -> bool:
        return self.mode is OpMode.AUTO and self.auto_state is AutoState.ACTIVE

    def handle(
        self,
        command: Command,
        *,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
    ) -> CommandResult:
        if isinstance(command, ModeCommand):
            return self._handle_mode(command, vehicle, now)
        if isinstance(command, ManualCommand):
            return self._handle_manual(command, vehicle, grid, safety, now)
        if isinstance(command, AutoCommand):
            return self._handle_auto(command, vehicle, now)
        raise TypeError("unsupported controller command")

    def tick(
        self,
        *,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        map_delta: LocalMapDelta | None,
        advance_result: SafetyAdvanceResult,
        now: float,
    ) -> None:
        if self.mode is OpMode.MANUAL:
            self._tick_manual(vehicle, grid, safety, now)
            return
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.auto_state is not AutoState.ACTIVE:
            vehicle.stop()
            return

        if self._needs_start:
            self._start_or_resume(anchor, pose, local_map, vehicle.radius)
        if self.auto_state is not AutoState.ACTIVE or self.active_mission is None:
            vehicle.stop()
            return

        desired = self.navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=vehicle.linear_speed,
            max_angular_rps=vehicle.angular_speed,
            advance_result=advance_result,
            map_delta=map_delta,
            safety=safety,
        )
        if self.navigation.status == "reached":
            self._finish_reached(vehicle)
            return
        if self.navigation.status == "blocked":
            self._finish_blocked(vehicle)
            return

        decision = safety.evaluate(
            vehicle,
            grid,
            desired[0],
            desired[1],
            automatic=True,
        )
        if decision.state in {"stopped", "fault"}:
            vehicle.stop()
            if decision.state == "fault":
                self.navigation.block(decision.reason or "safety_sensor_fault")
                self._finish_blocked(vehicle)
            return
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)

    def disconnect(self, vehicle: Vehicle) -> None:
        vehicle.stop()
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.mode is OpMode.AUTO and (
            self.active_mission is not None or self._pending
        ):
            self._pause_active("controller_disconnected")

    def fail_safe_stop(self, vehicle: Vehicle, reason: str) -> None:
        vehicle.stop()
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.mode is OpMode.AUTO and (
            self.active_mission is not None or self._pending
        ):
            self._pause_active(reason)

    @property
    def latest_event_seq(self) -> int:
        return len(self._events)

    def events_after(self, event_seq: int) -> tuple[ControllerEvent, ...]:
        if (
            isinstance(event_seq, bool)
            or not isinstance(event_seq, int)
            or event_seq < 0
        ):
            raise ValueError("event_seq must be a non-negative integer")
        return tuple(self._events[event_seq:])

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "auto_state": self.auto_state.value,
            "active_mission": (
                None if self.active_mission is None else self.active_mission.as_dict()
            ),
            "mission_queue": {
                "size": len(self._pending),
                "capacity": self.mission_capacity,
                "mission_ids": [mission.mission_id for mission in self._pending],
            },
            "manual_setpoint_active": self._manual_setpoint is not None,
            "navigation": self.navigation.snapshot(),
            "mission_events": {
                "event_epoch": self.event_epoch,
                "latest_event_seq": self.latest_event_seq,
                "retention": "process_lifetime",
            },
        }

    def _handle_mode(
        self,
        command: ModeCommand,
        vehicle: Vehicle,
        now: float,
    ) -> CommandResult:
        if command.action is ModeAction.STOP_MOTION:
            vehicle.stop(now)
            self._manual_setpoint = None
            self._manual_deadline = None
            if self.mode is OpMode.AUTO:
                if self.active_mission is not None or self._pending:
                    self._pause_active("stop_motion")
                else:
                    self.auto_state = AutoState.IDLE
            return CommandResult(True)

        if command.action is ModeAction.SWITCH_TO_MANUAL:
            if self.mode is OpMode.MANUAL:
                return CommandResult(True)
            vehicle.stop(now)
            self._manual_setpoint = None
            self._manual_deadline = None
            if self.active_mission is not None or self._pending:
                self._pause_active("manual_takeover")
            else:
                self.auto_state = AutoState.IDLE
            self.mode = OpMode.MANUAL
            return CommandResult(True)

        if self.mode is OpMode.AUTO:
            return CommandResult(True)
        vehicle.stop(now)
        self._manual_setpoint = None
        self._manual_deadline = None
        self.mode = OpMode.AUTO
        if self.active_mission is not None or self._pending:
            self.auto_state = AutoState.PAUSED
            self._needs_start = False
        else:
            self.auto_state = AutoState.IDLE
        return CommandResult(True)

    def _handle_manual(
        self,
        command: ManualCommand,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
    ) -> CommandResult:
        if self.mode is not OpMode.MANUAL:
            return CommandResult(False, "wrong_mode")
        if command.action is ManualAction.STOP:
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop(now)
            return CommandResult(True)

        desired = command.linear_mps, command.angular_rps
        decision = safety.evaluate(vehicle, grid, *desired, automatic=False)
        if decision.state in {"stopped", "fault"}:
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop(now)
            return CommandResult(False, decision.reason or "safety_rejected")
        self._manual_setpoint = desired
        self._manual_deadline = now + vehicle.command_timeout
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)
        return CommandResult(True)

    def _handle_auto(
        self,
        command: AutoCommand,
        vehicle: Vehicle,
        now: float,
    ) -> CommandResult:
        if self.mode is not OpMode.AUTO:
            return CommandResult(False, "wrong_mode")
        if command.action is AutoAction.PUSH:
            return self._push(command.missions)
        if command.action is AutoAction.PAUSE:
            vehicle.stop(now)
            if self.active_mission is not None or self._pending:
                self._pause_active("paused")
            else:
                self.auto_state = AutoState.IDLE
                self._needs_start = False
            return CommandResult(True)
        if command.action is AutoAction.RESUME:
            if self.auto_state is AutoState.ACTIVE:
                return CommandResult(True)
            vehicle.stop(now)
            if self.active_mission is None and not self._pending:
                self.auto_state = AutoState.IDLE
                self._needs_start = False
                return CommandResult(True)
            self.auto_state = AutoState.ACTIVE
            self._needs_start = True
            return CommandResult(True)

        vehicle.stop(now)
        self._cancel_all("cancelled")
        return CommandResult(True)

    def _push(self, missions: tuple[GotoMission, ...]) -> CommandResult:
        if not missions:
            return CommandResult(False, "empty_mission_batch")
        ids = [mission.mission_id for mission in missions]
        if len(ids) != len(set(ids)):
            return CommandResult(False, "duplicate_mission_id")
        new_missions = []
        for mission in missions:
            known = self._mission_history.get(mission.mission_id)
            if known is None:
                new_missions.append(mission)
                continue
            if known != mission.fingerprint:
                return CommandResult(False, "mission_id_conflict")
        if len(self._pending) + len(new_missions) > self.mission_capacity:
            return CommandResult(False, "mission_queue_full")

        for mission in new_missions:
            self._pending.append(mission)
            self._mission_history[mission.mission_id] = mission.fingerprint
            self._emit(mission, "queued")
        if new_missions and self.auto_state is AutoState.IDLE:
            self.auto_state = AutoState.ACTIVE
            self._needs_start = True
        return CommandResult(True)

    def _tick_manual(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
    ) -> None:
        if (
            self._manual_setpoint is None
            or self._manual_deadline is None
            or now >= self._manual_deadline
        ):
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop()
            return
        decision = safety.evaluate(
            vehicle,
            grid,
            *self._manual_setpoint,
            automatic=False,
        )
        if decision.state in {"stopped", "fault"}:
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop()
            return
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)

    def _start_or_resume(
        self,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        vehicle_radius_m: float,
    ) -> None:
        self._needs_start = False
        if self.active_mission is None:
            if not self._pending:
                self.auto_state = AutoState.IDLE
                return
            self.active_mission = self._pending.popleft()
        mission = self.active_mission
        local_x_m, local_y_m, _ = anchor.global_to_anchor(
            mission.x_m, mission.y_m
        )
        try:
            self.navigation.start(
                local_x_m,
                local_y_m,
                reported_goal=(mission.x_m, mission.y_m),
                local_map=local_map,
                pose=pose,
                vehicle_radius_m=vehicle_radius_m,
            )
        except ValueError as error:
            self.navigation.block("invalid_goal", str(error))
            self._finish_blocked_without_vehicle()
            return
        if pose.quality == "lost":
            self.navigation.block("localization_lost")
            self._finish_blocked_without_vehicle()
            return
        self._emit(
            mission,
            "active",
            navigation=self.navigation.snapshot(),
        )

    def _finish_reached(self, vehicle: Vehicle) -> None:
        assert self.active_mission is not None
        mission = self.active_mission
        vehicle.stop()
        self._emit(
            mission,
            "reached",
            self.navigation.reason,
            self.navigation.detail,
            self.navigation.snapshot(),
        )
        self.active_mission = None
        self._needs_start = bool(self._pending)
        self.auto_state = (
            AutoState.ACTIVE if self._needs_start else AutoState.IDLE
        )

    def _finish_blocked(self, vehicle: Vehicle) -> None:
        vehicle.stop()
        self._finish_blocked_without_vehicle()

    def _finish_blocked_without_vehicle(self) -> None:
        assert self.active_mission is not None
        self.auto_state = AutoState.BLOCKED
        self._needs_start = False
        self._emit(
            self.active_mission,
            "blocked",
            self.navigation.reason or "blocked",
            self.navigation.detail,
            self.navigation.snapshot(),
        )

    def _pause_active(self, reason: str) -> None:
        if self.auto_state is AutoState.PAUSED:
            self._needs_start = False
            return
        if self.active_mission is not None:
            if self.navigation.status == "active":
                self.navigation.cancel(reason)
            self._emit(
                self.active_mission,
                "paused",
                reason,
                navigation=self.navigation.snapshot(),
            )
        self.auto_state = AutoState.PAUSED
        self._needs_start = False

    def _cancel_all(self, reason: str) -> None:
        missions = (
            (() if self.active_mission is None else (self.active_mission,))
            + tuple(self._pending)
        )
        if self.navigation.status in {"active", "blocked"}:
            self.navigation.cancel(reason)
        for mission in missions:
            self._emit(mission, "cancelled", reason)
        self.active_mission = None
        self._pending.clear()
        self.auto_state = AutoState.IDLE
        self._needs_start = False

    def _emit(
        self,
        mission: GotoMission,
        status: str,
        reason: str | None = None,
        detail: str | None = None,
        navigation: dict[str, object] | None = None,
    ) -> None:
        self._events.append(
            ControllerEvent(
                self.latest_event_seq + 1,
                self.event_epoch,
                mission,
                status,
                reason,
                detail,
                navigation,
            )
        )
