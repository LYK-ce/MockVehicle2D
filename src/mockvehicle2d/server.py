#!/usr/bin/env python3
"""Controllable 2D vehicle and Tmini-style WebSocket simulator."""

import asyncio
import json
import math
import random
import re
import signal
import struct
import time

from mockvehicle2d.instruction.compiler import TaskCompiler
from mockvehicle2d.instruction.llm_client import FakeModelClient
from mockvehicle2d.instruction.state_machine import InstructionState, InstructionStateMachine
from mockvehicle2d.instruction.validator import SchemaValidator, SemanticValidator
from mockvehicle2d.map_grid import MapGrid, VOID
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.pathfinding import PathFollowingController
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.scan import TMINI_SCAN_CONFIG, scan_message
from mockvehicle2d.vehicle import COMMANDS, Vehicle


HOST = "0.0.0.0"
PORT = 19090
DEFAULT_VEHICLE_ID = "mock_vehicle_01"
SPAWN_X = 10.0
SPAWN_Y = 10.0
MAX_JSON_INTEGER_DIGITS = 4300
VEHICLE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


class CommandMessageError(ValueError):
    def __init__(self, code: str, message: str, seq: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.seq = seq


def validate_vehicle_id(value: str) -> str:
    if not VEHICLE_ID_PATTERN.fullmatch(value):
        raise ValueError("vehicle id must be 1-64 ASCII letters, digits, dots, underscores, or hyphens")
    return value


def _next_deadline(deadline: float, now: float, period: float) -> float:
    if now < deadline:
        return deadline
    return deadline + (math.floor((now - deadline) / period) + 1) * period


def _safe_seq(message: object) -> int | None:
    if not isinstance(message, dict):
        return None
    seq = message.get("seq")
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0 else None


def _bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer is too long")
    return int(value)


def _decode_message(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise CommandMessageError("invalid_json_text", "command must be a JSON text message")
    try:
        message = json.loads(raw, parse_int=_bounded_json_int)
    except (ValueError, RecursionError) as error:
        raise CommandMessageError("invalid_json", "command is not valid JSON text") from error
    if not isinstance(message, dict):
        raise CommandMessageError("invalid_message", "command JSON must be an object")
    return message


def _parse_command_object(message: dict[str, object]) -> tuple[str, int | None]:
    seq = _safe_seq(message)
    if set(message) == {"cmd"}:
        command = message["cmd"]
        if not isinstance(command, str) or command not in COMMANDS:
            raise CommandMessageError("invalid_cmd", "unsupported cmd", None)
        return command, None

    if "type" not in message:
        raise CommandMessageError("missing_type", "canonical command requires type", seq)
    if message["type"] != "cmd":
        raise CommandMessageError("invalid_type", "type must be cmd", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "canonical command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if "cmd" not in message or not isinstance(message["cmd"], str) or message["cmd"] not in COMMANDS:
        raise CommandMessageError("invalid_cmd", "unsupported cmd", seq)
    if set(message) != {"type", "seq", "cmd"}:
        raise CommandMessageError("invalid_fields", "canonical command has unexpected fields", seq)
    return message["cmd"], seq


def parse_command_message(raw: object) -> tuple[str, int | None]:
    """Validate canonical commands and the exact legacy ``{"cmd": ...}`` form."""
    return _parse_command_object(_decode_message(raw))


def _parse_drive_object(
    message: dict[str, object], linear_limit: float, angular_limit: float
) -> tuple[float, float, int]:
    seq = _safe_seq(message)
    if message.get("type") != "drive":
        raise CommandMessageError("invalid_type", "type must be drive", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "drive command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if set(message) != {"type", "seq", "linear_mps", "angular_rps"}:
        raise CommandMessageError("invalid_fields", "drive command has missing or unexpected fields", seq)

    linear_mps = message["linear_mps"]
    angular_rps = message["angular_rps"]
    values = (linear_mps, angular_rps)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise CommandMessageError("invalid_drive", "drive velocities must be JSON numbers", seq)
    if any(isinstance(value, float) and not math.isfinite(value) for value in values):
        raise CommandMessageError("invalid_drive", "drive velocities must be finite", seq)
    if abs(linear_mps) > linear_limit or abs(angular_rps) > angular_limit:
        raise CommandMessageError("drive_out_of_range", "drive velocities exceed configured limits", seq)
    return float(linear_mps), float(angular_rps), seq


def parse_drive_message(
    raw: object, linear_limit: float, angular_limit: float
) -> tuple[float, float, int]:
    """Validate one bounded continuous-velocity command."""
    return _parse_drive_object(_decode_message(raw), linear_limit, angular_limit)


def _parse_goto_object(message: dict[str, object]) -> tuple[float, float, int]:
    seq = _safe_seq(message)
    if message.get("type") != "goto":
        raise CommandMessageError("invalid_type", "type must be goto", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "goto command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if set(message) != {"type", "seq", "x_m", "y_m"}:
        raise CommandMessageError("invalid_fields", "goto command has missing or unexpected fields", seq)

    values = (message["x_m"], message["y_m"])
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise CommandMessageError("invalid_goto", "goto coordinates must be JSON numbers", seq)
    try:
        x_m, y_m = (float(value) for value in values)
    except OverflowError as error:
        raise CommandMessageError("invalid_goto", "goto coordinates must be finite", seq) from error
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise CommandMessageError("invalid_goto", "goto coordinates must be finite", seq)
    return x_m, y_m, seq


def parse_goto_message(raw: object) -> tuple[float, float, int]:
    """Validate one local-odometry go-to-goal command."""
    return _parse_goto_object(_decode_message(raw))


def handle_command_message(
    raw: object,
    vehicle: Vehicle,
    grid: MapGrid,
    monotonic_now: float,
    wall_timestamp: float,
    navigation: GotoController | None = None,
    safety: LocalSafetyRuntime | None = None,
    path_following: PathFollowingController | None = None,
) -> dict[str, object]:
    """Advance the prior command, then acknowledge or fail-safe stop."""
    if safety is None:
        handoff_collided = vehicle.advance(grid, monotonic_now)
        handoff_safety_stop = None
    else:
        handoff = safety.advance(
            vehicle,
            grid,
            monotonic_now,
            automatic=(
                (navigation is not None and navigation.status == "active")
                or (path_following is not None and path_following.status == "active")
            ),
        )
        handoff_collided = handoff.collided
        handoff_safety_stop = handoff.reason if handoff.stopped else None
    rejection_reason: str | None = None
    try:
        message = _decode_message(raw)
        if message.get("type") == "goto":
            x_m, y_m, seq = _parse_goto_object(message)
            if navigation is None:
                raise CommandMessageError("goto_unavailable", "goto controller is unavailable", seq)
            vehicle.stop()
            if path_following is not None:
                path_following.cancel("manual_override")
            navigation.start(x_m, y_m)
            if handoff_collided:
                navigation.status = "blocked"
                navigation.reason = "collision"
            elif handoff_safety_stop is not None:
                navigation.status = "blocked"
                navigation.reason = handoff_safety_stop
            accepted = navigation.status == "active"
            reply = {
                "type": "goto_ack",
                "ts": wall_timestamp,
                "seq": seq,
                "goal": {"x_m": x_m, "y_m": y_m},
                "accepted": accepted,
            }
            if not accepted:
                reply["reason"] = navigation.reason
            return reply
        if message.get("type") == "drive":
            linear_mps, angular_rps, seq = _parse_drive_object(
                message, vehicle.linear_speed, vehicle.angular_speed
            )
            command = "drive"
        else:
            command, seq = _parse_command_object(message)
            linear_mps, angular_rps = vehicle.velocities_for_command(command)
        if navigation is not None:
            navigation.cancel("manual_override")
        if path_following is not None:
            path_following.cancel("manual_override")
        decision = (
            safety.enforce_manual(vehicle, grid, (linear_mps, angular_rps))
            if safety is not None
            else None
        )
        if handoff_collided:
            rejection_reason = "collision"
        elif decision is not None and decision.state in {"stopped", "fault"}:
            rejection_reason = decision.reason or "safety_rejected"
        elif command == "drive":
            vehicle.install_drive(linear_mps, angular_rps, monotonic_now)
        else:
            vehicle.install_command(command, monotonic_now)
    except CommandMessageError as error:
        vehicle.stop()
        if navigation is not None:
            navigation.cancel("invalid_command")
        if path_following is not None:
            path_following.cancel("invalid_command")
        return {
            "type": "error",
            "ts": wall_timestamp,
            "seq": error.seq,
            "code": error.code,
            "message": str(error),
        }

    reply = {
        "type": "cmd_ack",
        "ts": wall_timestamp,
        "seq": seq,
        "cmd": command,
        "accepted": rejection_reason is None,
    }
    if rejection_reason is not None:
        reply["reason"] = rejection_reason
    return reply


def _handle_nl_command(
    message: dict[str, object],
    vehicle: Vehicle,
    grid: MapGrid,
    navigation: GotoController,
    wall_timestamp: float,
    monotonic_now: float,
    nl_client: FakeModelClient,
    schema_v: SchemaValidator,
    semantic_v: SemanticValidator,
    state_machine: InstructionStateMachine,
    task_compiler: TaskCompiler,
    scan_data: dict[str, object] | None = None,
    path_following: PathFollowingController | None = None,
) -> list[dict[str, object]]:
    """Process one nl_command message. Returns a list of reply dicts to send."""
    seq = message.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        seq = 0
    text = message.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return [{
            "type": "nl_parse_result",
            "ts": wall_timestamp,
            "seq": seq,
            "instruction": None,
            "accepted": False,
            "reason": "empty command text",
        }]

    # Reset state machine if in terminal state or CONFIRMING
    current = state_machine.current_state
    if current == InstructionState.CONFIRMING:
        # Cancel the pending confirmation and restart
        state_machine.transition(InstructionState.CANCELLED)
        state_machine.transition(InstructionState.IDLE)
    elif current not in (InstructionState.IDLE, InstructionState.REJECTED, InstructionState.COMPLETED,
                         InstructionState.BLOCKED, InstructionState.CANCELLED, InstructionState.FAILED):
        return [{
            "type": "nl_parse_result",
            "ts": wall_timestamp,
            "seq": seq,
            "instruction": None,
            "accepted": False,
            "reason": f"busy: state machine is {current.name.lower()}",
        }]

    replies: list[dict[str, object]] = []

    # 1. Parse
    state_machine.transition(InstructionState.PARSING)
    instruction = nl_client.parse(text)
    if instruction is None:
        state_machine.transition(InstructionState.FAILED)
        replies.append({
            "type": "nl_parse_result",
            "ts": wall_timestamp,
            "seq": seq,
            "instruction": None,
            "accepted": False,
            "reason": "parse failed: no result",
        })
        state_machine.transition(InstructionState.IDLE)
        return replies

    # 2. Validate (always go through VALIDATING, even for clarify)
    state_machine.transition(InstructionState.VALIDATING)
    schema_ok, schema_err = schema_v.validate(instruction)
    if not schema_ok:
        state_machine.transition(InstructionState.REJECTED)
        replies.append({
            "type": "nl_parse_result",
            "ts": wall_timestamp,
            "seq": seq,
            "instruction": instruction,
            "accepted": False,
            "reason": f"schema validation failed: {schema_err}",
        })
        state_machine.transition(InstructionState.IDLE)
        return replies

    intent = instruction.get("intent")

    # Clarify: ask user for more info
    if intent == "clarify":
        state_machine.transition(InstructionState.CONFIRMING)
        params = instruction.get("parameters", {}) or {}
        replies.append({
            "type": "nl_confirm_request",
            "ts": wall_timestamp,
            "seq": seq,
            "question": params.get("question", "请提供更多信息"),
            "missing": params.get("missing_parameters", []),
        })
        return replies

    semantic_ok, semantic_err = semantic_v.validate(instruction)
    if not semantic_ok:
        state_machine.transition(InstructionState.REJECTED)
        replies.append({
            "type": "nl_parse_result",
            "ts": wall_timestamp,
            "seq": seq,
            "instruction": instruction,
            "accepted": False,
            "reason": f"semantic validation failed: {semantic_err}",
        })
        state_machine.transition(InstructionState.IDLE)
        return replies

    # 3. Accept + Compile + Execute
    state_machine.transition(InstructionState.ACCEPTED)
    replies.append({
        "type": "nl_parse_result",
        "ts": wall_timestamp,
        "seq": seq,
        "instruction": instruction,
        "accepted": True,
        "reason": "instruction accepted",
    })

    params = instruction.get("parameters", {}) or {}

    if intent == "stop":
        vehicle.stop()
        navigation.cancel("nl_stop")
        if path_following is not None:
            path_following.cancel("nl_stop")
        state_machine.transition(InstructionState.ACTIVE)
        state_machine.transition(InstructionState.COMPLETED)
        replies.append({
            "type": "nl_task_update",
            "ts": wall_timestamp,
            "seq": seq,
            "status": "completed",
            "reason": "vehicle stopped",
        })
        state_machine.transition(InstructionState.IDLE)
        return replies

    if intent == "status":
        state_machine.transition(InstructionState.ACTIVE)
        nav_snap = navigation.snapshot()
        replies.append({
            "type": "nl_task_update",
            "ts": wall_timestamp,
            "seq": seq,
            "status": "completed",
            "reason": f"position: ({vehicle.x:.2f}, {vehicle.y:.2f}), nav: {nav_snap.get('status')}",
        })
        state_machine.transition(InstructionState.COMPLETED)
        state_machine.transition(InstructionState.IDLE)
        return replies

    if intent == "scan_report":
        state_machine.transition(InstructionState.ACTIVE)
        summary = _summarize_scan_for_nl(scan_data) if scan_data else {}
        replies.append({
            "type": "nl_scan_report",
            "ts": wall_timestamp,
            "seq": seq,
            "summary": summary.get("text", "扫描完成"),
            "points_summary": summary.get("sectors", {}),
        })
        state_machine.transition(InstructionState.COMPLETED)
        state_machine.transition(InstructionState.IDLE)
        return replies

    if intent == "goto_point":
        x_m = params["x_m"]
        y_m = params["y_m"]
        # Phase 3: use task compiler to decide controller
        pose_snap = {"pose": {"x": vehicle.x, "y": vehicle.y, "yaw": vehicle.yaw}}
        task = task_compiler.compile(instruction, pose_snap)
        controller_choice = task.get("controller", "GotoController")

        if controller_choice == "blocked":
            # Go through ACTIVE first (state machine: ACCEPTED→ACTIVE→BLOCKED)
            state_machine.transition(InstructionState.ACTIVE)
            state_machine.transition(InstructionState.BLOCKED)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "blocked",
                "reason": task.get("reason", "no path found"),
            })
            state_machine.transition(InstructionState.IDLE)
            return replies

        vehicle.stop()
        if controller_choice == "PathFollowingController" and path_following is not None:
            path = task.get("path")
            if not path or len(path) < 2:
                state_machine.transition(InstructionState.ACTIVE)
                state_machine.transition(InstructionState.BLOCKED)
                replies.append({
                    "type": "nl_task_update",
                    "ts": wall_timestamp,
                    "seq": seq,
                    "status": "blocked",
                    "reason": "invalid path",
                })
                state_machine.transition(InstructionState.IDLE)
                return replies
            navigation.cancel("path_following_active")
            path_following.start(path)
            if path_following.status != "active":
                state_machine.transition(InstructionState.ACTIVE)
                state_machine.transition(InstructionState.BLOCKED)
                replies.append({
                    "type": "nl_task_update",
                    "ts": wall_timestamp,
                    "seq": seq,
                    "status": "blocked",
                    "reason": path_following.reason or "path rejected",
                })
                state_machine.transition(InstructionState.IDLE)
                return replies
            state_machine.transition(InstructionState.ACTIVE)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "active",
                "reason": f"path-following to ({x_m}, {y_m}) via {len(path)} waypoints",
            })
            return replies

        # Default: GotoController
        if path_following is not None:
            path_following.cancel("goto_active")
        navigation.start(x_m, y_m)
        if navigation.status != "active":
            state_machine.transition(InstructionState.ACTIVE)
            state_machine.transition(InstructionState.BLOCKED)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "blocked",
                "reason": navigation.reason or "goal rejected",
            })
            state_machine.transition(InstructionState.IDLE)
            return replies
        state_machine.transition(InstructionState.ACTIVE)
        replies.append({
            "type": "nl_task_update",
            "ts": wall_timestamp,
            "seq": seq,
            "status": "active",
            "reason": f"navigating to ({x_m}, {y_m})",
        })
        return replies

    if intent == "move_distance":
        distance_m = params["distance_m"]
        direction = params["direction"]
        sign = 1.0 if direction == "forward" else -1.0
        goal_x = vehicle.x + sign * distance_m * math.cos(vehicle.yaw)
        goal_y = vehicle.y + sign * distance_m * math.sin(vehicle.yaw)
        # Phase 3: use task compiler to decide controller
        pose_snap = {"pose": {"x": vehicle.x, "y": vehicle.y, "yaw": vehicle.yaw}}
        task = task_compiler.compile(instruction, pose_snap)
        controller_choice = task.get("controller", "GotoController")

        if controller_choice == "blocked":
            state_machine.transition(InstructionState.ACTIVE)
            state_machine.transition(InstructionState.BLOCKED)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "blocked",
                "reason": task.get("reason", "no path found"),
            })
            state_machine.transition(InstructionState.IDLE)
            return replies

        vehicle.stop()
        if controller_choice == "PathFollowingController" and path_following is not None:
            path = task.get("path")
            if not path or len(path) < 2:
                state_machine.transition(InstructionState.ACTIVE)
                state_machine.transition(InstructionState.BLOCKED)
                replies.append({
                    "type": "nl_task_update",
                    "ts": wall_timestamp,
                    "seq": seq,
                    "status": "blocked",
                    "reason": "invalid path",
                })
                state_machine.transition(InstructionState.IDLE)
                return replies
            navigation.cancel("path_following_active")
            path_following.start(path)
            if path_following.status != "active":
                state_machine.transition(InstructionState.ACTIVE)
                state_machine.transition(InstructionState.BLOCKED)
                replies.append({
                    "type": "nl_task_update",
                    "ts": wall_timestamp,
                    "seq": seq,
                    "status": "blocked",
                    "reason": path_following.reason or "move_distance path rejected",
                })
                state_machine.transition(InstructionState.IDLE)
                return replies
            state_machine.transition(InstructionState.ACTIVE)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "active",
                "reason": f"path-following {direction} {distance_m}m via {len(path)} waypoints",
            })
            return replies

        # Default: GotoController
        if path_following is not None:
            path_following.cancel("goto_active")
        navigation.start(goal_x, goal_y)
        if navigation.status != "active":
            state_machine.transition(InstructionState.ACTIVE)
            state_machine.transition(InstructionState.BLOCKED)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "blocked",
                "reason": navigation.reason or "move_distance rejected",
            })
            state_machine.transition(InstructionState.IDLE)
            return replies
        state_machine.transition(InstructionState.ACTIVE)
        replies.append({
            "type": "nl_task_update",
            "ts": wall_timestamp,
            "seq": seq,
            "status": "active",
            "reason": f"moving {direction} {distance_m}m",
        })
        return replies

    if intent == "rotate":
        angle_deg = params["angle_deg"]
        direction = params["direction"]
        sign = 1.0 if direction == "left" else -1.0
        target_yaw = vehicle.yaw + sign * math.radians(angle_deg)
        # Set a virtual goal 0.1m ahead in the target direction
        goal_x = vehicle.x + 0.1 * math.cos(target_yaw)
        goal_y = vehicle.y + 0.1 * math.sin(target_yaw)
        vehicle.stop()
        if path_following is not None:
            path_following.cancel("manual_override")
        navigation.start(goal_x, goal_y)
        if navigation.status != "active":
            state_machine.transition(InstructionState.BLOCKED)
            replies.append({
                "type": "nl_task_update",
                "ts": wall_timestamp,
                "seq": seq,
                "status": "blocked",
                "reason": navigation.reason or "rotate rejected",
            })
            state_machine.transition(InstructionState.IDLE)
            return replies
        state_machine.transition(InstructionState.ACTIVE)
        replies.append({
            "type": "nl_task_update",
            "ts": wall_timestamp,
            "seq": seq,
            "status": "active",
            "reason": f"rotating {direction} {angle_deg}°",
        })
        return replies

    # Unknown intent
    state_machine.transition(InstructionState.FAILED)
    replies.append({
        "type": "nl_task_update",
        "ts": wall_timestamp,
        "seq": seq,
        "status": "failed",
        "reason": f"unsupported intent: {intent}",
    })
    state_machine.transition(InstructionState.IDLE)
    return replies


def _summarize_scan_for_nl(scan_data: dict[str, object] | None) -> dict[str, object]:
    """Build a human-readable NL scan summary from raw scan frame."""
    if scan_data is None:
        return {"text": "无扫描数据", "sectors": {}}
    points = scan_data.get("points", [])
    if not isinstance(points, list) or not points:
        return {"text": "无扫描点", "sectors": {}}

    sectors: dict[str, list[float]] = {"front": [], "left": [], "right": [], "back": []}
    for pt in points:
        if not isinstance(pt, dict):
            continue
        angle = pt.get("angle", 0.0)
        rng = pt.get("range", 0.0)
        if not isinstance(rng, (int, float)) or rng <= 0:
            continue
        if -math.pi / 4 <= angle < math.pi / 4:
            sectors["front"].append(rng)
        elif math.pi / 4 <= angle < 3 * math.pi / 4:
            sectors["left"].append(rng)
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:
            sectors["right"].append(rng)
        else:
            sectors["back"].append(rng)

    summary: dict[str, float] = {}
    for sector, ranges in sectors.items():
        if ranges:
            summary[sector] = round(min(ranges), 2)

    # Build human-readable text
    parts = []
    for sector in ("front", "left", "right", "back"):
        label = {"front": "前方", "left": "左侧", "right": "右侧", "back": "后方"}[sector]
        if sector in summary:
            parts.append(f"{label} {summary[sector]:.1f}m")
        else:
            parts.append(f"{label} 无数据")
    text = "障碍物距离 — " + "，".join(parts)

    return {"text": text, "sectors": summary}


def _cancel_nl_task(
    navigation: GotoController,
    state_machine: InstructionStateMachine,
    reason: str,
    path_following: PathFollowingController | None = None,
) -> dict[str, object] | None:
    """Cancel any active NL task and return the nl_task_update to send, or None."""
    current = state_machine.current_state
    if current in (InstructionState.ACCEPTED, InstructionState.ACTIVE, InstructionState.CONFIRMING):
        navigation.cancel(reason)
        if path_following is not None:
            path_following.cancel(reason)
        state_machine.transition(InstructionState.CANCELLED)
        update: dict[str, object] = {
            "type": "nl_task_update",
            "seq": 0,
            "status": "cancelled",
            "reason": reason,
        }
        # Allow going back to IDLE after cancelling
        if state_machine.current_state == InstructionState.CANCELLED:
            pass  # Will be reset by next nl_command
        return update
    return None


def _encode_map_chunks(voxels: list[dict[str, object]], map_size: int) -> list[bytes]:
    """Encode voxels into Pictor-compatible binary chunk frames.

    Each chunk is a 256×256 cell sub-grid, encoded as:
        [u8 type=0][i32 chunk_x BE][i32 chunk_y BE][65536 bytes cells]
    Cells use state values directly: 0=free, 1=wall, 2=void.
    """
    CHUNK_SIZE = 256
    cells_per_chunk = CHUNK_SIZE * CHUNK_SIZE
    # Build flat (gx, gy) → state lookup
    state = {}
    for v in voxels:
        state[(v["gx"], v["gy"])] = v.get("state", 0)
    chunks = []
    for chunk_y in range(0, map_size, CHUNK_SIZE):
        for chunk_x in range(0, map_size, CHUNK_SIZE):
            cell_bytes = bytearray(cells_per_chunk)
            for gy in range(CHUNK_SIZE):
                ay = chunk_y + gy
                row_offset = gy * CHUNK_SIZE
                for gx in range(CHUNK_SIZE):
                    ax = chunk_x + gx
                    cell_bytes[row_offset + gx] = state.get((ax, ay), 0)
            header = struct.pack(">Bii", 0, chunk_x, chunk_y)
            chunks.append(header + bytes(cell_bytes))
    return chunks


async def _send_map_chunks(websocket, chunks: list[bytes], wall_time: float) -> None:
    """Send map_full as binary chunk frames to Pictor."""
    total_bytes = sum(len(c) for c in chunks)
    print(f"[→] sending map_full ({total_bytes} bytes, {len(chunks)} chunk(s))")
    for chunk in chunks:
        await websocket.send(chunk)


def generate_map(size: int = 256, seed: int = 42, radius: float = 0.5) -> tuple[list[dict[str, object]], MapGrid]:
    """Create the deterministic ground-truth grid and clear the spawn area."""
    rng = random.Random(seed)
    clear_min_x = math.floor(SPAWN_X - radius) - 1
    clear_max_x = math.ceil(SPAWN_X + radius) + 1
    clear_min_y = math.floor(SPAWN_Y - radius) - 1
    clear_max_y = math.ceil(SPAWN_Y + radius) + 1
    voxels = []
    for gx in range(size):
        for gy in range(size):
            in_spawn = clear_min_x <= gx <= clear_max_x and clear_min_y <= gy <= clear_max_y
            is_wall = rng.random() < 0.05
            is_void = size >= 32 and 24 <= gx <= 26 and 9 <= gy <= 12
            state = VOID if is_void and not in_spawn else int(is_wall and not in_spawn)
            voxels.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "gz": 0,
                    "state": state,
                    "conf": 1.0,
                }
            )
    return voxels, MapGrid.from_voxels(voxels)


def telemetry_messages(
    vehicle: Vehicle,
    grid: MapGrid,
    sequence: int,
    timestamp: float,
    navigation: GotoController | None = None,
    path_following: PathFollowingController | None = None,
    safety: LocalSafetyRuntime | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build a pose/scan pair from one state snapshot and wall-clock timestamp."""
    vx, vy, omega = vehicle.velocities()
    # Determine control_mode: PathFollowingController wins if both are active
    if path_following is not None and path_following.status == "active":
        control_mode = path_following.control_mode
        nav_snapshot = path_following.snapshot()
    elif navigation is not None:
        control_mode = navigation.control_mode
        nav_snapshot = navigation.snapshot()
    else:
        control_mode = "manual"
        nav_snapshot = {"status": "idle", "goal": None, "reason": None}
    pose = {
        "type": "pose",
        "ts": timestamp,
        "seq": sequence,
        "source": "simulator_ground_truth",
        "x": vehicle.x,
        "y": vehicle.y,
        "z": 0.0,
        "yaw": vehicle.yaw,
        "vx": vx,
        "vy": vy,
        "omega": omega,
        "collision": vehicle.collision,
        "command": vehicle.command,
        "control_mode": control_mode,
        "navigation": nav_snapshot,
        "safety": (
            safety.snapshot()
            if safety is not None
            else {
                "state": "clear",
                "reason": None,
                "obstacle_clearance_m": None,
                "edge_clearance_m": None,
            }
        ),
    }
    scan = scan_message(grid, vehicle.x, vehicle.y, vehicle.yaw, timestamp, TMINI_SCAN_CONFIG)
    scan["seq"] = sequence
    return pose, scan


async def handler(
    websocket,
    *,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    _monotonic=time.monotonic,
    _wall_time=time.time,
    _safety_healthy: bool = True,
) -> None:
    """Serve one client; all receives and sends stay serialized in this coroutine."""
    vehicle_id = validate_vehicle_id(vehicle_id)
    addr = websocket.remote_address
    print(f"[+] client connected: {addr}")
    started_at = _monotonic()
    voxels, grid = generate_map(radius=radius)
    vehicle = Vehicle(
        SPAWN_X,
        SPAWN_Y,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        radius=radius,
        command_timeout=command_timeout,
        now=started_at,
    )
    navigation = GotoController()
    path_following = PathFollowingController()
    safety = LocalSafetyRuntime(healthy=_safety_healthy)

    # NL instruction pipeline
    nl_client = FakeModelClient()
    schema_v = SchemaValidator()
    semantic_v = SemanticValidator(grid)
    state_machine = InstructionStateMachine()
    task_compiler = TaskCompiler(grid)

    frame_sequence = 0
    next_deadline = started_at
    active_nl_seq: int | None = None
    last_scan_data: dict[str, object] | None = None

    try:
        await websocket.send(json.dumps({"type": "hello", "vehicle_id": vehicle_id}))
        map_chunks = _encode_map_chunks(voxels, 256)
        await _send_map_chunks(websocket, map_chunks, _wall_time())

        while True:
            now = _monotonic()
            if now >= next_deadline:
                # Phase 3: update path_following if active, else navigation
                if path_following.status == "active":
                    path_following.update(vehicle, grid, now, safety)
                elif navigation.status == "active":
                    navigation.update(vehicle, grid, now, safety)
                else:
                    navigation.update(vehicle, grid, now, safety)
                timestamp = _wall_time()
                pose, scan = telemetry_messages(
                    vehicle, grid, frame_sequence, timestamp, navigation, path_following, safety
                )
                await websocket.send(json.dumps(pose))
                await websocket.send(json.dumps(scan))
                last_scan_data = scan
                print(f"[→] pose #{frame_sequence}: x={vehicle.x:.2f} y={vehicle.y:.2f} cmd={vehicle.command}")

                # Check NL task status — determine active controller
                active_controller = (
                    path_following if path_following.status == "active"
                    else navigation if navigation.status == "active"
                    else None
                )
                # Also handle completed/blocked from non-active controllers
                if state_machine.current_state == InstructionState.ACTIVE:
                    # Check path_following first
                    pf_status = path_following.status
                    nav_status = navigation.status
                    if pf_status == "reached" or nav_status == "reached":
                        state_machine.transition(InstructionState.COMPLETED)
                        await websocket.send(json.dumps({
                            "type": "nl_task_update",
                            "ts": timestamp,
                            "seq": active_nl_seq if active_nl_seq is not None else 0,
                            "status": "completed",
                            "reason": "goal_reached",
                        }))
                        state_machine.transition(InstructionState.IDLE)
                        active_nl_seq = None
                        print(f"[NL] task completed: goal reached")
                    elif pf_status == "blocked" or nav_status == "blocked":
                        reason = path_following.reason if pf_status == "blocked" else navigation.reason
                        state_machine.transition(InstructionState.BLOCKED)
                        await websocket.send(json.dumps({
                            "type": "nl_task_update",
                            "ts": timestamp,
                            "seq": active_nl_seq if active_nl_seq is not None else 0,
                            "status": "blocked",
                            "reason": reason or "unknown",
                        }))
                        state_machine.transition(InstructionState.IDLE)
                        active_nl_seq = None
                        print(f"[NL] task blocked: {reason}")
                    elif pf_status == "cancelled" or nav_status == "cancelled":
                        reason = path_following.reason if pf_status == "cancelled" else navigation.reason
                        state_machine.transition(InstructionState.CANCELLED)
                        await websocket.send(json.dumps({
                            "type": "nl_task_update",
                            "ts": timestamp,
                            "seq": active_nl_seq if active_nl_seq is not None else 0,
                            "status": "cancelled",
                            "reason": reason or "unknown",
                        }))
                        state_machine.transition(InstructionState.IDLE)
                        active_nl_seq = None
                        print(f"[NL] task cancelled: {reason}")

                frame_sequence += 1
                next_deadline = _next_deadline(next_deadline, _monotonic(), TMINI_SCAN_CONFIG.scan_time)
                continue

            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=next_deadline - now)
            except asyncio.TimeoutError:
                continue

            # Check if this is an NL message
            try:
                message = _decode_message(raw)
                if message.get("type") == "nl_command":
                    replies = _handle_nl_command(
                        message, vehicle, grid, navigation,
                        _wall_time(), _monotonic(),
                        nl_client, schema_v, semantic_v,
                        state_machine, task_compiler,
                        scan_data=last_scan_data,
                        path_following=path_following,
                    )
                    for r in replies:
                        await websocket.send(json.dumps(r))
                    # Track active seq
                    if state_machine.current_state == InstructionState.ACTIVE:
                        active_nl_seq = message.get("seq")
                    continue
                if message.get("type") == "nl_clarify_response":
                    # Treat as a follow-up nl_command with the response text
                    text = message.get("text", "")
                    if isinstance(text, str) and text.strip():
                        synthetic = {"type": "nl_command", "seq": message.get("seq", 0), "text": text}
                        replies = _handle_nl_command(
                            synthetic, vehicle, grid, navigation,
                            _wall_time(), _monotonic(),
                            nl_client, schema_v, semantic_v,
                            state_machine, task_compiler,
                            scan_data=last_scan_data,
                            path_following=path_following,
                        )
                        for r in replies:
                            await websocket.send(json.dumps(r))
                        if state_machine.current_state == InstructionState.ACTIVE:
                            active_nl_seq = message.get("seq")
                    else:
                        await websocket.send(json.dumps({
                            "type": "nl_parse_result",
                            "ts": _wall_time(),
                            "seq": message.get("seq", 0),
                            "instruction": None,
                            "accepted": False,
                            "reason": "empty clarify response",
                        }))
                    continue
                # Otherwise, cancel any active NL task (manual override)
            except CommandMessageError:
                message = None  # Will be handled by handle_command_message

            # Cancel active NL task if manual command arrives
            if state_machine.current_state in (InstructionState.ACCEPTED, InstructionState.ACTIVE, InstructionState.CONFIRMING):
                navigation.cancel("manual_override")
                path_following.cancel("manual_override")
                state_machine.transition(InstructionState.CANCELLED)
                await websocket.send(json.dumps({
                    "type": "nl_task_update",
                    "ts": _wall_time(),
                    "seq": active_nl_seq if active_nl_seq is not None else 0,
                    "status": "cancelled",
                    "reason": "manual_override",
                }))
                active_nl_seq = None
                print(f"[NL] task cancelled by manual override")

            reply = handle_command_message(
                raw, vehicle, grid, _monotonic(), _wall_time(), navigation, safety, path_following=path_following
            )
            await websocket.send(json.dumps(reply))
    except Exception as error:
        print(f"[!] connection ended: {error}")
    finally:
        navigation.cancel("disconnected")
        path_following.cancel("disconnected")
        vehicle.stop()
        print(f"[-] client disconnected: {addr}")


async def main(
    *,
    port: int = PORT,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
) -> None:
    from websockets.asyncio.server import serve

    vehicle_id = validate_vehicle_id(vehicle_id)
    stop = asyncio.Event()
    _shutting_down = False

    def _sig_handler():
        nonlocal _shutting_down
        if not _shutting_down:
            _shutting_down = True
            print("\n[!] shutting down...")
            stop.set()
        else:
            # Second Ctrl+C while already shutting down — force exit.
            print("\n[!] forcing exit...")
            import os as _os

            _os._exit(1)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig_handler)

    async def configured_handler(websocket):
        await handler(
            websocket,
            vehicle_id=vehicle_id,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            radius=radius,
            command_timeout=command_timeout,
        )

    try:
        async with serve(configured_handler, HOST, port):
            print(f"Mock Vehicle Server listening on ws://{HOST}:{port}")
            print("Waiting for a controller connection...\n")
            await stop.wait()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
