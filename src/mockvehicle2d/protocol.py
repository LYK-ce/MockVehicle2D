"""Canonical Robot Controller WebSocket command protocol."""

from __future__ import annotations

import json
import math

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    Command,
    CommandResult,
    GotoMission,
    ManualAction,
    ManualCommand,
    ModeAction,
    ModeCommand,
)


MAX_SEQUENCE = 2**64 - 1
MAX_JSON_INTEGER_DIGITS = 4300
MAX_MESSAGE_BYTES = 64 * 1024
MAX_ABS_COORDINATE_M = 1_000_000.0


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, seq: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.seq = seq


def parse_command(
    raw: object,
    *,
    linear_limit_mps: float,
    angular_limit_rps: float,
    mission_batch_limit: int,
) -> Command:
    message = _decode_message(raw)
    seq = _require_seq(message)
    message_type = message.get("type")
    if message_type == "mode":
        return _parse_mode(message, seq)
    if message_type == "manual":
        return _parse_manual(
            message,
            seq,
            linear_limit_mps,
            angular_limit_rps,
        )
    if message_type == "auto":
        return _parse_auto(message, seq, mission_batch_limit)
    raise ProtocolError(
        "invalid_type",
        "type must be one of: mode, manual, auto",
        seq,
    )


def command_ack(
    command: Command,
    result: CommandResult,
    *,
    timestamp: float,
    controller: dict[str, object],
) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "command_ack",
        "timestamp_s": timestamp,
        "seq": command.seq,
        "command": {
            "type": _command_type(command),
            "action": command.action.value,
        },
        "accepted": result.accepted,
        "controller": controller,
    }
    if result.reason is not None:
        message["reason"] = result.reason
    return message


def error_message(error: ProtocolError, *, timestamp: float) -> dict[str, object]:
    return {
        "type": "error",
        "timestamp_s": timestamp,
        "seq": error.seq,
        "code": error.code,
        "message": str(error),
    }


def _parse_mode(message: dict[str, object], seq: int) -> ModeCommand:
    _require_fields(message, {"type", "seq", "action"}, seq)
    try:
        action = ModeAction(message["action"])
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid_action", "unsupported mode action", seq) from error
    return ModeCommand(seq, action)


def _parse_manual(
    message: dict[str, object],
    seq: int,
    linear_limit_mps: float,
    angular_limit_rps: float,
) -> ManualCommand:
    try:
        action = ManualAction(message.get("action"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid_action", "unsupported manual action", seq) from error
    if action is ManualAction.STOP:
        _require_fields(message, {"type", "seq", "action"}, seq)
        return ManualCommand(seq, action)

    _require_fields(
        message,
        {"type", "seq", "action", "linear_mps", "angular_rps"},
        seq,
    )
    linear_mps = _finite_number(message["linear_mps"], "linear_mps", seq)
    angular_rps = _finite_number(message["angular_rps"], "angular_rps", seq)
    if abs(linear_mps) > linear_limit_mps:
        raise ProtocolError(
            "drive_out_of_range",
            "linear_mps exceeds the configured limit",
            seq,
        )
    if abs(angular_rps) > angular_limit_rps:
        raise ProtocolError(
            "drive_out_of_range",
            "angular_rps exceeds the configured limit",
            seq,
        )
    return ManualCommand(seq, action, linear_mps, angular_rps)


def _parse_auto(
    message: dict[str, object],
    seq: int,
    mission_batch_limit: int,
) -> AutoCommand:
    try:
        action = AutoAction(message.get("action"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid_action", "unsupported auto action", seq) from error
    if action is not AutoAction.PUSH:
        _require_fields(message, {"type", "seq", "action"}, seq)
        return AutoCommand(seq, action)

    _require_fields(message, {"type", "seq", "action", "missions"}, seq)
    payload = message["missions"]
    if not isinstance(payload, list) or not payload:
        raise ProtocolError(
            "invalid_missions",
            "missions must be a non-empty JSON array",
            seq,
        )
    if len(payload) > mission_batch_limit:
        raise ProtocolError(
            "mission_batch_too_large",
            "mission batch exceeds the configured queue capacity",
            seq,
        )
    missions = tuple(_parse_mission(item, seq) for item in payload)
    if len({mission.mission_id for mission in missions}) != len(missions):
        raise ProtocolError(
            "duplicate_mission_id",
            "mission_id must be unique within a push batch",
            seq,
        )
    return AutoCommand(seq, action, missions)


def _parse_mission(payload: object, seq: int) -> GotoMission:
    if not isinstance(payload, dict):
        raise ProtocolError("invalid_mission", "each mission must be an object", seq)
    fields = {"mission_id", "type", "frame_id", "x_m", "y_m"}
    _require_fields(payload, fields, seq, subject="mission")
    if payload["type"] != "goto":
        raise ProtocolError("invalid_mission_type", "mission type must be goto", seq)
    mission_id = payload["mission_id"]
    frame_id = payload["frame_id"]
    if not isinstance(mission_id, str) or not isinstance(frame_id, str):
        raise ProtocolError(
            "invalid_mission",
            "mission_id and frame_id must be strings",
            seq,
        )
    x_m = _finite_number(payload["x_m"], "x_m", seq)
    y_m = _finite_number(payload["y_m"], "y_m", seq)
    if abs(x_m) > MAX_ABS_COORDINATE_M or abs(y_m) > MAX_ABS_COORDINATE_M:
        raise ProtocolError(
            "goal_out_of_range",
            f"goal coordinates must be within ±{MAX_ABS_COORDINATE_M:g} m",
            seq,
        )
    try:
        return GotoMission(mission_id, frame_id, x_m, y_m, seq)
    except ValueError as error:
        raise ProtocolError("invalid_mission", str(error), seq) from error


def _decode_message(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise ProtocolError(
            "invalid_json_text",
            "commands must be JSON text messages",
        )
    if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message_too_large", "command exceeds 64 KiB")
    try:
        message = json.loads(
            raw,
            parse_int=_bounded_json_int,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (ValueError, RecursionError) as error:
        raise ProtocolError("invalid_json", "command is not valid JSON") from error
    if not isinstance(message, dict):
        raise ProtocolError("invalid_message", "command JSON must be an object")
    return message


def _require_seq(message: dict[str, object]) -> int:
    if "seq" not in message:
        raise ProtocolError("missing_seq", "command requires seq")
    seq = message["seq"]
    if (
        isinstance(seq, bool)
        or not isinstance(seq, int)
        or not 0 <= seq <= MAX_SEQUENCE
    ):
        raise ProtocolError(
            "invalid_seq",
            "seq must be an unsigned 64-bit integer",
        )
    return seq


def _require_fields(
    message: dict[str, object],
    expected: set[str],
    seq: int,
    *,
    subject: str = "command",
) -> None:
    if set(message) != expected:
        raise ProtocolError(
            "invalid_fields",
            f"{subject} has missing or unexpected fields",
            seq,
        )


def _finite_number(value: object, field: str, seq: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_number", f"{field} must be a JSON number", seq)
    try:
        number = float(value)
    except OverflowError as error:
        raise ProtocolError("invalid_number", f"{field} must be finite", seq) from error
    if not math.isfinite(number):
        raise ProtocolError("invalid_number", f"{field} must be finite", seq)
    return number


def _bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer is too long")
    return int(value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _command_type(command: Command) -> str:
    if isinstance(command, ModeCommand):
        return "mode"
    if isinstance(command, ManualCommand):
        return "manual"
    return "auto"
