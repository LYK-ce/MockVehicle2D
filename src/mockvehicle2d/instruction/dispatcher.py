"""Translate validated intent objects into Robot Controller v4 commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from mockvehicle2d.protocol import MAX_SEQUENCE, parse_command


@dataclass(frozen=True)
class TranslatedInstruction:
    """One intent, its function-call view, and an optional v4 command."""

    function_call: dict[str, Any]
    command: dict[str, Any] | None
    instruction: dict[str, Any]


def translate(
    instruction: dict[str, Any],
    *,
    seq: int,
    mission_id: str | None = None,
) -> TranslatedInstruction:
    """Translate one validated v3-style intent at the client boundary.

    ``goto`` produces only ``auto/push``. The caller must explicitly select
    Auto mode before sending it. ``clarify`` has no executable command.
    """

    if not isinstance(instruction, dict):
        raise ValueError("instruction must be an object")
    if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq <= MAX_SEQUENCE:
        raise ValueError("seq must be an unsigned 64-bit integer")
    parameters = instruction.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")

    intent = instruction.get("intent")
    if intent == "stop":
        function_call = {"name": "stop", "arguments": {}}
        command: dict[str, Any] | None = {
            "type": "mode",
            "seq": seq,
            "action": "stop_motion",
        }
    elif intent == "goto":
        if mission_id is None:
            raise ValueError("goto requires mission_id")
        try:
            x_m = parameters["x_m"]
            y_m = parameters["y_m"]
        except KeyError as error:
            raise ValueError("goto requires x_m and y_m") from error
        function_call = {
            "name": "goto",
            "arguments": {"x_m": x_m, "y_m": y_m},
        }
        command = {
            "type": "auto",
            "seq": seq,
            "action": "push",
            "missions": [
                {
                    "mission_id": mission_id,
                    "type": "goto",
                    "frame_id": "global_map",
                    "x_m": x_m,
                    "y_m": y_m,
                }
            ],
        }
    elif intent == "clarify":
        function_call = {
            "name": "clarify",
            "arguments": {
                "question": parameters.get("question", "请提供更多信息"),
                "missing_parameters": parameters.get("missing_parameters", []),
            },
        }
        command = None
    elif intent == "patrol":
        raise ValueError("patrol is not supported by Robot Controller v4")
    else:
        raise ValueError(f"unknown intent: {intent}")

    if command is not None:
        parse_command(
            json.dumps(command),
            linear_limit_mps=float("inf"),
            angular_limit_rps=float("inf"),
            mission_batch_limit=1,
        )
    return TranslatedInstruction(function_call, command, instruction)


def translate_all(
    instructions: list[dict[str, Any]],
    *,
    seqs: list[int],
    mission_ids: list[str | None] | None = None,
) -> list[TranslatedInstruction]:
    """Translate a batch whose sequence numbers are owned by the caller."""

    if len(instructions) != len(seqs):
        raise ValueError("instructions and seqs must have equal lengths")
    ids = mission_ids if mission_ids is not None else [None] * len(instructions)
    if len(ids) != len(instructions):
        raise ValueError("instructions and mission_ids must have equal lengths")
    return [
        translate(instruction, seq=seq, mission_id=mission_id)
        for instruction, seq, mission_id in zip(instructions, seqs, ids)
    ]
