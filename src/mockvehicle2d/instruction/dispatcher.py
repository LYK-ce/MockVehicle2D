"""将 v3 NL 意图 JSON 翻译为函数调用和 Robot Controller 协议命令。

翻译层：纯确定性查表。LLM 输出不变，服务端翻译。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ClarifyRequest(Exception):
    """clarify 意图：不走 Vehicle API，触发状态机 CONFIRMING。"""

    def __init__(self, question: str, missing_parameters: list[str]) -> None:
        self.question = question
        self.missing_parameters = missing_parameters


@dataclass
class TranslatedInstruction:
    """单条指令的翻译产物。

    Attributes
    ----------
    function_call : dict
        {"name": "goto", "arguments": {"x_m": 100, "y_m": 200}}
    command : dict | None
        Robot Controller 协议命令，clarify 时为 None。
    instruction : dict
        原始 v3 JSON，向后兼容。
    """

    function_call: dict[str, Any]
    command: dict[str, Any] | None
    instruction: dict[str, Any]


def translate(instruction: dict[str, Any]) -> TranslatedInstruction:
    """v3 intent JSON → function_call + command + instruction。

    Parameters
    ----------
    instruction : dict
        单条 v3 指令，如 {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}

    Returns
    -------
    TranslatedInstruction

    Raises
    ------
    ValueError
        intent 不在翻译表中。
    """
    intent = instruction.get("intent")
    params: dict[str, Any] = instruction.get("parameters", {}) or {}

    translator = _TRANSLATORS.get(intent)
    if translator is None:
        raise ValueError(f"unknown intent: {intent}")

    fc = translator(params)
    return TranslatedInstruction(
        function_call={"name": fc["name"], "arguments": fc["arguments"]},
        command=fc["command"],
        instruction=instruction,
    )


def translate_all(instructions: list[dict[str, Any]]) -> list[TranslatedInstruction]:
    """批量翻译。"""
    return [translate(inst) for inst in instructions]


# ── 翻译表 ────────────────────────────────────────────────────


def _translate_stop(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "stop",
        "arguments": {},
        "command": {"cmd": "manual", "action": "stop"},
    }


def _translate_goto(params: dict[str, Any]) -> dict[str, Any]:
    x = params["x_m"]
    y = params["y_m"]
    return {
        "name": "goto",
        "arguments": {"x_m": x, "y_m": y},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "goto", "x_m": x, "y_m": y}],
        },
    }


def _translate_patrol(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "patrol",
        "arguments": {},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "patrol"}],
        },
    }


def _translate_clarify(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "clarify",
        "arguments": {
            "question": params.get("question", "请提供更多信息"),
            "missing_parameters": params.get("missing_parameters", []),
        },
        "command": None,
    }


_TRANSLATORS: dict[str, Any] = {
    "stop": _translate_stop,
    "goto": _translate_goto,
    "patrol": _translate_patrol,
    "clarify": _translate_clarify,
}
