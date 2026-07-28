import json

import pytest
from mockvehicle2d.controller import AutoAction, AutoCommand, ModeAction, ModeCommand
from mockvehicle2d.instruction import translate, translate_all
from mockvehicle2d.protocol import parse_command


def _parse(command):
    return parse_command(
        json.dumps(command),
        linear_limit_mps=0.5,
        angular_limit_rps=1.5,
        mission_batch_limit=16,
    )


def test_translator_emits_only_valid_v4_commands():
    stop = translate({"intent": "stop", "parameters": {}}, seq=7)
    assert _parse(stop.command) == ModeCommand(7, ModeAction.STOP_MOTION)

    goto = translate(
        {"intent": "goto", "parameters": {"x_m": 20.0, "y_m": 30.0}},
        seq=8,
        mission_id="nl-8",
    )
    parsed = _parse(goto.command)
    assert isinstance(parsed, AutoCommand)
    assert parsed.action is AutoAction.PUSH
    assert parsed.missions[0].mission_id == "nl-8"
    assert parsed.missions[0].frame_id == "global_map"


def test_translator_keeps_execution_authority_outside_the_boundary():
    clarify = translate(
        {
            "intent": "clarify",
            "parameters": {
                "question": "目标在哪里？",
                "missing_parameters": ["x_m", "y_m"],
            },
        },
        seq=9,
    )
    assert clarify.command is None

    with pytest.raises(ValueError, match="not supported"):
        translate({"intent": "patrol", "parameters": {}}, seq=10)
    with pytest.raises(ValueError, match="mission_id"):
        translate(
            {"intent": "goto", "parameters": {"x_m": 1.0, "y_m": 2.0}},
            seq=11,
        )

    translated = translate_all(
        [
            {"intent": "stop", "parameters": {}},
            {"intent": "goto", "parameters": {"x_m": 1.0, "y_m": 2.0}},
        ],
        seqs=[12, 13],
        mission_ids=[None, "nl-13"],
    )
    assert [item.command["seq"] for item in translated] == [12, 13]
