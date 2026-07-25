"""Three-layer instruction validation: schema, semantics, and safety.

SchemaValidator   — JSON Schema v1 conformance (jsonschema)
SemanticValidator — map bounds, passability, distance limits
SafetyValidator   — delegates to existing SafetyRuntime
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import LocalSafetyRuntime


_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "v1.json"

_MAX_MAP_SIZE = 255
_DEFAULT_MAX_DISTANCE_M = 10.0


class SchemaValidator:
    """Validates JSON against the v1 instruction schema using jsonschema."""

    def __init__(self) -> None:
        self._schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self._validator_cls = jsonschema.validators.validator_for(self._schema)
        self._validator = self._validator_cls(self._schema)

    def validate(self, instruction: dict) -> tuple[bool, str]:
        """Return (is_valid, error_message)."""
        errors = list(self._validator.iter_errors(instruction))
        if not errors:
            return True, ""
        messages = [self._format_error(error) for error in errors[:5]]
        return False, "; ".join(messages)

    @staticmethod
    def _format_error(error: jsonschema.ValidationError) -> str:
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
        return f"{path}: {error.message}"


@dataclass
class ValidationResult:
    """Unified validation result from the three-layer pipeline."""

    valid: bool
    layer: str  # "schema" | "semantic" | "safety"
    message: str = ""

    detailed_errors: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> ValidationResult:
        return cls(valid=True, layer="")

    @classmethod
    def fail(cls, layer: str, message: str, details: list[str] | None = None) -> ValidationResult:
        return cls(valid=False, layer=layer, message=message, detailed_errors=details or [])


class SemanticValidator:
    """Context-aware checks: map bounds, passability, distance limits.

    Parameters
    ----------
    grid : MapGrid
        The map for bounds and passability checks.
    max_distance_m : float
        Maximum allowed move_distance (default 10.0).
    """

    def __init__(
        self, grid: MapGrid | None, max_distance_m: float = _DEFAULT_MAX_DISTANCE_M
    ) -> None:
        self._grid = grid
        self._max_distance_m = max_distance_m

    def validate(self, instruction: dict) -> tuple[bool, str]:
        """Return (is_valid, error_message)."""
        intent = instruction.get("intent")
        params = instruction.get("parameters", {}) or {}
        if intent == "goto_point":
            return self._validate_goto_point(params)
        if intent == "move_distance":
            return self._validate_move_distance(params)
        if intent == "rotate":
            return self._validate_rotate(params)
        return True, ""

    def _validate_goto_point(self, params: dict) -> tuple[bool, str]:
        x_m = params.get("x_m")
        y_m = params.get("y_m")
        if x_m is None or y_m is None:
            return False, "goto_point requires x_m and y_m"
        if not (0 <= x_m <= _MAX_MAP_SIZE and 0 <= y_m <= _MAX_MAP_SIZE):
            return False, f"target ({x_m}, {y_m}) out of map bounds [0, {_MAX_MAP_SIZE}]"
        if self._grid is None:
            return True, ""
        gx, gy = int(x_m), int(y_m)
        if not self._grid.in_bounds(gx, gy):
            return False, f"target cell ({gx}, {gy}) out of map bounds"
        if self._grid.is_wall(gx, gy):
            return False, f"target cell ({gx}, {gy}) is a wall"
        if self._grid.is_void(gx, gy):
            return False, f"target cell ({gx}, {gy}) is void (no ground)"
        return True, ""

    def _validate_move_distance(self, params: dict) -> tuple[bool, str]:
        distance = params.get("distance_m", 0)
        direction = params.get("direction")
        if direction not in ("forward", "backward"):
            return False, f"invalid direction: {direction!r}"
        if not (0.01 <= distance <= self._max_distance_m):
            return False, (
                f"distance {distance}m outside allowed range "
                f"[0.01, {self._max_distance_m}]"
            )
        return True, ""

    @staticmethod
    def _validate_rotate(params: dict) -> tuple[bool, str]:
        angle = params.get("angle_deg", 0)
        if angle == 0:
            return False, "rotate angle must be non-zero"
        return True, ""


class SafetyValidator:
    """Delegates safety checks to the existing SafetyRuntime."""

    def __init__(self, safety: LocalSafetyRuntime) -> None:
        self._safety = safety

    def validate(self, instruction: dict | None = None) -> tuple[bool, str]:
        """Return (is_safe, reason).  Instruction is unused in Phase 1 but
        accepted for future use (e.g. checking target proximity)."""
        _ = instruction
        state = self._safety.decision.state
        if state == "fault":
            return False, "safety is in fault state"
        if state == "stopped":
            return False, f"safety blocked: {self._safety.decision.reason or 'hard stop'}"
        return True, ""


def run_validation_pipeline(
    instruction: dict,
    schema_validator: SchemaValidator | None = None,
    semantic_validator: SemanticValidator | None = None,
    safety_validator: SafetyValidator | None = None,
) -> ValidationResult:
    """Run schema → semantic → safety validation and return the first failure.

    Returns ValidationResult.ok() if all layers pass.
    """
    sv = schema_validator or SchemaValidator()
    valid, error = sv.validate(instruction)
    if not valid:
        return ValidationResult.fail("schema", error)

    if semantic_validator is not None:
        valid, error = semantic_validator.validate(instruction)
        if not valid:
            return ValidationResult.fail("semantic", error)

    if safety_validator is not None:
        valid, error = safety_validator.validate(instruction)
        if not valid:
            return ValidationResult.fail("safety", error)

    return ValidationResult.ok()
