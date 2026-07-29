"""MockVehicle2D NL Instruction — natural language vehicle command processing.

Phase 1: model parsing + validation without vehicle control (offline closed loop).
"""

from mockvehicle2d.instruction.authority import AuthorityManager, AuthorityLevel
from mockvehicle2d.instruction.compiler import TaskCompiler
from mockvehicle2d.instruction.dispatcher import (
    ClarifyRequest,
    TranslatedInstruction,
    translate,
    translate_all,
)
from mockvehicle2d.instruction.llm_client import LLMClient
from mockvehicle2d.instruction.state_machine import (
    InstructionState,
    InstructionStateMachine,
)
from mockvehicle2d.instruction.validator import (
    SchemaValidator,
    SemanticValidator,
    SafetyValidator,
    ValidationResult,
)

__all__ = [
    "AuthorityManager",
    "AuthorityLevel",
    "ClarifyRequest",
    "InstructionState",
    "InstructionStateMachine",
    "LLMClient",
    "SafetyValidator",
    "SchemaValidator",
    "SemanticValidator",
    "TaskCompiler",
    "TranslatedInstruction",
    "ValidationResult",
    "translate",
    "translate_all",
]
