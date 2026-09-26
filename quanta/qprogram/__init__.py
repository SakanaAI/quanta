"""Compiler for explicitly annotated, exact functional Q-programs."""

from .compiler import (
    CompileOptions,
    FunctionalProgram,
    compile_qprogram,
    validate_exhaustive_domain,
)
from .errors import CompilationError, ComplexityError, DefinitionError, QProgramError, TraceError
from .domains import SequenceExample, prediction_events, validate_greedy
from .runtime import QRegistry, SemanticValue, unwrap, when, when_item
from .neural import ParameterAudit, activity_tensors, audit_homogeneous_parameters
from .supervision import CompiledSupervisionIndex, EventSupervision
from .types import (
    COMPILER_VERSION,
    AlignedEventTrace,
    CompiledQProgram,
    ExhaustiveValidationReport,
    PredictionEvent,
    PredictiveState,
    canonical_key,
)

__all__ = [
    "COMPILER_VERSION",
    "AlignedEventTrace",
    "CompilationError",
    "CompileOptions",
    "CompiledQProgram",
    "CompiledSupervisionIndex",
    "ComplexityError",
    "DefinitionError",
    "FunctionalProgram",
    "EventSupervision",
    "ExhaustiveValidationReport",
    "PredictionEvent",
    "PredictiveState",
    "QProgramError",
    "QRegistry",
    "ParameterAudit",
    "SemanticValue",
    "SequenceExample",
    "TraceError",
    "compile_qprogram",
    "activity_tensors",
    "audit_homogeneous_parameters",
    "canonical_key",
    "prediction_events",
    "unwrap",
    "validate_greedy",
    "validate_exhaustive_domain",
    "when",
    "when_item",
]
