from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .compiler import FunctionalProgram
from .errors import CompilationError
from .types import PredictionEvent, PredictiveState, canonical_value


@dataclass(frozen=True)
class SequenceExample:
    id: str
    digits: tuple[int, ...]
    tokens: tuple[Any, ...]
    separator: str | None = None
    extra: tuple[tuple[str, Any], ...] = ()


def prediction_events(
    examples: Iterable[SequenceExample],
    *,
    eos_token: Any,
) -> tuple[PredictionEvent, ...]:
    """Expand complete examples into teacher-forced next-token events."""
    events = []
    for example in examples:
        sequence = (*example.tokens, eos_token)
        for position, target in enumerate(sequence):
            prefix = tuple(str(token) for token in sequence[:position])
            events.append(
                PredictionEvent(
                    id=f"{example.id}:{position}",
                    state=PredictiveState(
                        digits=example.digits,
                        prefix=prefix,
                        separator=example.separator,
                        position=position,
                        extra=example.extra,
                    ),
                    target=target,
                )
            )
    return tuple(events)


def validate_greedy(
    program: FunctionalProgram,
    examples: Iterable[SequenceExample],
    *,
    eos_token: Any,
    max_steps: int,
) -> None:
    """Verify deterministic greedy rollout, exactness, and termination."""
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    for example in examples:
        state = PredictiveState(
            digits=example.digits,
            prefix=(),
            separator=example.separator,
            position=0,
            extra=example.extra,
        )
        produced = []
        for step in range(int(max_steps)):
            token = program.predict(state)
            repeated = program.predict(state)
            if canonical_value(token) != canonical_value(repeated):
                raise CompilationError(
                    f"greedy execution is nondeterministic for example {example.id!r} at step {step}"
                )
            produced.append(token)
            if canonical_value(token) == canonical_value(eos_token):
                break
            state = replace(
                state,
                prefix=(*state.prefix, str(token)),
                position=step + 1,
            )
        else:
            raise CompilationError(
                f"greedy execution for example {example.id!r} did not emit EOS within {max_steps} steps"
            )
        expected = (*example.tokens, eos_token)
        if canonical_value(tuple(produced)) != canonical_value(expected):
            raise CompilationError(
                f"greedy mismatch for example {example.id!r}: expected {expected!r}, produced {tuple(produced)!r}"
            )
