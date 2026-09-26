from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterable

import torch

from .errors import CompilationError
from .types import AlignedEventTrace, CompiledQProgram, EventTrace, PredictiveState, fingerprint


@dataclass(frozen=True)
class EventSupervision:
    activity: tuple[int, ...]
    eligibility: tuple[int, ...]
    semantics: tuple[object | None, ...]
    target: object


class CompiledSupervisionIndex:
    """Canonical state-to-supervision lookup backed only by compiler traces."""

    def __init__(self, compiled: CompiledQProgram) -> None:
        self.compiled = compiled
        self._by_state: dict[str, EventTrace | AlignedEventTrace] = {}
        for trace in chain(
            compiled.validation_traces,
            compiled.training_traces,
            compiled.evaluation_traces,
        ):
            key = fingerprint(trace.state)
            previous = self._by_state.get(key)
            if previous is not None and _event_supervision(previous) != _event_supervision(trace):
                raise CompilationError(f"compiled traces disagree for predictive state in event {trace.event_id!r}.")
            self._by_state[key] = trace

    def lookup(self, state: PredictiveState) -> EventSupervision:
        key = fingerprint(state.to_dict())
        try:
            return _event_supervision(self._by_state[key])
        except KeyError as exc:
            raise KeyError(f"predictive state is absent from compiled Q-program domain: {state!r}") from exc

    def tensors(
        self,
        states: Iterable[PredictiveState],
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[object | None, ...], ...]]:
        rows = tuple(self.lookup(state) for state in states)
        activity = torch.tensor([row.activity for row in rows], dtype=torch.float32, device=device)
        eligibility = torch.tensor([row.eligibility for row in rows], dtype=torch.float32, device=device)
        return activity, eligibility, tuple(row.semantics for row in rows)


def _event_supervision(trace: EventTrace | AlignedEventTrace) -> EventSupervision:
    return EventSupervision(
        activity=trace.activity_targets,
        eligibility=trace.activity_mask,
        semantics=trace.semantic_targets,
        target=trace.target,
    )
