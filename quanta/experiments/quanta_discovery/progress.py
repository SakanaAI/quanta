from __future__ import annotations

import sys
from typing import Any, TextIO

from tqdm import tqdm


_PHASE_LABELS = {
    "configuration_setup": "setup",
    "model_and_event_panel": "event panel",
    "training_trajectory": "training trajectory",
    "trajectory_serialization": "save trajectory",
    "event_priority_analysis": "event priority",
    "composition_graph": "composition graph",
}

_PHASE_UNITS = {
    "training_trajectory": "step",
    "event_priority_analysis": "checkpoint",
}


class QuantaDiscoveryProgress:
    """Compact tqdm console surface for one quanta-discovery run."""

    def __init__(
        self,
        save_dir: str,
        *,
        file: TextIO | None = None,
        mininterval: float = 1.0,
    ) -> None:
        self.save_dir = str(save_dir)
        self.file = sys.stderr if file is None else file
        self.mininterval = float(mininterval)
        self._bar: tqdm[Any] | None = None
        self._phase: str | None = None
        self._completed = 0
        self._postfix: dict[str, str] = {}
        tqdm.write(
            f"quanta_discovery | output={self.save_dir}",
            file=self.file,
        )

    def start_phase(
        self,
        phase: str,
        *,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None:
        del fields
        self._close_bar()
        self._phase = str(phase)
        self._completed = 0
        self._postfix = {}
        description = _PHASE_LABELS.get(self._phase, self._phase)
        if worker is not None:
            description = f"{description} worker {int(worker)}"
        self._bar = tqdm(
            total=total,
            desc=description,
            unit=_PHASE_UNITS.get(self._phase, "item"),
            dynamic_ncols=True,
            mininterval=self.mininterval,
            leave=True,
            file=self.file,
        )

    def update_phase(
        self,
        phase: str,
        *,
        completed: int,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None:
        del total, worker
        if self._bar is None or self._phase != str(phase):
            return
        completed = int(completed)
        postfix = _important_fields(str(phase), fields)
        if postfix:
            self._postfix.update(postfix)
            self._bar.set_postfix(self._postfix, refresh=False)
        self._bar.update(max(0, completed - self._completed))
        self._completed = max(self._completed, completed)

    def complete_phase(
        self,
        phase: str,
        *,
        completed: int,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None:
        self.update_phase(
            phase,
            completed=completed,
            total=total,
            worker=worker,
            **fields,
        )
        if self._bar is not None and self._phase == str(phase):
            postfix = _important_fields(str(phase), fields)
            if postfix:
                self._postfix.update(postfix)
                self._bar.set_postfix(self._postfix, refresh=False)
            self._bar.refresh()
            self._bar.close()
            self._bar = None
            self._phase = None
            self._postfix = {}

    def complete_run(self, *, elapsed_seconds: float, **fields: Any) -> None:
        self._close_bar()
        summary = [f"complete in {_duration(elapsed_seconds)}"]
        for key, label in (
            ("analyzed_layers", "layers"),
            ("candidate_slot_ceiling", "candidate slots"),
        ):
            if key in fields:
                summary.append(f"{label}={int(fields[key])}")
        summary.append(f"output={self.save_dir}")
        tqdm.write("quanta_discovery | " + " | ".join(summary), file=self.file)

    def fail_run(self, *, elapsed_seconds: float, error: BaseException) -> None:
        self._close_bar()
        tqdm.write(
            "quanta_discovery | failed after "
            f"{_duration(elapsed_seconds)} | {type(error).__name__}: {error}",
            file=self.file,
        )

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            self._phase = None
            self._postfix = {}


def _important_fields(phase: str, fields: dict[str, Any]) -> dict[str, str]:
    if phase == "configuration_setup":
        return _select(fields, device="device", analysis_gpus="gpus")
    if phase == "model_and_event_panel":
        return _select(
            fields,
            train_prediction_events="train events",
            eval_prediction_events="eval events",
            transformer_layers="layers",
        )
    if phase == "training_trajectory":
        return _select(
            fields,
            mean_train_event_loss="train loss",
            mean_eval_event_loss="eval loss",
            checkpoints="checkpoints",
        )
    if phase == "trajectory_serialization":
        return _select(
            fields,
            checkpoints="checkpoints",
            optimizer_step="step",
        )
    if phase == "event_priority_analysis":
        return _select(
            fields,
            optimizer_step="step",
            priority_closure_error="priority identity error",
        )
    return {}


def _select(fields: dict[str, Any], **mapping: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for source, label in mapping.items():
        if source not in fields or fields[source] is None:
            continue
        value = fields[source]
        if isinstance(value, float):
            selected[label] = f"{value:.4g}"
        else:
            selected[label] = str(value)
    return selected


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remainder:02d}s"
    if minutes:
        return f"{minutes:d}m {remainder:02d}s"
    return f"{remainder:d}s"
