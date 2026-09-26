from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import time
from typing import Any, Protocol

from quanta.experiments.common import log_event


class ProgressObserver(Protocol):
    """Optional human-facing observer for persisted phase progress."""

    def start_phase(
        self,
        phase: str,
        *,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None: ...

    def update_phase(
        self,
        phase: str,
        *,
        completed: int,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None: ...

    def complete_phase(
        self,
        phase: str,
        *,
        completed: int,
        total: int | None,
        worker: int | None,
        **fields: Any,
    ) -> None: ...

    def complete_run(self, *, elapsed_seconds: float, **fields: Any) -> None: ...

    def fail_run(self, *, elapsed_seconds: float, error: BaseException) -> None: ...


@dataclass
class PhaseProgress:
    """Live phase timing with a phase-local ETA."""

    phase: str
    total: int | None = None
    worker: int | None = None
    experiment_name: str = "quanta_discovery"
    emit_log_events: bool = True
    observer: ProgressObserver | None = None
    run_started_monotonic: float | None = None
    started_monotonic: float = field(default_factory=time.perf_counter)
    _last_monotonic: float = field(init=False)
    _last_completed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._last_monotonic = self.started_monotonic
        if self.total is not None and int(self.total) < 0:
            raise ValueError("progress total must be nonnegative.")

    def start(self, **fields: Any) -> dict[str, Any]:
        payload = self.snapshot(0)
        if self.emit_log_events:
            log_event(
                self.experiment_name,
                "phase_start",
                phase=self.phase,
                worker=self.worker,
                total_items=self.total,
                **payload,
                **fields,
            )
        if self.observer is not None:
            self.observer.start_phase(
                self.phase,
                total=self.total,
                worker=self.worker,
                **payload,
                **fields,
            )
        return payload

    def update(self, completed: int, **fields: Any) -> dict[str, Any]:
        completed = int(completed)
        if completed < self._last_completed:
            raise ValueError("phase progress cannot move backwards.")
        if self.total is not None and completed > int(self.total):
            raise ValueError("phase progress cannot exceed its total.")
        now = time.perf_counter()
        interval_seconds = now - self._last_monotonic
        interval_items = completed - self._last_completed
        payload = self.snapshot(completed, now=now)
        if self.emit_log_events:
            log_event(
                self.experiment_name,
                "phase_progress",
                phase=self.phase,
                worker=self.worker,
                completed_items=completed,
                total_items=self.total,
                interval_items=interval_items,
                interval_seconds=interval_seconds,
                **payload,
                **fields,
            )
        if self.observer is not None:
            self.observer.update_phase(
                self.phase,
                completed=completed,
                total=self.total,
                worker=self.worker,
                interval_items=interval_items,
                interval_seconds=interval_seconds,
                **payload,
                **fields,
            )
        self._last_monotonic = now
        self._last_completed = completed
        return payload

    def complete(self, **fields: Any) -> float:
        now = time.perf_counter()
        duration = now - self.started_monotonic
        completed = self.total
        if completed is None:
            completed = self._last_completed
        payload = self.snapshot(int(completed), now=now)
        if self.emit_log_events:
            log_event(
                self.experiment_name,
                "phase_complete",
                phase=self.phase,
                worker=self.worker,
                completed_items=completed,
                total_items=self.total,
                phase_duration_seconds=duration,
                **payload,
                **fields,
            )
        if self.observer is not None:
            self.observer.complete_phase(
                self.phase,
                completed=int(completed),
                total=self.total,
                worker=self.worker,
                phase_duration_seconds=duration,
                **payload,
                **fields,
            )
        return duration

    def snapshot(
        self,
        completed: int,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        current = time.perf_counter() if now is None else float(now)
        elapsed = max(0.0, current - self.started_monotonic)
        completed = int(completed)
        rate = float(completed / elapsed) if completed > 0 and elapsed > 0 else None
        eta = None
        progress_fraction = None
        if self.total is not None:
            total = int(self.total)
            progress_fraction = 1.0 if total == 0 else completed / total
            if rate is not None:
                eta = max(0.0, (total - completed) / rate)
        total_elapsed = None
        if self.run_started_monotonic is not None:
            total_elapsed = max(
                0.0,
                current - float(self.run_started_monotonic),
            )
        return {
            "phase_elapsed_seconds": elapsed,
            "phase_eta_seconds": eta,
            "progress_fraction": progress_fraction,
            "items_per_second": rate,
            "total_elapsed_seconds": total_elapsed,
        }


class RunTimingRecorder:
    """Persist completed phase durations so crashes retain timing evidence."""

    def __init__(
        self,
        save_dir: str,
        *,
        started_monotonic: float | None = None,
        started_at_unix: float | None = None,
        experiment_name: str = "quanta_discovery",
        emit_log_events: bool = True,
        observer: ProgressObserver | None = None,
    ) -> None:
        self.save_dir = str(save_dir)
        self.experiment_name = str(experiment_name)
        self.emit_log_events = bool(emit_log_events)
        self.observer = observer
        self.started_monotonic = (
            time.perf_counter()
            if started_monotonic is None
            else float(started_monotonic)
        )
        self.started_at_unix = (
            time.time() if started_at_unix is None else float(started_at_unix)
        )
        self.phases: dict[str, dict[str, Any]] = {}
        self.status = "running"
        self._write()

    def start_phase(
        self,
        name: str,
        *,
        total: int | None = None,
        **fields: Any,
    ) -> PhaseProgress:
        if name in self.phases:
            raise ValueError(f"phase {name!r} was already recorded.")
        progress = PhaseProgress(
            phase=name,
            total=total,
            experiment_name=self.experiment_name,
            emit_log_events=self.emit_log_events,
            observer=self.observer,
            run_started_monotonic=self.started_monotonic,
        )
        self.phases[name] = {
            "status": "running",
            "started_after_run_seconds": (
                progress.started_monotonic - self.started_monotonic
            ),
            "total_items": total,
        }
        self._write()
        progress.start(**fields)
        return progress

    def finish_phase(
        self,
        progress: PhaseProgress,
        **fields: Any,
    ) -> float:
        duration = progress.complete(**fields)
        record = self.phases[progress.phase]
        record.update(
            {
                "status": "complete",
                "duration_seconds": duration,
                "completed_at_run_seconds": (
                    time.perf_counter() - self.started_monotonic
                ),
            }
        )
        self._write()
        return duration

    def complete(self, **fields: Any) -> float:
        self.status = "complete"
        elapsed = time.perf_counter() - self.started_monotonic
        self._write(total_elapsed_seconds=elapsed)
        if self.observer is not None:
            self.observer.complete_run(
                elapsed_seconds=elapsed,
                **fields,
            )
        return elapsed

    def fail(self, error: BaseException) -> float:
        self.status = "failed"
        elapsed = time.perf_counter() - self.started_monotonic
        for record in self.phases.values():
            if record["status"] == "running":
                record["status"] = "failed"
                record["failed_at_run_seconds"] = elapsed
        self._write(
            total_elapsed_seconds=elapsed,
            error_type=type(error).__name__,
            error=str(error),
        )
        if self.observer is not None:
            self.observer.fail_run(
                elapsed_seconds=elapsed,
                error=error,
            )
        return elapsed

    def _write(self, **extra: Any) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        now = time.time()
        elapsed = time.perf_counter() - self.started_monotonic
        payload = {
            "started_at_unix": self.started_at_unix,
            "updated_at_unix": now,
            "status": self.status,
            "total_elapsed_seconds": elapsed,
            "phases": self.phases,
            **extra,
        }
        with open(
            os.path.join(self.save_dir, "timings.json"),
            "w",
        ) as handle:
            json.dump(_finite_json(payload), handle, indent=2)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
