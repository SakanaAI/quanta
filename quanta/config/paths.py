from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONFIG_ROOT = Path("configs")
EXPERIMENT_ROOT = CONFIG_ROOT
UTILITY_ROOT = CONFIG_ROOT
DEFAULT_PLOT_CONFIG = CONFIG_ROOT / "plot" / "default.yaml"
EXPERIMENT_NAMES = {
    "quanta_discovery",
    "quanta_net",
    "quanta_steering",
    "posets_probing",
    "scaling_laws",
}


@dataclass(frozen=True)
class ResolvedRunConfig:
    experiment_name: str
    experiment_path: Path
    plot_path: Path


def resolve_run_config(
    config_name: str | Path,
    *,
    plot_name: str | Path | None = None,
) -> ResolvedRunConfig:
    experiment_path = resolve_experiment_config(config_name)
    return ResolvedRunConfig(
        experiment_name=experiment_name_from_path(experiment_path),
        experiment_path=experiment_path,
        plot_path=resolve_utility_config("plot", plot_name),
    )


def resolve_experiment_config(config_name: str | Path) -> Path:
    requested = _with_yaml_suffix(Path(config_name))
    candidates = _explicit_candidates(requested)
    simplified = _strip_config_prefix(requested)

    candidates.append(CONFIG_ROOT / simplified)
    return _first_existing(requested, candidates)


def resolve_utility_config(kind: str, config_name: str | Path | None) -> Path:
    if kind != "plot":
        raise ValueError(f"Unsupported utility config kind: {kind!r}")
    if config_name is None:
        return DEFAULT_PLOT_CONFIG

    requested = _with_yaml_suffix(Path(config_name))
    candidates = _explicit_candidates(requested)
    simplified = _strip_config_prefix(requested)

    candidates.append(CONFIG_ROOT / kind / _strip_prefix(simplified, kind))
    return _first_existing(requested, candidates)


def experiment_name_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    for marker in ("experiments",):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]

    if "configs" in parts:
        index = parts.index("configs")
        if index + 1 < len(parts) and parts[index + 1] not in {"plot", "utilities"}:
            return parts[index + 1]
    matches = EXPERIMENT_NAMES.intersection(parts)
    if len(matches) == 1:
        return matches.pop()
    raise ValueError(
        "Cannot infer the experiment from the config path. Use a config under "
        "configs/<experiment>/."
    )


def _with_yaml_suffix(path: Path) -> Path:
    return path if path.suffix in {".yaml", ".yml"} else path.with_suffix(".yaml")


def _explicit_candidates(path: Path) -> list[Path]:
    if path.is_absolute() or str(path).startswith(".") or path.exists():
        return [path]
    return []


def _strip_config_prefix(path: Path) -> Path:
    if path.parts[:1] == ("configs",):
        return Path(*path.parts[1:])
    return path


def _strip_prefix(path: Path, prefix: str) -> Path:
    if path.parts[:1] == (prefix,):
        return Path(*path.parts[1:])
    return path


def _first_existing(requested: Path, candidates: list[Path]) -> Path:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config file not found: {requested}")
