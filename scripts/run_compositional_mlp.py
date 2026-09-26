#!/usr/bin/env python3
"""CLI for the scalable compositional-MLP loss-decomposition experiment."""

from __future__ import annotations

import argparse

from quanta.experiments.scaling_laws.compositional_mlp import (
    CompositionalMLPConfig,
    run,
)


def parse_args() -> CompositionalMLPConfig:
    parser = argparse.ArgumentParser()
    for field in CompositionalMLPConfig.__dataclass_fields__.values():
        option = "--" + field.name.replace("_", "-")
        parser.add_argument(option, type=type(field.default), default=field.default)
    return CompositionalMLPConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
