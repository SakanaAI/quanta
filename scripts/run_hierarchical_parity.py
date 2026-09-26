#!/usr/bin/env python3
"""CLI for hierarchical fixed-pool sparse-parity learning dynamics."""

from __future__ import annotations

import argparse

from quanta.experiments.scaling_laws.hierarchical_parity import (
    HierarchicalParityConfig,
    run,
)


def parse_args() -> HierarchicalParityConfig:
    parser = argparse.ArgumentParser()
    for field in HierarchicalParityConfig.__dataclass_fields__.values():
        option = "--" + field.name.replace("_", "-")
        parser.add_argument(option, type=type(field.default), default=field.default)
    return HierarchicalParityConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
