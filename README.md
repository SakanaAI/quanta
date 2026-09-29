# Quanta

This repository contains the code for the paper "Neural Dynamics as the Composition of Quantized Units". We experiment with a discrete approximation of learning dynamics in which reusable computations are acquired in an ordered, input-selective way. It provides the models, controlled Hierarchical Sparse Parity experiments, NumberNaming Q-discovery pipeline, and Q-alignment experiments used to test loss decomposition, resource scaling, executable recovery, and steering while keeping experiment logic and saved-artifact contracts explicit.

## Repository structure

- `quanta/` contains reusable datasets, models, Q-programs, experiment code, metrics, and plotting utilities.
- `configs/` contains one maintained configuration for each config-driven replication.
- `scripts/` contains portable training and analysis entry points for the main experiments.
- `tests/` checks configuration, training, discovery, compilation, and analysis behavior.
- `.experiments/` and `.artifacts/` are ignored output directories created at runtime.

## Main replications

Install the environment and run the tests:

```bash
uv sync
uv run python -m pytest
```

Run the compact loss-decomposition control through the config CLI, or reproduce the composed-task panel:

```bash
uv run main.py scaling_laws/cnand/loss_decomposition
uv run python -m scripts.reproduce_loss_decomposition
```

Run every point in the parameter- and data-scaling panels. These are long GPU campaigns; each script also accepts one `--alpha` and, for parameter scaling, one `--width`.

```bash
uv run python -m scripts.reproduce_parameter_scaling --all
uv run python -m scripts.reproduce_data_scaling --all
```

Run Q-discovery and its executable Q-model, then compile a program-derived Q-model and run the matched Q-alignment/control pair:

```bash
uv run python -m scripts.reproduce_q_discovery --device cuda
uv run python -m scripts.reproduce_q_steering --device cuda
```

Outputs, including resolved configurations, status files, summaries, trajectories, and checkpoints, are saved under `.experiments/`.

## Paper

Jacopo Minniti, Aravinth Kulanthaivelu, Richard Sproat. 2026. "Neural Dynamics
as the Composition of Quantized Units". https://arxiv.org/abs/2609.32487.
