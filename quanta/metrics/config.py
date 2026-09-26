import os


def load_metric_config() -> tuple[float, float]:
    unlearned = float(os.environ.get("UNLEARNED_QUANTA_THRESHOLD", "0.85"))
    learned = float(os.environ.get("LEARNED_QUANTA_THRESHOLD", "0.05"))

    if not (0 < learned < unlearned):
        raise ValueError(
            "Thresholds must satisfy "
            "0 < LEARNED_QUANTA_THRESHOLD < UNLEARNED_QUANTA_THRESHOLD, "
            f"got learned={learned}, unlearned={unlearned}"
        )

    return unlearned, learned
