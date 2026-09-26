import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def load_results(results_pkl_path):
    if not os.path.exists(results_pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {results_pkl_path}")
    with open(results_pkl_path, "rb") as f:
        return pickle.load(f)


def load_config_metadata(results_pkl_path):
    config_path = os.path.join(os.path.dirname(results_pkl_path), "config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("figure config_metadata=unreadable error=%s", e)
        return {}


def graph_dependencies(is_cxor, config):
    if not is_cxor:
        return None
    graph_deps = {4: [0, 1, 2, 3]}
    raw_deps = config.get("graph_dependencies", None)
    if not raw_deps:
        return graph_deps
    try:
        if isinstance(raw_deps, str):
            raw_deps = json.loads(raw_deps)
        return {int(k): [int(x) for x in v] for k, v in raw_deps.items()}
    except Exception as e:
        logging.warning("figure graph_dependencies=unreadable error=%s", e)
        return graph_deps


def node_depth(node, deps):
    if not deps or node not in deps or not deps[node]:
        return 0
    return 1 + max(node_depth(parent, deps) for parent in deps[node])


def should_save_pdf() -> bool:
    val = os.environ.get("SAVE_IMAGES_AS_PDF", "false").lower()
    return val in ("true", "1", "yes")


def save_figure(fig, output_image_path, *, dpi=500, save_pdf=False, tight_layout=True):
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    fig.savefig(output_image_path, bbox_inches="tight", dpi=dpi)
    if save_pdf or should_save_pdf():
        pdf_path = os.path.splitext(output_image_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
