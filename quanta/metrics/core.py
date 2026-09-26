import numpy as np

def compute_learning_times(
    loss_curves: dict,
    steps: np.ndarray | dict,
    learned_threshold: float,
) -> dict:
    results = {}
    for node_id, curve in loss_curves.items():
        curve = np.asarray(curve, dtype=float)
        node_steps = _steps_for_node(steps, node_id)
        n_points = min(len(curve), len(node_steps))
        curve = curve[:n_points]
        node_steps = node_steps[:n_points]

        below = curve < learned_threshold
        learned = bool(below.any())

        learned_step = None
        learned_eval_idx = None

        if learned:
            idx = int(np.argmax(below))
            learned_eval_idx = idx
            if idx == 0:
                learned_step = node_steps[0]
            else:
                y0, y1 = curve[idx - 1], curve[idx]
                x0, x1 = node_steps[idx - 1], node_steps[idx]
                if y1 == y0:
                    learned_step = x0
                else:
                    learned_step = x0 + (learned_threshold - y0) * (x1 - x0) / (y1 - y0)

        results[node_id] = {
            "learned": learned,
            "learned_step": learned_step,
            "learned_eval_idx": learned_eval_idx,
            "final_loss": float(curve[-1]) if len(curve) > 0 else float('nan'),
            "min_loss": (
                float(np.min(curve[np.isfinite(curve)]))
                if np.isfinite(curve).any()
                else float("nan")
            ),
        }
    return results

def compute_pde(
    graph: dict[int, list[int]],
    learning_times: dict,
    steps: np.ndarray,
    node_depths: dict[int, int]
) -> dict:
    # Full transitive closure
    def get_ancestors(node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)
        ancestors = set()
        for p in graph.get(node, []):
            ancestors.add(p)
            ancestors.update(get_ancestors(p, visited.copy()))
        return ancestors

    diffs = np.diff(steps)
    tolerance = 0.5 * np.median(diffs) if len(diffs) > 0 else 0.0

    checked_pairs = 0
    violations = 0
    censored_pairs = 0

    for child in learning_times:
        child_info = learning_times[child]
        ancestors = get_ancestors(child)
        for ancestor in ancestors:
            if ancestor not in learning_times:
                continue

            if not child_info["learned"]:
                censored_pairs += 1
                continue

            checked_pairs += 1
            ancestor_info = learning_times[ancestor]

            if not ancestor_info["learned"]:
                violations += 1
            elif ancestor_info["learned_step"] > child_info["learned_step"] + tolerance:
                violations += 1

    pde = violations / checked_pairs if checked_pairs > 0 else 0.0

    # Learned fractions
    learned_count = sum(1 for info in learning_times.values() if info["learned"])
    total_nodes = len(learning_times)
    learned_fraction = learned_count / total_nodes if total_nodes > 0 else 0.0

    depths = set(node_depths.values())
    learned_fraction_by_depth = {}
    for d in depths:
        nodes_at_depth = [n for n, depth in node_depths.items() if depth == d and n in learning_times]
        if nodes_at_depth:
            l_count = sum(1 for n in nodes_at_depth if learning_times[n]["learned"])
            learned_fraction_by_depth[d] = l_count / len(nodes_at_depth)
        else:
            learned_fraction_by_depth[d] = 0.0

    return {
        "PDE": float(pde),
        "checked_pairs": int(checked_pairs),
        "violations": int(violations),
        "censored_pairs": int(censored_pairs),
        "learned_fraction": float(learned_fraction),
        "learned_fraction_by_depth": learned_fraction_by_depth,
    }

def compute_dte(
    loss_curves: dict,
    overall_curve: np.ndarray,
    steps: np.ndarray | dict,
    learned_threshold: float,
    unlearned_threshold: float,
    overall_steps: np.ndarray | None = None,
) -> dict:
    eps = 1e-9

    def process_curve(curve, curve_steps):
        curve = np.asarray(curve, dtype=float)
        curve_steps = np.asarray(curve_steps, dtype=float)
        n_points = min(len(curve), len(curve_steps))
        curve = curve[:n_points]
        curve_steps = curve_steps[:n_points]
        total_training_steps = (
            float(curve_steps[-1] - curve_steps[0])
            if len(curve_steps) > 1
            else 1.0
        )

        below = curve < learned_threshold
        if not below.any():
            return None, "never_learned"

        end_idx = int(np.argmax(below))

        if end_idx == 0:
            return None, "already_below_unlearned_initially"

        # Interpolate end
        y0, y1 = curve[end_idx - 1], curve[end_idx]
        x0, x1 = curve_steps[end_idx - 1], curve_steps[end_idx]
        if y1 == y0:
            end_step = x0
        else:
            end_step = x0 + (learned_threshold - y0) * (x1 - x0) / (y1 - y0)

        # Look before end to find start
        above = curve[:end_idx] > unlearned_threshold
        if not above.any():
            return None, "no_valid_unlearned_to_learned_transition"

        start_idx = int(np.argmax(above[::-1])) # index from end_idx backwards
        start_idx = (end_idx - 1) - start_idx

        if start_idx == len(curve) - 1:
            y0, y1 = curve[start_idx-1], curve[start_idx]
            x0, x1 = curve_steps[start_idx-1], curve_steps[start_idx]
        else:
            y0, y1 = curve[start_idx], curve[start_idx + 1]
            x0, x1 = curve_steps[start_idx], curve_steps[start_idx + 1]

        if y1 == y0:
            start_step = x0
        else:
            start_step = x0 + (unlearned_threshold - y0) * (x1 - x0) / (y1 - y0)

        dte_steps_prelearn = (end_step - start_step) / max(end_step, eps)
        dte_steps_total = (end_step - start_step) / total_training_steps

        # dte_area
        # clip and rescale to [0, 1]
        clipped = np.clip(curve, learned_threshold, unlearned_threshold)
        rescaled = (clipped - learned_threshold) / (unlearned_threshold - learned_threshold)

        # We find best ideal step. Ideal step is 1.0 before tau, 0.0 after tau.
        # tau can be any threshold. We can just test tau at every mid-point between steps.
        areas = []
        for i in range(len(rescaled)):
            # ideal step drops at step[i]
            target = np.where(np.arange(len(rescaled)) < i, 1.0, 0.0)
            area = np.trapezoid(
                np.abs(rescaled - target),
                x=curve_steps,
            ) / total_training_steps
            areas.append(area)

        dte_area = float(np.min(areas)) if areas else 0.0

        return {
            "dte_steps_prelearn": float(dte_steps_prelearn),
            "dte_steps_total": float(dte_steps_total),
            "dte_area": float(dte_area),
            "end_step": float(end_step),
            "start_step": float(start_step),
        }, "valid"

    invalid_counts = {
        "never_learned": 0,
        "already_below_unlearned_initially": 0,
        "no_valid_unlearned_to_learned_transition": 0,
    }

    node_dtes = {}
    for node_id, curve in loss_curves.items():
        res, status = process_curve(curve, _steps_for_node(steps, node_id))
        if res is None:
            invalid_counts[status] += 1
        else:
            node_dtes[node_id] = res

    # Overall curve
    if overall_steps is None:
        overall_steps = _representative_steps(steps)
    overall_res, _ = process_curve(overall_curve, overall_steps)
    overall_dte_area = overall_res["dte_area"] if overall_res else float('nan')
    overall_dte_steps = overall_res["dte_steps_total"] if overall_res else float('nan')

    if node_dtes:
        areas = [v["dte_area"] for v in node_dtes.values()]
        steps_list = [v["dte_steps_total"] for v in node_dtes.values()]
        mean_dte_area = float(np.mean(areas))
        median_dte_area = float(np.median(areas))
        mean_dte_steps = float(np.mean(steps_list))
    else:
        mean_dte_area = float('nan')
        median_dte_area = float('nan')
        mean_dte_steps = float('nan')

    return {
        "mean_dte_area": mean_dte_area,
        "median_dte_area": median_dte_area,
        "overall_dte_area": overall_dte_area,
        "mean_minus_overall_dte_area": float(mean_dte_area - overall_dte_area) if not np.isnan(mean_dte_area) and not np.isnan(overall_dte_area) else float('nan'),
        "mean_dte_steps": mean_dte_steps,
        "overall_dte_steps": overall_dte_steps,
        "invalid_counts": invalid_counts,
        "valid_count": len(node_dtes),
    }


def compute_poset_dynamics_metrics(
    *,
    loss_curves: dict,
    steps: np.ndarray | dict,
    graph: dict | None = None,
    learned_threshold: float,
    unlearned_threshold: float,
    node_depths: dict | None = None,
    overall_curve: np.ndarray | None = None,
    overall_steps: np.ndarray | None = None,
    unit: str = "steps",
) -> dict:
    """Compute the shared PDE/DTE metric bundle for any probed poset."""
    finite_curves = {
        node_id: np.asarray(curve, dtype=float)
        for node_id, curve in loss_curves.items()
        if np.isfinite(np.asarray(curve, dtype=float)).any()
    }
    reference_steps = (
        np.asarray(overall_steps, dtype=float)
        if overall_steps is not None
        else _representative_steps(steps)
    )
    graph = graph or {}
    if node_depths is None:
        node_depths = _infer_node_depths(finite_curves.keys(), graph)
    learning_times = compute_learning_times(finite_curves, steps, learned_threshold)
    pde_metrics = compute_pde(
        graph=graph,
        learning_times=learning_times,
        steps=reference_steps,
        node_depths=node_depths,
    )
    if overall_curve is None:
        overall_curve = _mean_aligned_curve(finite_curves.values(), reference_steps)
    dte_metrics = compute_dte(
        loss_curves=finite_curves,
        overall_curve=overall_curve,
        steps=steps,
        learned_threshold=learned_threshold,
        unlearned_threshold=unlearned_threshold,
        overall_steps=reference_steps,
    )
    dte_excluded_count = int(sum(dte_metrics["invalid_counts"].values()))
    dte_total_count = int(dte_excluded_count + dte_metrics["valid_count"])
    dte_excluded_fraction = (
        dte_excluded_count / dte_total_count if dte_total_count > 0 else 0.0
    )
    pde_total_pairs = int(pde_metrics["checked_pairs"] + pde_metrics["censored_pairs"])
    pde_censored_fraction = (
        pde_metrics["censored_pairs"] / pde_total_pairs if pde_total_pairs > 0 else 0.0
    )
    return {
        "unit": str(unit),
        "learning_times": learning_times,
        "pde": pde_metrics,
        "dte": dte_metrics,
        "dte_excluded_count": dte_excluded_count,
        "dte_total_count": dte_total_count,
        "dte_excluded_fraction": float(dte_excluded_fraction),
        "dte_excluded_percent": float(100.0 * dte_excluded_fraction),
        "pde_censored_fraction": float(pde_censored_fraction),
        "pde_censored_percent": float(100.0 * pde_censored_fraction),
    }


def _steps_for_node(steps: np.ndarray | dict, node_id) -> np.ndarray:
    values = steps[node_id] if isinstance(steps, dict) else steps
    return np.asarray(values, dtype=float)


def _representative_steps(steps: np.ndarray | dict) -> np.ndarray:
    if not isinstance(steps, dict):
        return np.asarray(steps, dtype=float)
    arrays = [np.asarray(values, dtype=float) for values in steps.values()]
    min_len = min((len(values) for values in arrays), default=0)
    if min_len <= 0:
        return np.asarray([], dtype=float)
    return np.mean(np.stack([values[:min_len] for values in arrays]), axis=0)


def _infer_node_depths(nodes, graph: dict) -> dict:
    memo = {}

    def depth(node):
        if node in memo:
            return memo[node]
        parents = graph.get(node, [])
        memo[node] = 0 if not parents else 1 + max(depth(parent) for parent in parents)
        return memo[node]

    return {node: depth(node) for node in nodes}


def _mean_aligned_curve(curves, reference_steps: np.ndarray) -> np.ndarray:
    curve_list = [np.asarray(curve, dtype=float) for curve in curves]
    if not curve_list:
        return np.zeros_like(reference_steps, dtype=float)
    min_len = min([len(reference_steps), *[len(curve) for curve in curve_list]], default=0)
    if min_len <= 0:
        return np.asarray([], dtype=float)
    return np.nanmean(np.stack([curve[:min_len] for curve in curve_list]), axis=0)
