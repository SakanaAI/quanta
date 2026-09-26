from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from .errors import CompilationError
from .types import Dependency, PosetStructure


def build_poset(
    nodes: Iterable[str],
    dependency_kinds: Mapping[tuple[str, str], set[str]],
    witnesses: Mapping[tuple[str, str], str],
    source_reads: Mapping[str, tuple[str, ...]],
) -> PosetStructure:
    ordered_nodes = tuple(sorted(set(nodes)))
    unreduced = tuple(sorted(dependency_kinds))
    cycle = _find_cycle(ordered_nodes, unreduced)
    if cycle:
        details = []
        for parent, child in zip(cycle, cycle[1:]):
            kinds = "+".join(sorted(dependency_kinds[(parent, child)]))
            details.append(f"{parent} -[{kinds}, event={witnesses[(parent, child)]}]-> {child}")
        raise CompilationError("compiled dependency graph contains a cycle: " + "; ".join(details))
    reduced = transitive_reduction(ordered_nodes, unreduced)
    dependencies = tuple(
        Dependency(
            parent=parent,
            child=child,
            kinds=tuple(sorted(dependency_kinds[(parent, child)])),
            witness_event=witnesses[(parent, child)],
        )
        for parent, child in reduced
    )
    unreduced_dependencies = tuple(
        Dependency(
            parent=parent,
            child=child,
            kinds=tuple(sorted(dependency_kinds[(parent, child)])),
            witness_event=witnesses[(parent, child)],
        )
        for parent, child in unreduced
    )
    parents = {node: tuple(parent for parent, child in reduced if child == node) for node in ordered_nodes}
    children = {node: tuple(child for parent, child in reduced if parent == node) for node in ordered_nodes}
    topological_order = _topological_order(ordered_nodes, reduced)
    ancestors = {node: _closure(node, parents) for node in ordered_nodes}
    descendants = {node: _closure(node, children) for node in ordered_nodes}
    depth_by_node = {
        node: 0 if not parents[node] else 1 + max(_depth(parent, parents, {}) for parent in parents[node])
        for node in topological_order
    }
    max_depth = max(depth_by_node.values(), default=-1)
    levels = tuple(
        tuple(node for node in topological_order if depth_by_node[node] == level)
        for level in range(max_depth + 1)
    )
    return PosetStructure(
        nodes=ordered_nodes,
        edges=reduced,
        dependencies=dependencies,
        unreduced_edges=unreduced,
        unreduced_dependencies=unreduced_dependencies,
        parents=parents,
        children=children,
        ancestors=ancestors,
        descendants=descendants,
        topological_order=topological_order,
        levels=levels,
        depth=max_depth + 1,
        source_adjacent=tuple(node for node in ordered_nodes if source_reads.get(node)),
        leaves=tuple(node for node in ordered_nodes if not children[node]),
        joins=tuple(node for node in ordered_nodes if len(parents[node]) > 1),
    )


def transitive_reduction(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    ordered_edges = tuple(sorted(set(edges)))
    children: dict[str, set[str]] = defaultdict(set)
    for parent, child in ordered_edges:
        children[parent].add(child)

    def alternate_path(start: str, target: str, skipped: tuple[str, str]) -> bool:
        stack = [start]
        seen = {start}
        while stack:
            current = stack.pop()
            for child in sorted(children[current]):
                if (current, child) == skipped:
                    continue
                if child == target:
                    return True
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return False

    node_set = set(nodes)
    return tuple(
        edge
        for edge in ordered_edges
        if edge[0] in node_set and edge[1] in node_set and not alternate_path(edge[0], edge[1], edge)
    )


def _find_cycle(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> tuple[str, ...] | None:
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    visited: set[str] = set()
    active: list[str] = []

    def visit(node: str) -> tuple[str, ...] | None:
        if node in active:
            index = active.index(node)
            return tuple((*active[index:], node))
        if node in visited:
            return None
        active.append(node)
        for child in sorted(children[node]):
            cycle = visit(child)
            if cycle:
                return cycle
        active.pop()
        visited.add(node)
        return None

    for node in nodes:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _topological_order(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    incoming = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        incoming[child] += 1
        children[parent].append(child)
    ready = sorted(node for node, degree in incoming.items() if degree == 0)
    result = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for child in sorted(children[node]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort()
    return tuple(result)


def _closure(node: str, adjacency: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    result: set[str] = set()
    stack = list(adjacency[node])
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(adjacency[current])
    return tuple(sorted(result))


def _depth(node: str, parents: Mapping[str, tuple[str, ...]], cache: dict[str, int]) -> int:
    if node not in cache:
        cache[node] = 0 if not parents[node] else 1 + max(_depth(parent, parents, cache) for parent in parents[node])
    return cache[node]
