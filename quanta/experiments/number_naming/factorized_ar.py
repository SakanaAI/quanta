from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .names import english_number_name


class QuantumFamily(Enum):
    UNIT_LEX = "UNIT_LEX"
    TEEN_LEX = "TEEN_LEX"
    TEN = "TEN"
    ONE_HUNDRED = "ONE_HUNDRED"
    ONE_THOUSAND = "ONE_THOUSAND"
    TENS = "TENS"
    HUNDREDS = "HUNDREDS"
    THOUSANDS = "THOUSANDS"
    TEEN_THOUSANDS = "TEEN_THOUSANDS"
    TEN_THOUSANDS = "TEN_THOUSANDS"
    TENS_THOUSANDS = "TENS_THOUSANDS"
    HUNDRED_THOUSANDS = "HUNDRED_THOUSANDS"
    EOS_DECISION = "EOS_DECISION"
    EMIT = "EMIT"


@dataclass(frozen=True)
class QuantumInstance:
    family: QuantumFamily
    args: tuple[tuple[str, int], ...]
    path: tuple[str, ...]


@dataclass
class TokenQuantumLabel:
    number: int
    token_index: int
    prefix: tuple[str, ...]
    target: str
    owner: QuantumInstance
    active_closure: tuple[QuantumInstance, ...]


@dataclass
class FactorizedPlan:
    number: int
    words: tuple[str, ...]
    token_labels: tuple[TokenQuantumLabel, ...]
    instances: tuple[QuantumInstance, ...]
    edges: tuple[tuple[QuantumInstance, QuantumInstance], ...]


@dataclass
class Node:
    q: QuantumInstance
    words: tuple[str, ...]
    children: tuple["Node", ...]
    span: tuple[int, int]
    activation_span: tuple[int, int]


@dataclass(frozen=True)
class _TokenBuild:
    word: str
    owner: QuantumInstance
    active: tuple[QuantumInstance, ...]


def schema_edges() -> list[tuple[QuantumFamily, QuantumFamily]]:
    edges = [
        (QuantumFamily.UNIT_LEX, QuantumFamily.TENS),
        (QuantumFamily.TEN, QuantumFamily.TENS),
        (QuantumFamily.UNIT_LEX, QuantumFamily.HUNDREDS),
        (QuantumFamily.ONE_HUNDRED, QuantumFamily.HUNDREDS),
        (QuantumFamily.UNIT_LEX, QuantumFamily.THOUSANDS),
        (QuantumFamily.ONE_THOUSAND, QuantumFamily.THOUSANDS),
        (QuantumFamily.TEEN_LEX, QuantumFamily.TEEN_THOUSANDS),
        (QuantumFamily.ONE_THOUSAND, QuantumFamily.TEEN_THOUSANDS),
        (QuantumFamily.TEN, QuantumFamily.TEN_THOUSANDS),
        (QuantumFamily.ONE_THOUSAND, QuantumFamily.TEN_THOUSANDS),
        (QuantumFamily.TENS, QuantumFamily.TENS_THOUSANDS),
        (QuantumFamily.ONE_THOUSAND, QuantumFamily.TENS_THOUSANDS),
        (QuantumFamily.HUNDREDS, QuantumFamily.HUNDRED_THOUSANDS),
        (QuantumFamily.ONE_THOUSAND, QuantumFamily.HUNDRED_THOUSANDS),
    ]
    edges.extend((family, QuantumFamily.EMIT) for family in QuantumFamily if family is not QuantumFamily.EMIT)
    return edges


def build_token_labels(n: int) -> list[TokenQuantumLabel]:
    return list(build_factorized_plan(n).token_labels)


def build_factorized_plan(n: int) -> FactorizedPlan:
    n = int(n)
    token_builds = _token_builds_with_eos(n)
    words = tuple(token.word for token in token_builds)
    expected = tuple(english_number_name(n).split())
    if words[:-1] != expected or words[-1:] != ("EOS",):
        raise ValueError(f"factorized plan mismatch for {n}: {words} vs {expected}")
    root = _tree_from_tokens(n, token_builds)
    nodes = _collect_nodes(root)
    instances = _unique_instances(tuple(node.q for node in nodes) + tuple(q for token in token_builds for q in token.active))
    edges = _instance_edges(token_builds)
    labels = tuple(
        TokenQuantumLabel(
            number=n,
            token_index=index,
            prefix=words[:index],
            target=words[index],
            owner=token.owner,
            active_closure=token.active,
        )
        for index, token in enumerate(token_builds)
    )
    return FactorizedPlan(number=n, words=words, token_labels=labels, instances=instances, edges=edges)


def build_factorized_nodes(n: int) -> tuple[Node, ...]:
    return _collect_nodes(_tree_from_tokens(int(n), _token_builds_with_eos(int(n))))


def build_sub100(n: int, path: tuple[str, ...]) -> Node:
    tokens = _sub100_tokens(int(n), path, thousands=False)
    return _tree_from_tokens(int(n), tokens)


def build_chunk(n: int, path: tuple[str, ...]) -> Node:
    tokens = _chunk_tokens(int(n), path, thousands=False)
    return _tree_from_tokens(int(n), tokens)


def build_number(n: int) -> Node:
    tokens = _number_tokens(int(n))
    return _tree_from_tokens(int(n), tokens)


def _token_builds_with_eos(n: int) -> tuple[_TokenBuild, ...]:
    body = _number_tokens(n)
    eos = _instance(QuantumFamily.EOS_DECISION, {"number": n}, ("eos",))
    return body + (_TokenBuild("EOS", eos, (eos, _emit_instance())),)


def _number_tokens(n: int) -> tuple[_TokenBuild, ...]:
    if not 1 <= int(n) <= 999_999:
        raise ValueError(f"factorized number must be 1..999999, got {n}.")
    if n < 1000:
        return _chunk_tokens(n, ("number",), thousands=False)
    high, low = divmod(n, 1000)
    high_tokens = _chunk_tokens(high, ("high",), thousands=True)
    high_tokens = high_tokens + (_marker_token("thousand", _one_thousand(), ("high", "thousand")),)
    if low == 0:
        return high_tokens
    return high_tokens + _chunk_tokens(low, ("low",), thousands=False)


def _chunk_tokens(n: int, path: tuple[str, ...], *, thousands: bool) -> tuple[_TokenBuild, ...]:
    if not 1 <= int(n) <= 999:
        raise ValueError(f"chunk value must be 1..999, got {n}.")
    if n < 100:
        return _sub100_tokens(n, path, thousands=thousands)
    hundreds, remainder = divmod(n, 100)
    tokens = [
        _hundreds_value_token(hundreds, path + ("hundreds",), thousands=thousands),
        _marker_token("hundred", _one_hundred(), path + ("hundred",)),
    ]
    if remainder:
        tokens.extend(_sub100_tokens(remainder, path + ("remainder",), thousands=thousands))
    return tuple(tokens)


def _sub100_tokens(n: int, path: tuple[str, ...], *, thousands: bool) -> tuple[_TokenBuild, ...]:
    if not 1 <= int(n) <= 99:
        raise ValueError(f"sub-100 value must be 1..99, got {n}.")
    if n <= 9:
        return (_unit_value_token(n, path, thousands=thousands),)
    if n == 10:
        return (_ten_value_token(path, thousands=thousands),)
    if n <= 19:
        return (_teen_value_token(n, path, thousands=thousands),)
    tens, unit = divmod(n, 10)
    tokens = [_tens_value_token(tens, path + ("tens",), thousands=thousands)]
    if unit:
        tokens.append(_unit_value_token(unit, path + ("unit",), thousands=thousands))
    return tuple(tokens)


def _unit_value_token(value: int, path: tuple[str, ...], *, thousands: bool) -> _TokenBuild:
    unit = _instance(QuantumFamily.UNIT_LEX, {"d": value}, path + ("unit",))
    if not thousands:
        return _TokenBuild(english_number_name(value), unit, (unit, _emit_instance()))
    composite = _instance(QuantumFamily.THOUSANDS, {"d": value}, path + ("thousands",))
    return _TokenBuild(
        english_number_name(value),
        composite,
        (unit, _one_thousand(), composite, _emit_instance()),
    )


def _ten_value_token(path: tuple[str, ...], *, thousands: bool) -> _TokenBuild:
    ten = _ten()
    if not thousands:
        return _TokenBuild("ten", ten, (ten, _emit_instance()))
    composite = _instance(QuantumFamily.TEN_THOUSANDS, {}, path + ("ten_thousands",))
    return _TokenBuild("ten", composite, (ten, _one_thousand(), composite, _emit_instance()))


def _teen_value_token(value: int, path: tuple[str, ...], *, thousands: bool) -> _TokenBuild:
    teen = _instance(QuantumFamily.TEEN_LEX, {"n": value}, path + ("teen",))
    if not thousands:
        return _TokenBuild(english_number_name(value), teen, (teen, _emit_instance()))
    composite = _instance(QuantumFamily.TEEN_THOUSANDS, {"n": value}, path + ("teen_thousands",))
    return _TokenBuild(
        english_number_name(value),
        composite,
        (teen, _one_thousand(), composite, _emit_instance()),
    )


def _tens_value_token(value: int, path: tuple[str, ...], *, thousands: bool) -> _TokenBuild:
    unit = _instance(QuantumFamily.UNIT_LEX, {"d": value}, path + ("unit",))
    tens = _instance(QuantumFamily.TENS, {"t": value}, path)
    if not thousands:
        return _TokenBuild(english_number_name(value * 10), tens, (unit, _ten(), tens, _emit_instance()))
    composite = _instance(QuantumFamily.TENS_THOUSANDS, {"t": value}, path + ("tens_thousands",))
    return _TokenBuild(
        english_number_name(value * 10),
        composite,
        (unit, _ten(), tens, _one_thousand(), composite, _emit_instance()),
    )


def _hundreds_value_token(value: int, path: tuple[str, ...], *, thousands: bool) -> _TokenBuild:
    unit = _instance(QuantumFamily.UNIT_LEX, {"d": value}, path + ("unit",))
    hundreds = _instance(QuantumFamily.HUNDREDS, {"h": value}, path)
    if not thousands:
        return _TokenBuild(
            english_number_name(value),
            hundreds,
            (unit, _one_hundred(), hundreds, _emit_instance()),
        )
    composite = _instance(QuantumFamily.HUNDRED_THOUSANDS, {"h": value}, path + ("hundred_thousands",))
    return _TokenBuild(
        english_number_name(value),
        composite,
        (unit, _one_hundred(), hundreds, _one_thousand(), composite, _emit_instance()),
    )


def _marker_token(word: str, quantum: QuantumInstance, path: tuple[str, ...]) -> _TokenBuild:
    del path
    return _TokenBuild(word, quantum, (quantum, _emit_instance()))


def _ten() -> QuantumInstance:
    return _instance(QuantumFamily.TEN, {}, ("base", "ten"))


def _one_hundred() -> QuantumInstance:
    return _instance(QuantumFamily.ONE_HUNDRED, {}, ("base", "one_hundred"))


def _one_thousand() -> QuantumInstance:
    return _instance(QuantumFamily.ONE_THOUSAND, {}, ("base", "one_thousand"))


def _emit_instance() -> QuantumInstance:
    return _instance(QuantumFamily.EMIT, {}, ("emit",))


def _tree_from_tokens(n: int, tokens: tuple[_TokenBuild, ...]) -> Node:
    emit = _emit_instance()
    children = tuple(
        Node(q=q, words=(token.word,), children=(), span=(index, index + 1), activation_span=(index, index + 1))
        for index, token in enumerate(tokens)
        for q in token.active
        if q.family is not QuantumFamily.EMIT
    )
    return Node(q=emit, words=tuple(token.word for token in tokens), children=children, span=(0, len(tokens)), activation_span=(0, len(tokens)))


def _instance_edges(tokens: tuple[_TokenBuild, ...]) -> tuple[tuple[QuantumInstance, QuantumInstance], ...]:
    family_edges = set(schema_edges())
    edges: list[tuple[QuantumInstance, QuantumInstance]] = []
    for token in tokens:
        active = [q for q in token.active if q.family is not QuantumFamily.EMIT]
        for parent in active:
            for child in active:
                if (parent.family, child.family) in family_edges:
                    edges.append((parent, child))
            edges.append((parent, _emit_instance()))
    return tuple(dict.fromkeys(edges))


def _instance(family: QuantumFamily, args: dict[str, int], path: tuple[str, ...]) -> QuantumInstance:
    return QuantumInstance(
        family=family,
        args=tuple(sorted((str(key), int(value)) for key, value in args.items())),
        path=tuple(path),
    )


def _collect_nodes(node: Node) -> tuple[Node, ...]:
    return (node,) + tuple(grandchild for child in node.children for grandchild in _collect_nodes(child))


def _unique_instances(instances: tuple[QuantumInstance, ...]) -> tuple[QuantumInstance, ...]:
    return tuple(dict.fromkeys(sorted(instances, key=_instance_sort_key)))


def _instance_sort_key(instance: QuantumInstance) -> tuple[str, tuple[tuple[str, int], ...], tuple[str, ...]]:
    return (instance.family.value, instance.args, instance.path)
