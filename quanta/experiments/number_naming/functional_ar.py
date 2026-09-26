from __future__ import annotations

from dataclasses import dataclass

from .names import english_number_name

GROUP_HIGH = "GROUP_HIGH"
GROUP_LOW = "GROUP_LOW"
CHUNK_PRESENT = "CHUNK_PRESENT"
CHUNK_EMPTY = "CHUNK_EMPTY"
CHUNK_HAS_HUNDREDS = "CHUNK_HAS_HUNDREDS"
CHUNK_HAS_TAIL = "CHUNK_HAS_TAIL"
TAIL_IS_TEN = "TAIL_IS_TEN"
TAIL_IS_TEEN = "TAIL_IS_TEEN"
TAIL_HAS_TENS = "TAIL_HAS_TENS"
TAIL_HAS_UNITS = "TAIL_HAS_UNITS"
VALUE_READ = "VALUE_READ"
DIGIT_ONE = "DIGIT_ONE"
DIGIT_TWO = "DIGIT_TWO"
DIGIT_THREE = "DIGIT_THREE"
DIGIT_FOUR = "DIGIT_FOUR"
DIGIT_FIVE = "DIGIT_FIVE"
DIGIT_SIX = "DIGIT_SIX"
DIGIT_SEVEN = "DIGIT_SEVEN"
DIGIT_EIGHT = "DIGIT_EIGHT"
DIGIT_NINE = "DIGIT_NINE"
EMIT_VALUE = "EMIT_VALUE"
EMIT_HUNDRED = "EMIT_HUNDRED"
EMIT_THOUSAND = "EMIT_THOUSAND"
EMIT_EOS = "EMIT_EOS"
FORM_UNIT = "FORM_UNIT"
FORM_TEN = "FORM_TEN"
FORM_TENS = "FORM_TENS"
FORM_TEEN = "FORM_TEEN"
FORM_HUNDRED_DIGIT = "FORM_HUNDRED_DIGIT"

DIGIT_NODE_BY_DIGIT = {
    1: DIGIT_ONE,
    2: DIGIT_TWO,
    3: DIGIT_THREE,
    4: DIGIT_FOUR,
    5: DIGIT_FIVE,
    6: DIGIT_SIX,
    7: DIGIT_SEVEN,
    8: DIGIT_EIGHT,
    9: DIGIT_NINE,
}

DIGIT_NAME_BY_DIGIT = {
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE",
}

FUNCTIONAL_EMIT_NODES: tuple[str, ...] = ()

FUNCTIONAL_NODE_ORDER = (
    GROUP_HIGH,
    GROUP_LOW,
    CHUNK_PRESENT,
    CHUNK_EMPTY,
    CHUNK_HAS_HUNDREDS,
    CHUNK_HAS_TAIL,
    TAIL_IS_TEN,
    TAIL_IS_TEEN,
    TAIL_HAS_TENS,
    TAIL_HAS_UNITS,
    VALUE_READ,
    DIGIT_ONE,
    DIGIT_TWO,
    DIGIT_THREE,
    DIGIT_FOUR,
    DIGIT_FIVE,
    DIGIT_SIX,
    DIGIT_SEVEN,
    DIGIT_EIGHT,
    DIGIT_NINE,
    FORM_UNIT,
    FORM_TEN,
    FORM_TENS,
    FORM_TEEN,
    FORM_HUNDRED_DIGIT,
    EMIT_VALUE,
    EMIT_HUNDRED,
    EMIT_THOUSAND,
    EMIT_EOS,
)

FUNCTIONAL_SCHEMA_LABELS = {
    GROUP_HIGH: "GROUP_HIGH: token belongs to the thousands-side group",
    GROUP_LOW: "GROUP_LOW: token belongs to the low group",
    CHUNK_PRESENT: "CHUNK_PRESENT: the relevant 3-digit chunk emits at least one word",
    CHUNK_EMPTY: "CHUNK_EMPTY: the relevant 3-digit chunk emits no words",
    CHUNK_HAS_HUNDREDS: "CHUNK_HAS_HUNDREDS: current chunk has a nonzero hundreds slot",
    CHUNK_HAS_TAIL: "CHUNK_HAS_TAIL: current chunk has a nonzero sub-100 tail",
    TAIL_IS_TEN: "TAIL_IS_TEN: current tail is exactly ten",
    TAIL_IS_TEEN: "TAIL_IS_TEEN: current tail uses an eleven-to-nineteen lexical form",
    TAIL_HAS_TENS: "TAIL_HAS_TENS: current tail has a nonzero tens-decade slot",
    TAIL_HAS_UNITS: "TAIL_HAS_UNITS: current tail has a nonzero unit slot",
    VALUE_READ: "VALUE_READ: current token needs a value read from runtime memory",
    DIGIT_ONE: "DIGIT_ONE: digit identity 1",
    DIGIT_TWO: "DIGIT_TWO: digit identity 2",
    DIGIT_THREE: "DIGIT_THREE: digit identity 3",
    DIGIT_FOUR: "DIGIT_FOUR: digit identity 4",
    DIGIT_FIVE: "DIGIT_FIVE: digit identity 5",
    DIGIT_SIX: "DIGIT_SIX: digit identity 6",
    DIGIT_SEVEN: "DIGIT_SEVEN: digit identity 7",
    DIGIT_EIGHT: "DIGIT_EIGHT: digit identity 8",
    DIGIT_NINE: "DIGIT_NINE: digit identity 9",
    FORM_UNIT: "FORM_UNIT: value word uses the unit lexical form",
    FORM_TEN: "FORM_TEN: value word is the atomic word ten",
    FORM_TENS: "FORM_TENS: value word uses the tens-decade lexical form",
    FORM_TEEN: "FORM_TEEN: value word uses the eleven-to-nineteen lexical form",
    FORM_HUNDRED_DIGIT: "FORM_HUNDRED_DIGIT: value word is the digit argument of a hundred construction",
    EMIT_VALUE: "EMIT_VALUE: current instruction emits a value word",
    EMIT_HUNDRED: "EMIT_HUNDRED: current instruction emits the hundred scale word",
    EMIT_THOUSAND: "EMIT_THOUSAND: current instruction emits the thousand scale word",
    EMIT_EOS: "EMIT_EOS: current instruction terminates the sequence",
}

_SEMANTIC_SCHEMA_EDGES = [
    (CHUNK_PRESENT, CHUNK_HAS_TAIL),
    (CHUNK_HAS_TAIL, TAIL_HAS_UNITS),
    (CHUNK_PRESENT, VALUE_READ),
    (VALUE_READ, DIGIT_ONE),
    (CHUNK_PRESENT, EMIT_VALUE),
    (TAIL_HAS_UNITS, FORM_UNIT),
    (VALUE_READ, FORM_UNIT),
    (EMIT_VALUE, FORM_UNIT),
    (GROUP_LOW, EMIT_EOS),
    (VALUE_READ, DIGIT_TWO),
    (VALUE_READ, DIGIT_THREE),
    (VALUE_READ, DIGIT_FOUR),
    (VALUE_READ, DIGIT_FIVE),
    (VALUE_READ, DIGIT_SIX),
    (VALUE_READ, DIGIT_SEVEN),
    (VALUE_READ, DIGIT_EIGHT),
    (VALUE_READ, DIGIT_NINE),
    (CHUNK_HAS_TAIL, TAIL_IS_TEN),
    (TAIL_IS_TEN, FORM_TEN),
    (EMIT_VALUE, FORM_TEN),
    (CHUNK_HAS_TAIL, TAIL_IS_TEEN),
    (TAIL_IS_TEEN, FORM_TEEN),
    (VALUE_READ, FORM_TEEN),
    (EMIT_VALUE, FORM_TEEN),
    (CHUNK_HAS_TAIL, TAIL_HAS_TENS),
    (TAIL_HAS_TENS, FORM_TENS),
    (VALUE_READ, FORM_TENS),
    (EMIT_VALUE, FORM_TENS),
    (CHUNK_PRESENT, CHUNK_HAS_HUNDREDS),
    (CHUNK_HAS_HUNDREDS, FORM_HUNDRED_DIGIT),
    (VALUE_READ, FORM_HUNDRED_DIGIT),
    (EMIT_VALUE, FORM_HUNDRED_DIGIT),
    (CHUNK_HAS_HUNDREDS, EMIT_HUNDRED),
    (GROUP_HIGH, EMIT_THOUSAND),
    (CHUNK_PRESENT, EMIT_THOUSAND),
    (GROUP_LOW, CHUNK_EMPTY),
]

FUNCTIONAL_SCHEMA_EDGES = tuple(dict.fromkeys(_SEMANTIC_SCHEMA_EDGES))

FUNCTIONAL_PROBE_NODES = (
    GROUP_HIGH,
    GROUP_LOW,
    CHUNK_PRESENT,
    CHUNK_EMPTY,
    CHUNK_HAS_HUNDREDS,
    CHUNK_HAS_TAIL,
    TAIL_IS_TEN,
    TAIL_IS_TEEN,
    TAIL_HAS_TENS,
    TAIL_HAS_UNITS,
    VALUE_READ,
    DIGIT_ONE,
    DIGIT_TWO,
    DIGIT_THREE,
    DIGIT_FOUR,
    DIGIT_FIVE,
    DIGIT_SIX,
    DIGIT_SEVEN,
    DIGIT_EIGHT,
    DIGIT_NINE,
    EMIT_VALUE,
    EMIT_HUNDRED,
    EMIT_THOUSAND,
    EMIT_EOS,
    FORM_UNIT,
    FORM_TEN,
    FORM_TENS,
    FORM_TEEN,
    FORM_HUNDRED_DIGIT,
)

_PARENTS_BY_NODE: dict[str, tuple[str, ...]] = {node: () for node in FUNCTIONAL_NODE_ORDER}
for _parent, _child in FUNCTIONAL_SCHEMA_EDGES:
    _PARENTS_BY_NODE[_child] = _PARENTS_BY_NODE[_child] + (_parent,)


@dataclass(frozen=True)
class FunctionalTokenLabel:
    number: int
    token_index: int
    prefix: tuple[str, ...]
    target: str
    owner: str
    active: tuple[str, ...]
    context: str
    subtype: str


@dataclass(frozen=True)
class _FunctionalTokenBuild:
    word: str
    owner: str
    active: tuple[str, ...]
    context: str
    subtype: str


def build_functional_token_labels(n: int) -> list[FunctionalTokenLabel]:
    n = int(n)
    token_builds = _number_tokens(n)
    words = tuple(token.word for token in token_builds)
    expected = tuple(english_number_name(n).split()) + ("EOS",)
    if words != expected:
        raise ValueError(f"functional plan mismatch for {n}: {words} vs {expected}")
    return [
        FunctionalTokenLabel(
            number=n,
            token_index=index,
            prefix=words[:index],
            target=token.word,
            owner=token.owner,
            active=token.active,
            context=token.context,
            subtype=token.subtype,
        )
        for index, token in enumerate(token_builds)
    ]


def _number_tokens(n: int) -> tuple[_FunctionalTokenBuild, ...]:
    if not 1 <= int(n) <= 999_999:
        raise ValueError(f"functional number must be 1..999999, got {n}.")
    if n < 1000:
        tokens = list(_chunk_tokens(n, "number", in_thousands=False))
        tokens.append(_eos_token(low_value=n))
        return tuple(tokens)
    high, low = divmod(n, 1000)
    tokens = list(_chunk_tokens(high, "high", in_thousands=True))
    tokens.append(_thousand_token(high, "high/thousand", "thousand"))
    if low:
        tokens.extend(_chunk_tokens(low, "low", in_thousands=False))
    tokens.append(_eos_token(low_value=low))
    return tuple(tokens)


def _chunk_tokens(n: int, chunk: str, *, in_thousands: bool) -> tuple[_FunctionalTokenBuild, ...]:
    if not 1 <= int(n) <= 999:
        raise ValueError(f"functional chunk value must be 1..999, got {n}.")
    shape_leaves = _chunk_shape_leaves(n)
    if n < 100:
        return _sub100_tokens(n, chunk, "whole", in_thousands=in_thousands, shape_leaves=shape_leaves)
    hundreds, remainder = divmod(n, 100)
    tokens = [
        _digit_token(
            hundreds,
            "hundred_digit",
            f"{chunk}/hundreds/digit",
            "hundred_digit",
            in_thousands=in_thousands,
            shape_leaves=shape_leaves,
        ),
        _hundred_token(f"{chunk}/hundred_scale", in_thousands=in_thousands, shape_leaves=shape_leaves),
    ]
    if remainder:
        tokens.extend(_sub100_tokens(remainder, chunk, "remainder", in_thousands=in_thousands, shape_leaves=shape_leaves))
    return tuple(tokens)


def _sub100_tokens(
    n: int,
    chunk: str,
    slot: str,
    *,
    in_thousands: bool,
    shape_leaves: tuple[str, ...],
) -> tuple[_FunctionalTokenBuild, ...]:
    if not 1 <= int(n) <= 99:
        raise ValueError(f"functional sub-100 value must be 1..99, got {n}.")
    if n <= 9:
        return (
            _digit_token(
                n,
                "unit",
                f"{chunk}/{slot}/unit",
                "digit_unit",
                in_thousands=in_thousands,
                shape_leaves=shape_leaves,
            ),
        )
    if n == 10:
        return (
            _ten_token(
                f"{chunk}/{slot}/ten",
                "ten",
                in_thousands=in_thousands,
                shape_leaves=shape_leaves,
            ),
        )
    if 11 <= n <= 19:
        return (_teen_token(n, f"{chunk}/{slot}/teen", "digit_teen", in_thousands=in_thousands, shape_leaves=shape_leaves),)
    tens, unit = divmod(n, 10)
    tokens = [
        _digit_token(
            tens,
            "tens",
            f"{chunk}/{slot}/tens",
            "digit_tens",
            in_thousands=in_thousands,
            shape_leaves=shape_leaves,
        )
    ]
    if unit:
        tokens.append(
            _digit_token(
                unit,
                "unit",
                f"{chunk}/{slot}/unit",
                "digit_unit",
                in_thousands=in_thousands,
                shape_leaves=shape_leaves,
            )
        )
    return tuple(tokens)


def _digit_token(
    value: int,
    form: str,
    context: str,
    subtype: str,
    *,
    in_thousands: bool,
    shape_leaves: tuple[str, ...],
) -> _FunctionalTokenBuild:
    digit = int(value)
    if not 1 <= digit <= 9:
        raise ValueError(f"digit identity must be 1..9 for {form}, got {value}.")
    if form == "unit":
        word = english_number_name(digit)
        form_node = FORM_UNIT
    elif form == "hundred_digit":
        word = english_number_name(digit)
        form_node = FORM_HUNDRED_DIGIT
    elif form == "tens":
        word = english_number_name(10 * digit)
        form_node = FORM_TENS
    else:
        raise ValueError(f"Unknown digit form: {form!r}")
    active_leaves = _value_leaves(in_thousands, digit=digit, form_node=form_node, shape_leaves=shape_leaves)
    return _token(
        word=word,
        active_leaves=active_leaves,
        context=f"{context};digit={digit};form={form};thousands={int(in_thousands)}",
        subtype=subtype,
    )


def _ten_token(
    context: str,
    subtype: str,
    *,
    in_thousands: bool,
    shape_leaves: tuple[str, ...],
) -> _FunctionalTokenBuild:
    active_leaves = (_group_node(in_thousands), *shape_leaves, FORM_TEN)
    return _token(
        word="ten",
        active_leaves=active_leaves,
        context=f"{context};value=10;form=ten;thousands={int(in_thousands)}",
        subtype=subtype,
    )


def _teen_token(
    value: int,
    context: str,
    subtype: str,
    *,
    in_thousands: bool,
    shape_leaves: tuple[str, ...],
) -> _FunctionalTokenBuild:
    value = int(value)
    if not 11 <= value <= 19:
        raise ValueError(f"teen value must be 11..19, got {value}.")
    digit = value - 10
    active_leaves = _value_leaves(in_thousands, digit=digit, form_node=FORM_TEEN, shape_leaves=shape_leaves)
    return _token(
        word=english_number_name(value),
        active_leaves=active_leaves,
        context=f"{context};value={value};digit={digit};form=teen;thousands={int(in_thousands)}",
        subtype=subtype,
    )


def _hundred_token(context: str, *, in_thousands: bool, shape_leaves: tuple[str, ...]) -> _FunctionalTokenBuild:
    active_leaves = (_group_node(in_thousands), *shape_leaves, EMIT_HUNDRED)
    return _token(
        word="hundred",
        active_leaves=active_leaves,
        context=f"{context};thousands={int(in_thousands)}",
        subtype="hundred",
    )


def _thousand_token(group_value: int, context: str, subtype: str) -> _FunctionalTokenBuild:
    return _token(
        word="thousand",
        active_leaves=(GROUP_HIGH, *_chunk_shape_leaves(group_value), EMIT_THOUSAND),
        context=f"{context};group={int(group_value)}",
        subtype=f"scale_{subtype}",
    )


def _eos_token(*, low_value: int) -> _FunctionalTokenBuild:
    low_value = int(low_value)
    chunk_leaves = (CHUNK_EMPTY,) if low_value == 0 else _chunk_shape_leaves(low_value)
    return _token(
        word="EOS",
        active_leaves=(GROUP_LOW, *chunk_leaves, EMIT_EOS),
        context=f"eos;low={low_value};low_empty={int(low_value == 0)}",
        subtype="eos",
    )


def _digit_node(digit: int) -> str:
    try:
        return DIGIT_NODE_BY_DIGIT[int(digit)]
    except KeyError as exc:
        raise ValueError(f"digit identity must be 1..9, got {digit}.") from exc


def _group_node(in_thousands: bool) -> str:
    return GROUP_HIGH if bool(in_thousands) else GROUP_LOW


def _chunk_shape_leaves(n: int) -> tuple[str, ...]:
    n = int(n)
    if not 1 <= n <= 999:
        raise ValueError(f"chunk shape value must be 1..999, got {n}.")
    hundreds, tail = divmod(n, 100)
    leaves: list[str] = [CHUNK_PRESENT]
    if hundreds:
        leaves.append(CHUNK_HAS_HUNDREDS)
    if tail == 0:
        return tuple(leaves)
    leaves.append(CHUNK_HAS_TAIL)
    if 1 <= tail <= 9:
        leaves.append(TAIL_HAS_UNITS)
    elif tail == 10:
        leaves.append(TAIL_IS_TEN)
    elif 11 <= tail <= 19:
        leaves.append(TAIL_IS_TEEN)
    else:
        tens, unit = divmod(tail, 10)
        if tens:
            leaves.append(TAIL_HAS_TENS)
        if unit:
            leaves.append(TAIL_HAS_UNITS)
    return tuple(leaves)


def _value_leaves(
    in_thousands: bool,
    *,
    digit: int,
    form_node: str,
    shape_leaves: tuple[str, ...],
) -> tuple[str, ...]:
    return (_group_node(in_thousands), *shape_leaves, form_node, _digit_node(int(digit)))


def _token(
    *,
    word: str,
    active_leaves: tuple[str, ...],
    context: str,
    subtype: str,
) -> _FunctionalTokenBuild:
    leaf_set = tuple(dict.fromkeys(active_leaves))
    if not leaf_set:
        raise ValueError(f"functional token {word!r} must activate at least one quantum.")
    for leaf in leaf_set:
        if leaf not in FUNCTIONAL_PROBE_NODES:
            raise ValueError(f"{leaf!r} is not a functional probe node.")
    active = _active_closure(leaf_set)
    owner = leaf_set[0]
    return _FunctionalTokenBuild(
        word=word,
        owner=owner,
        active=active,
        context=context,
        subtype=subtype,
    )


def _active_closure(leaves: tuple[str, ...]) -> tuple[str, ...]:
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            return
        active.add(node)
        for parent in _PARENTS_BY_NODE[node]:
            visit(parent)

    for leaf in leaves:
        visit(leaf)
    return tuple(node for node in FUNCTIONAL_NODE_ORDER if node in active)
