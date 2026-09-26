"""Compact exact NumberNaming Q-program."""

from __future__ import annotations

from enum import Enum

from quanta.qprogram import (
    FunctionalProgram,
    PredictiveState,
    QRegistry,
    when_item,
)


UNIT_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
TEEN_WORDS = (
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS_WORDS = (
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


class Group(Enum):
    HIGH = "high"
    LOW = "low"


class Presence(Enum):
    PRESENT = "present"
    EMPTY = "empty"


class TailForm(Enum):
    EMPTY = "empty"
    SMALL = "small"
    TENS = "tens"
    TENS_UNIT = "tens_unit"


class EmissionPhase(Enum):
    HUNDRED_DIGIT = "hundred_digit"
    HUNDRED = "hundred"
    SMALL_UNIT = "small_unit"
    TEN = "ten"
    TEEN = "teen"
    TENS = "tens"
    UNIT = "unit"
    THOUSAND = "thousand"
    EOS = "eos"


ChunkContext = tuple[Group, int]
GroupProgress = tuple[Group, int]
TailContext = tuple[int, int, TailForm]
EmissionContext = tuple[EmissionPhase, int]


def build_number_naming_program(*, unit_complexity: int = 80) -> FunctionalProgram:
    """Build the exact next-token Q-program for positive integers below one million."""
    registry = QRegistry("number_naming", unit_complexity=int(unit_complexity))

    @registry.primitive(cost=18)
    def high_chunk(digits: tuple[int, ...]) -> int:
        if len(digits) <= 3:
            return 0
        if len(digits) == 4:
            return digits[0]
        if len(digits) == 5:
            return digits[0] * 10 + digits[1]
        return digits[0] * 100 + digits[1] * 10 + digits[2]

    @registry.primitive(cost=22)
    def low_chunk(digits: tuple[int, ...]) -> int:
        if len(digits) == 1:
            return digits[-1]
        if len(digits) == 2:
            return digits[-2] * 10 + digits[-1]
        return digits[-3] * 100 + digits[-2] * 10 + digits[-1]

    @registry.primitive(cost=8)
    def tail_form(tens_digit: int, unit_digit: int) -> TailForm:
        if tens_digit == 0 and unit_digit == 0:
            return TailForm.EMPTY
        if tens_digit < 2:
            return TailForm.SMALL
        if unit_digit == 0:
            return TailForm.TENS
        return TailForm.TENS_UNIT

    @registry.primitive(cost=4)
    def tail_token_count(form: TailForm) -> int:
        if form == TailForm.EMPTY:
            return 0
        if form == TailForm.TENS_UNIT:
            return 2
        return 1

    @registry.primitive(cost=8)
    def chunk_token_count(chunk: int) -> int:
        hundreds_count = 2 if chunk >= 100 else 0
        tail = chunk % 100 if chunk >= 100 else chunk
        return hundreds_count + tail_token_count(tail_form(tail // 10, tail % 10))

    @registry.primitive(cost=5)
    def control_group(digits: tuple[int, ...], position: int) -> Group:
        high = high_chunk(digits)
        high_tokens = chunk_token_count(high)
        if high > 0 and position <= high_tokens:
            return Group.HIGH
        return Group.LOW

    @registry.primitive(cost=8)
    def select_chunk(digits: tuple[int, ...], group: Group) -> ChunkContext:
        if group == Group.HIGH:
            return (group, high_chunk(digits))
        return (group, low_chunk(digits))

    @registry.primitive(cost=11)
    def group_progress(digits: tuple[int, ...], position: int, group: Group) -> GroupProgress:
        if group == Group.HIGH:
            return (group, position)
        high = high_chunk(digits)
        scale_tokens = 1 if high > 0 else 0
        low_position = position - chunk_token_count(high) - scale_tokens
        return (group, low_position)

    @registry.primitive(cost=3)
    def chunk_status(context: ChunkContext) -> Presence:
        return Presence.PRESENT if context[1] > 0 else Presence.EMPTY

    @registry.primitive(cost=5)
    def hundreds_component(context: ChunkContext) -> int:
        return context[1] // 100 if context[1] >= 100 else 0

    @registry.primitive(cost=5)
    def tail_tens_digit(context: ChunkContext) -> int:
        chunk = context[1]
        tail = chunk % 100 if chunk >= 100 else chunk
        return tail // 10

    @registry.primitive(cost=2)
    def tail_unit_digit(context: ChunkContext) -> int:
        return context[1] % 10

    @registry.primitive(cost=60)
    def emission_context(
        progress: GroupProgress,
        chunk_presence: Presence,
        hundreds: int,
        tail_context: TailContext,
    ) -> EmissionContext:
        group = progress[0]
        local_position = progress[1]
        tens_digit = tail_context[0]
        unit_digit = tail_context[1]
        form = tail_context[2]
        tail = tens_digit * 10 + unit_digit
        if group == Group.LOW and chunk_presence == Presence.EMPTY:
            return (EmissionPhase.EOS, 0)
        hundreds_tokens = 2 if hundreds > 0 else 0
        content_tokens = hundreds_tokens + tail_token_count(form)
        if group == Group.HIGH and local_position == content_tokens:
            return (EmissionPhase.THOUSAND, 0)
        if group == Group.LOW and local_position == content_tokens:
            return (EmissionPhase.EOS, 0)
        tail_offset = 2 if hundreds > 0 else 0
        if hundreds > 0:
            if local_position == 0:
                return (EmissionPhase.HUNDRED_DIGIT, hundreds)
            if local_position == 1:
                return (EmissionPhase.HUNDRED, 0)
        tail_position = local_position - tail_offset
        if form == TailForm.SMALL:
            if tail < 10:
                return (EmissionPhase.SMALL_UNIT, tail)
            if tail == 10:
                return (EmissionPhase.TEN, 0)
            return (EmissionPhase.TEEN, tail)
        if form == TailForm.TENS and tail_position == 0:
            return (EmissionPhase.TENS, tail // 10)
        if tail_position == 0:
            return (EmissionPhase.TENS, tail // 10)
        return (EmissionPhase.UNIT, tail % 10)

    @registry.primitive(cost=1)
    def select_lexical_value(emission: EmissionContext) -> int:
        return emission[1]

    @registry.primitive(cost=2, table_cardinality=9)
    def lexical_unit(value: int) -> str:
        return UNIT_WORDS[value - 1]

    @registry.primitive(cost=2, table_cardinality=9)
    def lexical_teen(value: int) -> str:
        return TEEN_WORDS[value - 11]

    @registry.primitive(cost=2, table_cardinality=8)
    def lexical_tens(digit: int) -> str:
        return TENS_WORDS[digit - 2]

    @registry.quantum("CONTROL_GROUP", source_reads=("digits", "position"), output_cardinality=2)
    def q_control_group(state: PredictiveState) -> Group:
        digits = state.read("digits")
        position = state.read("position")
        return control_group(digits, position)

    @registry.quantum("SELECT_CHUNK", source_reads=("digits",), output_cardinality=2000)
    def q_select_chunk(state: PredictiveState, group: Group) -> ChunkContext:
        digits = state.read("digits")
        return select_chunk(digits, group)

    @registry.quantum("GROUP_PROGRESS", source_reads=("digits", "position"), output_cardinality=14)
    def q_group_progress(state: PredictiveState, group: Group) -> GroupProgress:
        digits = state.read("digits")
        position = state.read("position")
        return group_progress(digits, position, group)

    @registry.quantum("CHUNK_STATUS", output_cardinality=2)
    def q_chunk_status(context: ChunkContext) -> Presence:
        return chunk_status(context)

    @registry.quantum("HUNDREDS_COMPONENT", output_cardinality=10)
    def q_hundreds_component(context: ChunkContext) -> int:
        return hundreds_component(context)

    @registry.quantum("TAIL_TENS_DIGIT", output_cardinality=10)
    def q_tail_tens_digit(context: ChunkContext) -> int:
        return tail_tens_digit(context)

    @registry.quantum("TAIL_UNIT_DIGIT", output_cardinality=10)
    def q_tail_unit_digit(context: ChunkContext) -> int:
        return tail_unit_digit(context)

    @registry.quantum("TAIL_FORM", output_cardinality=100)
    def q_tail_form(tens_digit: int, unit_digit: int) -> TailContext:
        return (tens_digit, unit_digit, tail_form(tens_digit, unit_digit))

    @registry.quantum("EMISSION_PHASE", output_cardinality=48)
    def q_emission_phase(
        progress: GroupProgress,
        chunk_presence: Presence,
        hundreds: int,
        tail_context: TailContext,
    ) -> EmissionContext:
        return emission_context(progress, chunk_presence, hundreds, tail_context)

    @registry.quantum("SELECT_LEXICAL_VALUE", output_cardinality=18)
    def q_select_lexical_value(emission: EmissionContext) -> int:
        return select_lexical_value(emission)

    @registry.quantum("LEX_UNIT", output_cardinality=9)
    def q_lex_unit(value: int) -> str:
        return lexical_unit(value)

    @registry.quantum("LEX_TEEN", output_cardinality=9)
    def q_lex_teen(value: int) -> str:
        return lexical_teen(value)

    @registry.quantum("LEX_TENS", output_cardinality=8)
    def q_lex_tens(digit: int) -> str:
        return lexical_tens(digit)

    @registry.quantum("EMIT_HUNDRED", output_cardinality=1)
    def q_emit_hundred(emission: EmissionContext) -> str:
        return "hundred"

    @registry.quantum("EMIT_TEN", output_cardinality=1)
    def q_emit_ten(emission: EmissionContext) -> str:
        return "ten"

    @registry.quantum("EMIT_THOUSAND", output_cardinality=1)
    def q_emit_thousand(emission: EmissionContext) -> str:
        return "thousand"

    @registry.quantum("EMIT_EOS", output_cardinality=1)
    def q_emit_eos(emission: EmissionContext) -> str:
        return "[EOS]"

    def predict(state: PredictiveState) -> str:
        group = q_control_group(state)
        chunk = q_select_chunk(state, group)
        progress = q_group_progress(state, group)
        chunk_presence = q_chunk_status(chunk)
        hundreds = q_hundreds_component(chunk)
        tens_digit = q_tail_tens_digit(chunk)
        unit_digit = q_tail_unit_digit(chunk)
        tail_context = q_tail_form(tens_digit, unit_digit)
        emission = q_emission_phase(progress, chunk_presence, hundreds, tail_context)
        with when_item(emission, 0, EmissionPhase.HUNDRED_DIGIT) as active_hundred_digit:
            if active_hundred_digit:
                value = q_select_lexical_value(emission)
                return q_lex_unit(value)
        with when_item(emission, 0, EmissionPhase.SMALL_UNIT) as active_small_unit:
            if active_small_unit:
                value = q_select_lexical_value(emission)
                return q_lex_unit(value)
        with when_item(emission, 0, EmissionPhase.TEN) as active_ten:
            if active_ten:
                return q_emit_ten(emission)
        with when_item(emission, 0, EmissionPhase.TEEN) as active_teen:
            if active_teen:
                value = q_select_lexical_value(emission)
                return q_lex_teen(value)
        with when_item(emission, 0, EmissionPhase.UNIT) as active_unit:
            if active_unit:
                value = q_select_lexical_value(emission)
                return q_lex_unit(value)
        with when_item(emission, 0, EmissionPhase.TENS) as active_tens:
            if active_tens:
                value = q_select_lexical_value(emission)
                return q_lex_tens(value)
        with when_item(emission, 0, EmissionPhase.HUNDRED) as active_hundred:
            if active_hundred:
                return q_emit_hundred(emission)
        with when_item(emission, 0, EmissionPhase.THOUSAND) as active_thousand:
            if active_thousand:
                return q_emit_thousand(emission)
        with when_item(emission, 0, EmissionPhase.EOS) as active_eos:
            if active_eos:
                return q_emit_eos(emission)
        return q_emit_eos(emission)

    return FunctionalProgram(registry, predict)


__all__ = [
    "EmissionPhase",
    "Group",
    "Presence",
    "TailForm",
    "build_number_naming_program",
]
