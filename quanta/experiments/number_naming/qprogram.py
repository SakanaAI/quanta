from __future__ import annotations

from enum import Enum

from quanta.qprogram import FunctionalProgram, PredictiveState, QRegistry, when


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


class TailKind(Enum):
    EMPTY = "empty"
    UNIT = "unit"
    TEN = "ten"
    TEEN = "teen"
    TENS = "tens"
    TENS_UNIT = "tens_unit"


class BoundaryAction(Enum):
    CONTENT = "content"
    THOUSAND = "thousand"
    EOS = "eos"


class ContentSlot(Enum):
    HUNDRED_DIGIT = "hundred_digit"
    HUNDRED_WORD = "hundred_word"
    TAIL_FIRST = "tail_first"
    TAIL_SECOND = "tail_second"


class LexicalForm(Enum):
    UNIT = "unit"
    TEN = "ten"
    TEEN = "teen"
    TENS = "tens"


ChunkContext = tuple[Group, int]


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

    @registry.primitive(cost=14)
    def tail_kind(tens_digit: int, unit_digit: int) -> TailKind:
        if tens_digit == 0 and unit_digit == 0:
            return TailKind.EMPTY
        if tens_digit == 0:
            return TailKind.UNIT
        if tens_digit == 1 and unit_digit == 0:
            return TailKind.TEN
        if tens_digit == 1:
            return TailKind.TEEN
        if unit_digit == 0:
            return TailKind.TENS
        return TailKind.TENS_UNIT

    @registry.primitive(cost=4)
    def tail_token_count(kind: TailKind) -> int:
        if kind == TailKind.EMPTY:
            return 0
        if kind == TailKind.TENS_UNIT:
            return 2
        return 1

    @registry.primitive(cost=8)
    def chunk_token_count(chunk: int) -> int:
        hundreds_tokens = 2 if chunk >= 100 else 0
        tail = chunk % 100 if chunk >= 100 else chunk
        return hundreds_tokens + tail_token_count(tail_kind(tail // 10, tail % 10))

    @registry.primitive(cost=2)
    def high_content_length(digits: tuple[int, ...]) -> int:
        return chunk_token_count(high_chunk(digits))

    @registry.primitive(cost=4)
    def control_group(high_length: int, position: int) -> Group:
        if high_length > 0 and position <= high_length:
            return Group.HIGH
        return Group.LOW

    @registry.primitive(cost=8)
    def select_chunk(digits: tuple[int, ...], group: Group) -> ChunkContext:
        if group == Group.HIGH:
            return (group, high_chunk(digits))
        return (group, low_chunk(digits))

    @registry.primitive(cost=6)
    def group_progress(group: Group, high_length: int, position: int) -> int:
        if group == Group.HIGH:
            return position
        scale_tokens = 1 if high_length > 0 else 0
        return position - high_length - scale_tokens

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

    @registry.primitive(cost=5)
    def selected_chunk_length(hundreds_digit: int, kind: TailKind) -> int:
        hundreds_tokens = 2 if hundreds_digit > 0 else 0
        return hundreds_tokens + tail_token_count(kind)

    @registry.primitive(cost=6)
    def boundary_action(
        group: Group, local_position: int, content_length: int
    ) -> BoundaryAction:
        if local_position < content_length:
            return BoundaryAction.CONTENT
        if group == Group.HIGH:
            return BoundaryAction.THOUSAND
        return BoundaryAction.EOS

    @registry.primitive(cost=11)
    def content_slot(local_position: int, hundreds_digit: int) -> ContentSlot:
        if hundreds_digit > 0:
            if local_position == 0:
                return ContentSlot.HUNDRED_DIGIT
            if local_position == 1:
                return ContentSlot.HUNDRED_WORD
            if local_position == 2:
                return ContentSlot.TAIL_FIRST
            return ContentSlot.TAIL_SECOND
        if local_position == 0:
            return ContentSlot.TAIL_FIRST
        return ContentSlot.TAIL_SECOND

    @registry.primitive(cost=12)
    def lexical_form(slot: ContentSlot, kind: TailKind) -> LexicalForm:
        if slot == ContentSlot.HUNDRED_DIGIT or slot == ContentSlot.TAIL_SECOND:
            return LexicalForm.UNIT
        if kind == TailKind.UNIT:
            return LexicalForm.UNIT
        if kind == TailKind.TEN:
            return LexicalForm.TEN
        if kind == TailKind.TEEN:
            return LexicalForm.TEEN
        return LexicalForm.TENS

    @registry.primitive(cost=8)
    def lexical_value(
        slot: ContentSlot,
        hundreds_digit: int,
        tens_digit: int,
        unit_digit: int,
    ) -> int:
        if slot == ContentSlot.HUNDRED_DIGIT:
            return hundreds_digit
        if slot == ContentSlot.TAIL_SECOND:
            return unit_digit
        if tens_digit >= 2:
            return tens_digit
        return tens_digit * 10 + unit_digit

    @registry.primitive(cost=2, table_cardinality=9)
    def lexical_unit(value: int) -> str:
        return UNIT_WORDS[value - 1]

    @registry.primitive(cost=2, table_cardinality=9)
    def lexical_teen(value: int) -> str:
        return TEEN_WORDS[value - 11]

    @registry.primitive(cost=2, table_cardinality=8)
    def lexical_tens(digit: int) -> str:
        return TENS_WORDS[digit - 2]

    @registry.quantum("HIGH_CONTENT_LENGTH", source_reads=("digits",), output_cardinality=5)
    def q_high_content_length(state: PredictiveState) -> int:
        return high_content_length(state.read("digits"))

    @registry.quantum("CONTROL_GROUP", source_reads=("position",), output_cardinality=2)
    def q_control_group(state: PredictiveState, high_length: int) -> Group:
        return control_group(high_length, state.read("position"))

    @registry.quantum("SELECT_CHUNK", source_reads=("digits",), output_cardinality=2000)
    def q_select_chunk(state: PredictiveState, group: Group) -> ChunkContext:
        return select_chunk(state.read("digits"), group)

    @registry.quantum("GROUP_PROGRESS", source_reads=("position",), output_cardinality=5)
    def q_group_progress(
        state: PredictiveState, group: Group, high_length: int
    ) -> int:
        return group_progress(group, high_length, state.read("position"))

    @registry.quantum("HUNDREDS_COMPONENT", output_cardinality=10)
    def q_hundreds_component(context: ChunkContext) -> int:
        return hundreds_component(context)

    @registry.quantum("TAIL_TENS_DIGIT", output_cardinality=10)
    def q_tail_tens_digit(context: ChunkContext) -> int:
        return tail_tens_digit(context)

    @registry.quantum("TAIL_UNIT_DIGIT", output_cardinality=10)
    def q_tail_unit_digit(context: ChunkContext) -> int:
        return tail_unit_digit(context)

    @registry.quantum("TAIL_KIND", output_cardinality=6)
    def q_tail_kind(tens_digit: int, unit_digit: int) -> TailKind:
        return tail_kind(tens_digit, unit_digit)

    @registry.quantum("CHUNK_LENGTH", output_cardinality=5)
    def q_chunk_length(hundreds_digit: int, kind: TailKind) -> int:
        return selected_chunk_length(hundreds_digit, kind)

    @registry.quantum("BOUNDARY_ACTION", output_cardinality=3)
    def q_boundary_action(
        group: Group, local_position: int, content_length: int
    ) -> BoundaryAction:
        return boundary_action(group, local_position, content_length)

    @registry.quantum("CONTENT_SLOT", output_cardinality=4)
    def q_content_slot(local_position: int, hundreds_digit: int) -> ContentSlot:
        return content_slot(local_position, hundreds_digit)

    @registry.quantum("LEXICAL_FORM", output_cardinality=4)
    def q_lexical_form(slot: ContentSlot, kind: TailKind) -> LexicalForm:
        return lexical_form(slot, kind)

    @registry.quantum("LEXICAL_VALUE", output_cardinality=18)
    def q_lexical_value(
        slot: ContentSlot,
        hundreds_digit: int,
        tens_digit: int,
        unit_digit: int,
    ) -> int:
        return lexical_value(slot, hundreds_digit, tens_digit, unit_digit)

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
    def q_emit_hundred(slot: ContentSlot) -> str:
        return "hundred"

    @registry.quantum("EMIT_TEN", output_cardinality=1)
    def q_emit_ten(form: LexicalForm) -> str:
        return "ten"

    @registry.quantum("EMIT_THOUSAND", output_cardinality=1)
    def q_emit_thousand(boundary: BoundaryAction) -> str:
        return "thousand"

    @registry.quantum("EMIT_EOS", output_cardinality=1)
    def q_emit_eos(boundary: BoundaryAction) -> str:
        return "[EOS]"

    def predict(state: PredictiveState) -> str:
        high_length = q_high_content_length(state)
        group = q_control_group(state, high_length)
        chunk = q_select_chunk(state, group)
        local_position = q_group_progress(state, group, high_length)
        hundreds = q_hundreds_component(chunk)
        tens = q_tail_tens_digit(chunk)
        unit = q_tail_unit_digit(chunk)
        kind = q_tail_kind(tens, unit)
        content_length = q_chunk_length(hundreds, kind)
        boundary = q_boundary_action(group, local_position, content_length)

        with when(boundary, BoundaryAction.THOUSAND) as at_thousand:
            if at_thousand:
                return q_emit_thousand(boundary)
        with when(boundary, BoundaryAction.EOS) as at_eos:
            if at_eos:
                return q_emit_eos(boundary)
        with when(boundary, BoundaryAction.CONTENT) as at_content:
            if at_content:
                slot = q_content_slot(local_position, hundreds)
                with when(slot, ContentSlot.HUNDRED_WORD) as at_hundred_word:
                    if at_hundred_word:
                        return q_emit_hundred(slot)
                form = q_lexical_form(slot, kind)
                with when(form, LexicalForm.TEN) as at_ten:
                    if at_ten:
                        return q_emit_ten(form)
                value = q_lexical_value(slot, hundreds, tens, unit)
                with when(form, LexicalForm.UNIT) as at_unit:
                    if at_unit:
                        return q_lex_unit(value)
                with when(form, LexicalForm.TEEN) as at_teen:
                    if at_teen:
                        return q_lex_teen(value)
                with when(form, LexicalForm.TENS) as at_tens:
                    if at_tens:
                        return q_lex_tens(value)
        return q_emit_eos(boundary)

    return FunctionalProgram(registry, predict)


__all__ = [
    "BoundaryAction",
    "ContentSlot",
    "Group",
    "LexicalForm",
    "TailKind",
    "build_number_naming_program",
]
