from __future__ import annotations


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

SCALES = [
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
]


def english_number_name(number: int) -> str:
    if number < 0 or number > 999_999_999_999:
        raise ValueError("english_number_name supports 0..999999999999.")
    if number < 1000:
        return _under_1000(number)
    words = []
    remainder = int(number)
    for scale, scale_word in SCALES:
        block, remainder = divmod(remainder, scale)
        if block:
            words.extend([english_number_name(block), scale_word])
    if remainder:
        words.append(_under_1000(remainder))
    return " ".join(words)


def _under_1000(number: int) -> str:
    if number < 20:
        return ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        words = [TENS[tens * 10]]
        if ones:
            words.append(ONES[ones])
        return " ".join(words)
    hundreds, remainder = divmod(number, 100)
    words = [ONES[hundreds], "hundred"]
    if remainder:
        words.append(_under_1000(remainder))
    return " ".join(words)


def number_name(number: int, *, language: str) -> str:
    if language.lower() != "english":
        raise ValueError("Only english is currently supported.")
    return english_number_name(int(number))
