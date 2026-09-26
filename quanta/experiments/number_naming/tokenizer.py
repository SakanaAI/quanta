from __future__ import annotations

from dataclasses import dataclass


PAD = "[PAD]"
BOS = "[BOS]"
SEP = "[SEP]"
EOS = "[EOS]"
UNK = "[UNK]"
SPECIAL_TOKENS = [PAD, BOS, SEP, EOS, UNK]
DIGIT_TOKENS = [f"<D{digit}>" for digit in range(10)]


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    target_ids: list[int]
    number: int
    text: str


class NumberNamingTokenizer:
    def __init__(self, word_tokens: list[str]):
        tokens = SPECIAL_TOKENS + DIGIT_TOKENS + sorted(set(word_tokens))
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.sep_id = self.token_to_id[SEP]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(
        self,
        number: int,
        text: str,
        *,
        max_seq_len: int,
    ) -> EncodedExample:
        digits = str(int(number))
        digit_ids = [self.token_to_id[f"<D{digit}>"] for digit in digits]
        word_ids = [self.token_to_id.get(word, self.unk_id) for word in text.split()]
        sequence = [self.bos_id] + digit_ids + [self.sep_id] + word_ids + [self.eos_id]
        if len(sequence) > max_seq_len:
            raise ValueError(f"Encoded example exceeds max_seq_len={max_seq_len}: {number} -> {text!r}")
        labels = [-100] * len(sequence)
        sep_index = sequence.index(self.sep_id)
        labels[sep_index + 1 :] = word_ids + [self.eos_id]
        return EncodedExample(
            input_ids=sequence,
            labels=labels,
            target_ids=word_ids + [self.eos_id],
            number=int(number),
            text=text,
        )

    def decode_words(self, token_ids: list[int]) -> list[str]:
        output = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token == EOS:
                break
            if token not in SPECIAL_TOKENS and token not in DIGIT_TOKENS:
                output.append(token)
        return output
