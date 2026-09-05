"""Incremental detokenization must produce exactly what a full re-decode would.

The stub tokenizer here is byte-level, like the BPE tokenizers the engine
actually runs: each token id is one byte, and decoding is a UTF-8 decode with
`errors="replace"`. That is what makes it possible to split a multi-byte
character across two tokens, which is the case naive concatenation gets wrong.
"""

from __future__ import annotations

import pytest

from liteinfer.tokenizer import IncrementalDetokenizer


class ByteTokenizer:
    """Decodes each id as one byte. Incomplete UTF-8 becomes U+FFFD, as HF's does."""

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


def _stream(token_ids: list[int]) -> str:
    """Feed the tokens one at a time, as the engine does."""
    tokenizer, detokenizer = ByteTokenizer(), IncrementalDetokenizer()
    for i in range(1, len(token_ids) + 1):
        detokenizer.update(tokenizer, token_ids[:i])
    return detokenizer.text


def _ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "café",           # 2-byte character
        "日本語テキスト",   # 3-byte characters
        "emoji 🙂 tail",   # 4-byte character
        "🙂🙂🙂",           # nothing but multi-byte
        "",
    ],
    ids=["ascii", "two-byte", "three-byte", "four-byte", "all-multi-byte", "empty"],
)
def test_streaming_matches_a_full_decode(text: str) -> None:
    assert _stream(_ids(text)) == ByteTokenizer().decode(_ids(text))


def test_a_character_split_across_tokens_is_not_emitted_early() -> None:
    """Half of a multi-byte character is not text yet, and must not reach the caller."""
    tokenizer, detokenizer = ByteTokenizer(), IncrementalDetokenizer()
    first_byte_of_smiley = _ids("🙂")[:1]

    assert detokenizer.update(tokenizer, first_byte_of_smiley) == ""


def test_the_character_appears_once_its_last_byte_arrives() -> None:
    tokenizer, detokenizer = ByteTokenizer(), IncrementalDetokenizer()
    smiley = _ids("🙂")
    for i in range(1, len(smiley)):
        detokenizer.update(tokenizer, smiley[:i])

    assert detokenizer.update(tokenizer, smiley) == "🙂"


def test_text_is_never_rebuilt_from_the_whole_prefix() -> None:
    """The point of the exercise: the decode window stays small as output grows."""
    tokenizer, detokenizer = ByteTokenizer(), IncrementalDetokenizer()
    token_ids = _ids("a" * 500)
    for i in range(1, len(token_ids) + 1):
        detokenizer.update(tokenizer, token_ids[:i])

    assert len(token_ids) - detokenizer.prefix_offset <= 2


def test_offsets_start_at_the_beginning() -> None:
    assert (IncrementalDetokenizer().prefix_offset, IncrementalDetokenizer().read_offset) == (0, 0)
