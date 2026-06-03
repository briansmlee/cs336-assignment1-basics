"""Byte-Pair Encoding (BPE) tokenizer.

This module contains the two deliverables for the BPE section of the
assignment:

  * ``train_bpe`` -- train a byte-level BPE tokenizer on a text corpus and
    return the learned vocabulary and merges.
  * ``Tokenizer`` -- a class that uses a trained vocabulary and merge list to
    encode text into token IDs and decode token IDs back into text.

These are exposed to the test suite through ``tests/adapters.py``
(``run_train_bpe`` and ``get_tokenizer``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the corpus at ``input_path``.

    The training procedure:
      1. Initialize the vocabulary with the 256 byte values and the given
         special tokens.
      2. Pre-tokenize the corpus (splitting on special tokens first, then on
         the GPT-2 regex pattern), counting the frequency of each pre-token.
      3. Repeatedly merge the most frequent adjacent byte pair (breaking ties
         by preferring the lexicographically greater pair) until the vocabulary
         reaches ``vocab_size``.

    Args:
        input_path: Path to a UTF-8 text file to train on.
        vocab_size: Target vocabulary size, including the initial 256 bytes and
            all special tokens.
        special_tokens: Strings that must each map to a single token and never
            be split or merged with surrounding text.

    Returns:
        vocab: Mapping from token ID to its byte sequence.
        merges: Ordered list of merged byte-pairs ``(token1, token2)``, in the
            order the merges were created.
    """
    raise NotImplementedError


class Tokenizer:
    """A byte-level BPE tokenizer.

    Given a vocabulary, an ordered list of merges, and an optional list of
    special tokens, this class encodes text into a sequence of token IDs and
    decodes token IDs back into text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Construct a tokenizer from a vocabulary, merge list, and special tokens.

        Args:
            vocab: Mapping from token ID to its byte sequence.
            merges: Ordered list of merged byte-pairs, in the order they were
                created during training.
            special_tokens: Strings that should each be encoded as a single
                token and never split.
        """
        raise NotImplementedError

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a ``Tokenizer`` from serialized vocabulary and merges files.

        The files are expected to be in the same format produced by
        ``train_bpe`` (e.g. the GPT-2 vocab JSON and merges TXT format).

        Args:
            vocab_filepath: Path to the serialized vocabulary.
            merges_filepath: Path to the serialized merges.
            special_tokens: Strings that should each be encoded as a single
                token and never split.

        Returns:
            A constructed ``Tokenizer`` instance.
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode an input string into a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            The token IDs representing ``text``.
        """
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings into token IDs.

        This is useful for memory-efficient tokenization of large files: given
        a file handle (an iterable of lines), it yields token IDs one at a time
        without loading the entire corpus into memory.

        Args:
            iterable: An iterable of strings (e.g. an open file handle).

        Yields:
            Token IDs, in order.
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text.

        Args:
            ids: The token IDs to decode.

        Returns:
            The decoded string. Bytes that do not form valid UTF-8 should be
            replaced with the Unicode replacement character.
        """
        raise NotImplementedError
