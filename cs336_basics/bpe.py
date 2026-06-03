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
import regex as re
from collections.abc import Iterable, Iterator
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the corpus at ``input_path``.

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

    """
    my notes
    
    word is list of tokens. token is one or more utf-8 bytes.

    - read file as unicode text
    - split (and drop) by special tokens
    - pre-tokenize with regex to word -> cnt
    - encode word from unicode to utf-8
    - merge top token pair, then update each word, until given vocab size
    """

    def update(word):
        updated = []
        i = 0
        while i < len(word) - 1:
            if word[i:i+2] == top_token_pair:
                updated.append(new_token)
                i += 2
            else:
                updated.append(word[i])
                i += 1
        if i == len(word) - 1:
            updated.append(word[i])
        
        return tuple(updated)

    split_re = "|".join([re.escape(token) for token in special_tokens])

    word_cnt = Counter()
    with open(input_path) as f:
        text = f.read()
        for split in re.split(split_re, text):
            for word in re.findall(PAT, split):
                utf8 = word.encode("utf-8")
                word_cnt[tuple(bytes([byte]) for byte in utf8)] += 1

    vocab = [bytes([i]) for i in range(256)] + [token.encode("utf-8") for token in special_tokens]
    merges = []

    while len(vocab) < vocab_size:
        token_pair_cnt = Counter()
        for word, cnt in word_cnt.items():
            for i in range(len(word) - 1):
                token_pair_cnt[(word[i], word[i+1])] += cnt
        
        _, top_token_pair = max((cnt, token_pair) for token_pair, cnt in token_pair_cnt.items())
        
        merges.append(top_token_pair)
        new_token = b"".join(top_token_pair)
        vocab.append(new_token)

        word_cnt = Counter({update(word): cnt for word, cnt in word_cnt.items()})

    inverted_vocab = { i : token for i, token in enumerate(vocab)}
    return inverted_vocab, merges


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
