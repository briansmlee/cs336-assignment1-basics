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
import json
from collections.abc import Iterable, Iterator
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def to_symbols(word):
    utf8 = word.encode("utf-8")
    return tuple(bytes([byte]) for byte in utf8)

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

    # drops the special tokens.
    special_token_pattern = "|".join([re.escape(token) for token in special_tokens])

    word_cnt = Counter()
    with open(input_path) as f:
        text = f.read()
        for document in re.split(special_token_pattern, text):
            for word in re.findall(PAT, document):
                word_cnt[to_symbols(word)] += 1

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
        special_tokens = special_tokens or []
        special_token_pattern = "|".join([re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)])
        # parenthesis keeps the special tokens.
        self.special_token_pattern = f"({special_token_pattern})"
        self.special_tokens = set(special_tokens)
        self.vocab = vocab
        self.inverted_vocab = {token : i for i, token in vocab.items()}
        self.merges = {merge : i for i, merge in enumerate(merges)}

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
        # vocab = json.load(vocab_filepath)
        # merges = 
        # return cls(vocab, , special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input string into a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            The token IDs representing ``text``.
        """
        def apply_merges(cur):
            while True:
                merge = None
                for i in range(len(cur) - 1):
                    pair = (cur[i], cur[i+1])
                    if pair in self.merges:
                        cand = (self.merges[pair], pair)
                        merge = min(cand, merge) if merge else cand

                if not merge:
                    return cur

                _, merge_pair = merge
                merged = b"".join(merge_pair)
                nxt = []
                i = 0
                while i < len(cur) - 1:
                    pair = (cur[i], cur[i+1])
                    if pair == merge_pair:
                        nxt.append(merged)
                        i += 2
                    else:
                        nxt.append(cur[i])
                        i += 1
                
                if i == len(cur) - 1:
                    nxt.append(cur[i])
                
                cur = tuple(nxt)

        ids = []
        for document in re.split(self.special_token_pattern, text) if self.special_tokens else [text]:
            if document in self.special_tokens:
                utf8 = document.encode("utf-8")    
                ids.append(self.inverted_vocab[utf8])
            else:
                for word in re.findall(PAT, document):
                    for token in apply_merges(to_symbols(word)):
                        ids.append(self.inverted_vocab[token])

        return ids


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
        for it in iterable:
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text.

        Args:
            ids: The token IDs to decode.

        Returns:
            The decoded string. Bytes that do not form valid UTF-8 should be
            replaced with the Unicode replacement character.
        """
        bytes = b"".join(self.vocab[id] for id in ids)
        return bytes.decode("utf-8", errors="replace")
