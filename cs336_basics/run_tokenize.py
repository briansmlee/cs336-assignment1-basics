from cs336_basics.bpe import *
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Tokenizes documents",
    )
    parser.add_argument(
        "--vocab-filepath",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--merges-filepath",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|endoftext|>"],
    )
    parser.add_argument(
        "--text-filepath",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-filepath",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath=args.merges_filepath,
        special_tokens=args.special_tokens,
    )

    with open(args.text_filepath, "r") as text_file:
        tokens = tokenizer.encode_iterable(text_file)
        arr = np.fromiter(tokens, dtype=np.uint16)
        np.save(args.output_filepath, arr)


if __name__ == "__main__":
    main()
