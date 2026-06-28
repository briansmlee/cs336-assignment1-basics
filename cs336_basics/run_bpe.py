from cs336_basics.bpe import train_bpe, save_bpe
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to the dataset text file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="max vocabulary size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory/prefix to write vocab+merges",
    )
    args = parser.parse_args()

    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        ["<|endoftext|>"],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_bpe(
        vocab,
        merges,
        args.output / "vocab.json",
        args.output / "merges.json",
    )


if __name__ == "__main__":
    main()
