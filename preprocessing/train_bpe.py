"""Utility script for training a byte-pair tokenizer."""

from argparse import ArgumentParser
import sys
import os

from model.tokenizer import BytePairTokenizer
from sentinel import panic


def main():
    """Train a byte-pair tokenizer on the provided dataset."""

    parser = ArgumentParser()
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outpath', required=True)
    parser.add_argument('-m', '--merges', required=True, type=int)
    parser.add_argument('-n', '--mincount', required=True, type=int)
    args = parser.parse_args()
    outpath = args.outpath
    inpath = args.inpath
    merges = args.merges
    mincount = args.mincount

    try:
        filepaths = [path.strip() for path in open(inpath).readlines()]
    except FileNotFoundError:
        panic(f"File list not found: {inpath}")
    try:
        tokenizer = BytePairTokenizer.train_bpe(filepaths, mincount, merges)
    except Exception as exc:
        panic(f"Tokenizer training failed: {exc}")
    os.makedirs(outpath, exist_ok=True)
    try:
        tokenizer.save(f'{outpath}')
    except Exception as exc:
        panic(f"Failed to save tokenizer: {exc}")


if __name__ == '__main__':
    main()
