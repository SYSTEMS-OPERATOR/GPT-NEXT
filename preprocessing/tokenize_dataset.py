"""Tokenization utilities used to prepare datasets for GPT training."""

from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import repeat
from typing import List
import os
import sys

from nltk import wordpunct_tokenize, sent_tokenize
from tqdm import tqdm

from model.tokenizer import BytePairTokenizer, count_byte_freqs
from sentinel import panic, sanitize_path


def tokenize_file(filepath: str, outdir: str, tokenizer: BytePairTokenizer,
                  line_length: int) -> None:
    """ Tokenize given file and write token ids to new file

    Args:
        filepath: filepath of file to tokenize
        outdir: directory to store tokenized file
        tokenizer: tokenizer instance to use to tokenize file
    """

    outpath = f"{outdir}/{os.path.basename(filepath)}"
    try:
        with open(filepath, encoding='utf-8-sig') as fh:
            lines = sent_tokenize(fh.read())
    except FileNotFoundError:
        panic(f"Input file not found: {filepath}")

    tokens = []
    for line in lines:
        if len(line) > 1:
            tokens += get_line_ids(line, tokenizer)

    start, end = 0, line_length 
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, 'w') as outfile:
        while start < len(tokens):
            if len(tokens[start:end]) == line_length:
                outstr = ' '.join([str(x) for x in tokens[start:end]])
                outfile.write(f'{outstr}\n')
            start += line_length
            end += line_length


def get_line_ids(line: str, tokenizer: BytePairTokenizer) -> List[int]:
    """ Take line and return list of token ids for line

    Args:
        line: line to tokenize 
        tokenizer: tokenizer to use to tokenize line

    Return:
        (List[int]): list of token ids
    """

    tokens = wordpunct_tokenize(line)
    tokens = [list(t) + [tokenizer.get_eow()] for t in tokens]

    lineids = []
    for token in tokens:
        token = tokenizer.merge_bytes(token)
        ids = tokenizer.get_byte_ids(token)
        lineids += ids
    
    sol_id = tokenizer.get_byte_id(tokenizer.get_sol())
    eol_id = tokenizer.get_byte_id(tokenizer.get_eol())
    lineids = [sol_id] + lineids + [eol_id]
    return lineids


def main():
    """Tokenize an input dataset using a trained tokenizer."""

    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-l', '--line_length', required=True, type=int)
    parser.add_argument('-j', '--jobs', required=True, type=int)
    args = parser.parse_args()
    line_length = args.line_length
    checkpoint = sanitize_path(args.checkpoint)
    outdir = sanitize_path(args.outdir)
    inpath = sanitize_path(args.inpath)
    jobs = args.jobs

    try:
        with open(inpath, 'r', encoding='utf-8') as infile:
            filepaths = [line.strip() for line in infile.readlines()]
    except FileNotFoundError:
        panic(f"File list not found: {inpath}")
    try:
        tokenizer = BytePairTokenizer.load(checkpoint)
    except Exception as exc:
        panic(f"Failed to load tokenizer: {exc}")

    progress = tqdm(total=len(filepaths))
    start, end = 0, jobs
    while start < len(filepaths):

        paths = filepaths[start:end]

        try:
            with Pool(jobs) as pool:
                pool.starmap(
                    tokenize_file,
                    zip(
                        paths,
                        repeat(outdir),
                        repeat(tokenizer),
                        repeat(line_length)
                    )
                )
        except Exception as exc:
            panic(f"Worker failure: {exc}")

        progress.update(len(paths))
        start += jobs
        end += jobs


if __name__ == '__main__':
    main()
