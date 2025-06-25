"""Command-line interface for generating text with a trained GPT model."""

import argparse
import os
import sys

from miniyaml import load as load_yaml

try:
    from torch import load
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required to run this script") from exc

from model.tokenizer import BytePairTokenizer
from model.sequencer import Sequencer
from model.model import GPT
from sentinel import panic


# Default configuration path. The repository stores configuration files under
# the `conf/` directory so the generation config should live there.
confpath = 'conf/generate.yml'


def main():
    """Run the text generation utility from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default=confpath)
    parser.add_argument('-l', '--length', type=int, default=128)
    args = parser.parse_args()
    length = args.length

    try:
        confs = load_yaml(args.conf)
    except FileNotFoundError:
        panic(f"Config not found: {args.conf}")
    except Exception as exc:  # pragma: no cover - unexpected parse errors
        panic(f"Failed to load config: {exc}")

    model_path = confs.get('pretrained_model')
    if not os.path.isfile(model_path):
        panic(f"Model file missing: {model_path}")

    try:
        model = GPT(**confs['model'])
        model.load_state_dict(load(model_path))
    except Exception as exc:
        panic(f"Could not load model: {exc}")

    tok_path = confs.get('trained_tokenizer')
    if not os.path.isdir(tok_path):
        panic(f"Tokenizer data missing: {tok_path}")

    try:
        tokenizer = BytePairTokenizer.load(tok_path)
    except Exception as exc:
        panic(f"Could not load tokenizer: {exc}")

    sequencer = Sequencer(model, tokenizer, **confs['sequencer'])
    try:
        sequence = sequencer.generate_sequence(length)
    except Exception as exc:
        panic(f"Generation failed: {exc}")

    print(sequence)


if __name__ == '__main__':
    main()
