"""Command-line interface for generating text with a trained GPT model."""

import argparse

from miniyaml import load as load_yaml

try:
    from torch import load
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required to run this script") from exc

from model.tokenizer import BytePairTokenizer
from model.sequencer import Sequencer
from model.model import GPT


# Default configuration path. The repository stores configuration files under
# the `conf/` directory so the generation config should live there.
confpath = 'conf/generate.yml'


def main():
    """Run the text generation utility from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=128)
    args = parser.parse_args()
    length = args.length

    confs = load_yaml(confpath)
    model = GPT(**confs['model'])
    model.load_state_dict(load(confs['pretrained_model'])) 
    tokenizer = BytePairTokenizer.load(confs['trained_tokenizer'])

    sequencer = Sequencer(model, tokenizer, **confs['sequencer'])
    sequence = sequencer.generate_sequence(length)
    print(sequence)


if __name__ == '__main__':
    main()
