import argparse
import types

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for missing PyYAML
    from minimal_yaml import safe_load
    yaml = types.SimpleNamespace(safe_load=safe_load)

try:
    from tqdm import trange
except ModuleNotFoundError:  # pragma: no cover - fallback if tqdm missing
    trange = range

try:
    from torch import load
    MISSING_TORCH = False
except ModuleNotFoundError:  # pragma: no cover - allow running without torch
    MISSING_TORCH = True
    def load(*a, **k):
        raise RuntimeError('PyTorch is required to load models')


"""Utility script for generating text from a trained GPT model."""


confpath = 'conf/generate.yml'


def main():
    """Generate a text sequence using a pretrained model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=128)
    args = parser.parse_args()
    length = args.length

    if MISSING_TORCH:  # pragma: no cover - informative exit
        print('PyTorch is required to run generation. Please install torch.')
        return

    from model.tokenizer import BytePairTokenizer
    from model.sequencer import Sequencer
    from model.model import GPT

    confs = yaml.safe_load(open(confpath))
    model = GPT(**confs['model'])
    model.load_state_dict(load(confs['pretrained_model'])) 
    tokenizer = BytePairTokenizer.load(confs['trained_tokenizer'])

    sequencer = Sequencer(model, tokenizer, **confs['sequencer'])
    sequence = sequencer.generate_sequence(length)
    print(sequence)


if __name__ == '__main__':
    main()
