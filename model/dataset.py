"""Datasets for handling token ID sequences used during GPT training."""

from random import randint, sample
import os

from sentinel import sanitize_path

from torch import FloatTensor, LongTensor, Tensor, stack, cat
from torch.utils.data import IterableDataset
from torch.nn.functional import one_hot


class TokenIDDataset(IterableDataset):


    def __init__(self, datapath: str, window_size: int, vocab_size: int, 
                 unk: int):
        """ Dataset class for dataset of variable length lines of text token
            byte pair ids

        Args:
            datapath: file where data is located
            window_size: size of window of data to return
            vocab_size: total vocab size for one-hot encodings
            unk: token id for unknown token
        """
        super().__init__()
        self.datapath = sanitize_path(datapath)
        if not os.path.isfile(self.datapath):
            raise FileNotFoundError(f"Dataset not found: {self.datapath}")
        try:
            with open(self.datapath, "r", encoding="utf-8") as infile:
                self.data = infile.readlines()
        except OSError as exc:
            raise RuntimeError(f"Failed to read dataset {self.datapath}") from exc
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.unk_token = unk


    def __iter__(self):
        """Yield one training sample at a time."""
        for line_idx in range(len(self.data)):
            line = self.data[line_idx].strip().split(' ')
            if len(line) <= self.window_size:
                continue  # Skip lines that are too short
            start = randint(0, len(line) - self.window_size - 1)
            end = start + self.window_size + 1

            try:
                int_ids = [int(x) for x in line[start:end]]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid token in {self.datapath} line {line_idx}: {exc}"
                ) from exc

            for tid in int_ids:
                if tid < 0 or tid >= self.vocab_size:
                    raise ValueError(
                        f"Token ID {tid} out of range in {self.datapath} line {line_idx}"
                    )

            ids = LongTensor(int_ids)
            ignore = (ids == self.unk_token).float()

            yield ids[:-1], ids[1:], ignore[:-1]


    def __len__(self):
        """Return the number of lines available for training."""
        return len(self.data)


    @staticmethod
    def collate(batch: Tensor) -> (Tensor, Tensor, Tensor):
        """ Join batch of TokenIDDataset members

        Args:
            batch: batch of ids 

        Returns:
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined indicators for indices to ignore
        """

        xids = cat([batch[i][0][None, :] for i in range(len(batch))], dim=0)
        yids = cat([batch[i][1][None, :] for i in range(len(batch))], dim=0)
        ignore = cat([batch[i][2][None, :] for i in range(len(batch))], dim=0)
        return xids, yids, ignore 


class TokenIDSubset(TokenIDDataset):


    def __init__(self, dataset: TokenIDDataset, size: int):
        """ Dataset class for subset of byte pair token id dataset 

        Args:
            dataset: token id dataset to subset
            size: number of lines to sample from token id dataset
        """
        self.data = sample(dataset.data, size)
        self.window_size = dataset.window_size
        self.vocab_size = dataset.vocab_size
        self.unk_token = dataset.unk_token


    def __iter__(self):
        """Iterate over a sampled subset of the dataset."""
        yield from super().__iter__()


    def __len__(self):
        """Return the subset length."""
        return super().__len__()
