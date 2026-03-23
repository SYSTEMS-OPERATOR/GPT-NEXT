"""Sequence generation utilities for autoregressive decoding."""

from typing import Any

from torch import LongTensor, Tensor, argsort, cat, full, multinomial, no_grad
from torch import long as long_
from torch.nn.functional import softmax
from tqdm import trange


class Sequencer:
    """Helper class that performs top-k autoregressive text generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        window_size: int,
        k: int,
        device: str,
    ):
        """Initialize sequence generator state."""
        # Dev Agent Breadcrumb: initialize dependencies used by all generation
        # steps to keep data flow explicit across helper methods.
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.k = k

    def generate_sequence(self, length: int, start: str | None = None) -> str:
        """Generate a text sequence with optional prompt text."""
        # Dev Agent Breadcrumb: bootstrap token state from prompt, then iterate
        # one token at a time using model probabilities and top-k sampling.
        tokens, token_ids, ignore_ids = self.generate_start_seq(start)
        idx = len(tokens) - 1

        self.model.eval()
        with no_grad():
            for _ in trange(length):
                probs, _ = self.model(token_ids, ignore_ids)
                next_id = self.gen_next_token(probs, idx)
                tokens.append(self.tokenizer.get_byte(next_id.item()))
                token_ids, ignore_ids, idx = self.update_token_ids(
                    idx, token_ids, next_id
                )

        return self.generate_text(tokens)

    def generate_start_seq(
        self,
        start: str | None = None,
    ) -> tuple[list[str], Tensor, Tensor]:
        """Create initial token/string/id tensors used by generation."""
        pad_id = self.tokenizer.get_byte_id(self.tokenizer.get_pad())
        tokens = [self.tokenizer.get_sol()]

        if start:
            for chunk in start.split(" "):
                bytes_ = list(chunk) + [self.tokenizer.get_eow()]
                tokens += self.tokenizer.merge_bytes(bytes_)

        token_ids = LongTensor(self.tokenizer.get_byte_ids(tokens)).unsqueeze(0)
        token_ids = token_ids.to(device=self.device)
        token_ids = self.pad_token_ids(token_ids, pad_id)
        ignore_ids = (token_ids == pad_id).float().to(device=self.device)
        return tokens, token_ids, ignore_ids

    def update_token_ids(
        self,
        idx: int,
        token_ids: Tensor,
        next_id: Tensor,
    ) -> tuple[Tensor, Tensor, int]:
        """Update sequence tensor by appending/rolling to fixed window size."""
        pad_id = self.tokenizer.get_byte_id(self.tokenizer.get_pad())
        if idx < self.window_size - 1:
            token_ids = cat([token_ids[:, : idx + 1], next_id.unsqueeze(0)], dim=1)
            token_ids = self.pad_token_ids(token_ids, pad_id)
            ignore_ids = (token_ids == pad_id).float()
            idx += 1
        else:
            token_ids = cat([token_ids[:, 1:], next_id.unsqueeze(0)], dim=1)
            ignore_ids = (token_ids == pad_id).float()

        return token_ids, ignore_ids, idx

    def gen_next_token(self, probs: Tensor, idx: int) -> Tensor:
        """Sample the next token from the model's top-k logits."""
        next_id_probs = probs[:, idx, :].flatten()
        next_id_cands = argsort(next_id_probs, descending=True)[: self.k]
        next_id_probs = next_id_probs[next_id_cands]
        next_id_probs = softmax(next_id_probs, dim=0)
        return next_id_cands[multinomial(next_id_probs, 1)]

    def pad_token_ids(self, token_ids: Tensor, pad_id: int) -> Tensor:
        """Pad or trim IDs so the tensor always matches ``window_size``."""
        pad_size = self.window_size - token_ids.size(1)
        if pad_size > 0:
            padding = full((1, pad_size), pad_id, dtype=long_, device=self.device)
            token_ids = cat([token_ids, padding], dim=1)
        elif pad_size < 0:
            token_ids = token_ids[:, -self.window_size :]
        return token_ids

    def generate_text(self, tokens: list[str]) -> str:
        """Convert a list of tokenizer bytes into human-readable text."""
        text, word = [], ""
        for token in tokens:
            word += token
            if word.startswith("<line/>"):
                word = word[7:]
            elif word.endswith("</line>"):
                word = word[:-7]

            if word.endswith("</w>"):
                text.append(word[:-4])
                word = ""

        return " ".join(text)
