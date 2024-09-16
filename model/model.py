# model.py
from typing import Dict, List
import torch
from torch.nn import ModuleList, Embedding, LayerNorm, Dropout, Softmax, \
                     Linear, Module, GELU
from torch import LongTensor, Tensor, einsum, ones, sqrt, tril, triu, cat
from torch.nn.init import normal_, ones_, zeros_

class GPT(Module):
    def __init__(self, vocab: int, seq: int, n_layers: int, n_heads: int, 
                 dim: int, hidden: int, dropout: float, device: str, tokenizer=None):
        """ Initialize GPT-1 replica module

        Args:
            vocab: vocabulary size
            seq: window size
            n_layers: number of transformer block layers
            n_heads: number of multi-headed attention heads per block
            dim: embedding dimension
            hidden: number of hidden units
            dropout: dropout amount
            device: device to load model onto
            tokenizer: tokenizer to be used with the model
        """
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.seq = seq
        self.bpe_embed = Embedding(vocab, dim).to(device)
        self.pos_embed = Embedding(seq, dim).to(device)
        self.pos = torch.arange(seq, device=device).unsqueeze(0)  # Adjusted for dynamic sequence length
        self.blocks = ModuleList([
            TransformerBlock(n_heads, dim, hidden, dropout, device) \
            for _ in range(n_layers)
        ])
        self.output = Linear(dim, vocab).to(device)
        self.drop = Dropout(dropout).to(device)
        self.init_weights()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # Placeholder for tokenizer initialization
            # You should replace this with your actual tokenizer
            self.tokenizer = None

    def init_weights(self):
        normal_(self.bpe_embed.weight, mean=0.0, std=0.02) 
        normal_(self.pos_embed.weight, mean=0.0, std=0.02) 
        normal_(self.output.weight, mean=0.0, std=0.02) 
        zeros_(self.output.bias) 

    def forward(self, x, ignore=None):
        """
        Forward pass of the GPT model.
        Args:
            x (Tensor): Input tensor of token IDs.
            ignore (Tensor, optional): Ignore mask tensor.
        Returns:
            Tuple[Tensor, Tensor]: Logits and hidden states.
        """
        seq_length = x.size(1)
        be = self.bpe_embed(x)
        pe = self.pos_embed(self.pos[:, :seq_length])
        out = self.drop(be + pe)
        for block in self.blocks:
            out = block(out, ignore)
        logits = self.output(out)
        return logits, out  # Return both logits and hidden states

    def get_next_token_probabilities(self, sequence):
        """
        Get the probabilities of the next token given a sequence.
        Args:
            sequence (list): List of token IDs.
        Returns:
            Tensor: Probabilities for the next token.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.forward(x)
            logits = logits[:, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs

    def get_parameters(self) -> List[Dict]:
        """ Return model parameters with corresponding weight decay parameter
            for optimizer
        Returns:
            (List[Dict]): list of parameters with corresponding weight decay
        """
        params = [
            { 'params': [], 'weight_decay': 1e-2 },
            { 'params': [], 'weight_decay': 0.00 },
        ]
        for name, parameter in self.named_parameters():
            if ('att' in name or 'ffl' in name or 'output' in name) and \
               name.endswith('weight'):
                params[0]['params'].append(parameter)
            else:
                params[1]['params'].append(parameter)
        return params

class TransformerBlock(Module):
    def __init__(self, n_heads: int, dim: int, hidden: int, dropout: float,
                 device: str):
        """ Initialize transformer block
        Args:
            n_heads: number of multi-headed attention heads
            dim: dimension of input / output
            hidden: hidden units in feed-forward layer
            dropout: dropout amount
            device: device to load layer onto
        """
        super().__init__()
        self.att = MultiHeadAttentionLayer(n_heads, dim, device)
        self.ffl = FeedForwardLayer(dim, hidden, device)
        self.norm1 = LayerNorm(dim).to(device)
        self.norm2 = LayerNorm(dim).to(device)
        self.drop1 = Dropout(dropout).to(device)
        self.drop2 = Dropout(dropout).to(device)
        self.init_weights()

    def init_weights(self):
        ones_(self.norm1.weight)
        ones_(self.norm2.weight)
        zeros_(self.norm1.bias)
        zeros_(self.norm2.bias)

    def forward(self, x, ignore=None):
        out = self.norm1(x + self.drop1(self.att(x, ignore)))
        return self.norm2(out + self.drop2(self.ffl(out)))

class MultiHeadAttentionLayer(Module):
    def __init__(self, n_heads: int, dim: int, device: str):
        """ Initialize multi-headed attention layer
        Args:
            n_heads: number of attention heads
            dim: dimension of input / output
            device: device to load layer onto
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.device = device
        self.head_dim = dim // n_heads
        self.heads = ModuleList([
            SelfAttentionLayer(dim, self.head_dim, device) \
            for _ in range(n_heads)
        ])
        self.wo = Linear(dim, dim).to(device)
        self.init_weights()

    def init_weights(self):
        normal_(self.wo.weight, mean=0.0, std=0.02) 
        zeros_(self.wo.bias) 

    def forward(self, x, ignore=None):
        out = cat([head(x, ignore) for head in self.heads], dim=2)
        return self.wo(out)

class SelfAttentionLayer(Module):
    def __init__(self, d_in: int, d_out: int, device: str):
        """ Initialize self-attention layer
        Args:
            d_in: input tensor size
            d_out: output tensor size
            device: device to load layer onto
        """
        super().__init__()
        self.wq = Linear(d_in, d_out).to(device)
        self.wk = Linear(d_in, d_out).to(device)
        self.wv = Linear(d_in, d_out).to(device)
        self.scale = sqrt(torch.tensor(d_out, dtype=torch.float32)).to(device)
        self.softmax = Softmax(dim=2).to(device)
        self.device = device
        self.init_weights()

    def init_weights(self):
        normal_(self.wq.weight, mean=0.0, std=0.02) 
        normal_(self.wk.weight, mean=0.0, std=0.02) 
        normal_(self.wv.weight, mean=0.0, std=0.02) 
        zeros_(self.wq.bias) 
        zeros_(self.wk.bias) 
        zeros_(self.wv.bias) 

    def forward(self, x, ignore=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        att = einsum('bqd,bkd->bqk', q, k)
        att /= self.scale
        seq_length = x.size(1)
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device))
        if ignore is not None:
            ignore_mask = self.get_ignore_mask(att, ignore)
            mask = causal_mask.unsqueeze(0) * ignore_mask
        else:
            mask = causal_mask.unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = self.softmax(att)
        att = einsum('bqk,bkd->bqd', att, v)
        return att

    def get_ignore_mask(self, att: Tensor, ignore: Tensor) -> Tensor:
        """ Create mask for attention indices to be ignored
        Args:
            att: attention matrix, outcome of qk/d_k
            ignore: tensor indicating indices of input to ignore
        Returns:
            (Tensor): tensor matching att shape zeroing out all columns to be
                      ignored
        """
        batch_size, seq_length = ignore.size()
        ignore = ignore.view(batch_size, 1, seq_length)
        ignore_mask = ignore.repeat(1, seq_length, 1)
        return ignore_mask

class FeedForwardLayer(Module):
    def __init__(self, d_in: int, d_h: int, device: str):
        """ Initialize feed-forward layer
        Args:
            d_in: input units
            d_h: hidden units
            device: device to run layer on
        """
        super().__init__()
        self.l1 = Linear(d_in, d_h).to(device)
        self.l2 = Linear(d_h, d_in).to(device)
        self.gelu = GELU().to(device)
        self.init_weights()

    def init_weights(self) -> None:
        normal_(self.l1.weight, mean=0.0, std=0.02) 
        normal_(self.l2.weight, mean=0.0, std=0.02) 
        zeros_(self.l1.bias) 
        zeros_(self.l2.bias) 

    def forward(self, x: Tensor) -> Tensor:
        out = self.gelu(self.l1(x))
        out = self.l2(out)
        return out
