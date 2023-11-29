import torch
import torch.nn as nn

class GPT1(nn.Module):
    """
    Simplified version of the GPT-1 model.
    Args:
        vocab_size (int): Size of the vocabulary.
        max_seq_len (int): Maximum sequence length.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        d_ff (int): Dimension of the feed-forward network.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, vocab_size, max_seq_len, n_layers=12, n_heads=12, d_model=768, d_ff=3072, dropout_rate=0.1):
        super(GPT1, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass of the GPT-1 model.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor.
        """
        positions = torch.arange(0, x.size(1)).expand(x.size(0), x.size(1)).to(x.device)
        x = self.token_embeddings(x) + self.position_embeddings(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block used in the GPT-1 model.
    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the Transformer block.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor.
        """
        x2 = self.norm1(x)
        x = x + self.dropout(self.attention(x2, x2, x2))
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        """
        Forward pass of the multi-head attention.
        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
        Returns:
            Tensor: Output tensor.
        """
        bs = q.size(0)

        # Perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # Transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # Calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(context)

        return output
