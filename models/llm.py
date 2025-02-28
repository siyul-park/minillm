import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms) + self.bias

class FlashAttention(nn.Module):
    def __init__(self, causal: bool = False, tile_size: int = 64):
        super().__init__()
        self.causal = causal
        self.tile_size = tile_size

        self.attn_mask = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """FlashAttention with tiling"""
        seq_len = q.size(1)

        if self.causal and self.attn_mask is None:
            self.attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(q.device)

        attn_mask = self.attn_mask if self.causal else None

        output = torch.zeros_like(q)
        for i in range(0, seq_len, self.tile_size):
            end_idx = min(i + self.tile_size, seq_len)
            q_tile, k_tile, v_tile = q[:, i:end_idx, :], k[:, i:end_idx, :], v[:, i:end_idx, :]
            tile_output = F.scaled_dot_product_attention(q_tile, k_tile, v_tile, attn_mask=attn_mask, dropout_p=0.0)
            output[:, i:end_idx, :] = tile_output

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, causal: bool = False, tile_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention = FlashAttention(causal, tile_size)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm).reshape(x.size(0), x.size(1), 3, self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=2)

        attn_output = self.attention(q.squeeze(2), k.squeeze(2), v.squeeze(2))
        attn_output = attn_output.reshape(x.size(0), x.size(1), self.embed_dim)

        x = x + self.out_proj(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

class LLM(nn.Module):
    """
    A Transformer-based Language Model (LLM) with FlashAttention.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            max_seq_length: int = 512,
            dropout: float = 0.1,
            causal: bool = False,
            tile_size: int = 64,
    ) -> None:
        """
        Initialize the language model.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            max_seq_length (int): Maximum sequence length.
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimension of the feed-forward network.
            dropout (float): Dropout probability.
            causal (bool): If True, apply causal masking (for autoregressive generation).
            tile_size (int): Tile size for FlashAttention.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Token and positional embeddings.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer encoder layers.
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout, causal=causal, tile_size=tile_size)
            for _ in range(num_layers)
        ])

        # Final normalization and output projection.
        self.norm = RMSNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_length = max_seq_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the language model.

        Parameters:
            input_ids (Tensor): Input token indices of shape (B, T).

        Returns:
            Tensor: Logits of shape (B, T, vocab_size).
        """
        B, T = input_ids.shape
        if T > self.max_seq_length:
            raise ValueError("Input sequence length exceeds maximum sequence length")

        # Compute token and positional embeddings.
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Pass through encoder layers.
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output_projection(x)
        return logits
