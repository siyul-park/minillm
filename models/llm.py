import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    FlashAttention module implemented as an nn.Module.

    This module computes scaled dot-product attention using a tile-based online
    softmax algorithm, which avoids materializing the full attention matrix.
    """

    def __init__(self, causal: bool = False, dropout_p: float = 0.0, tile_size: int = 128):
        """
        Initialize the FlashAttention module.

        Parameters:
            causal (bool): If True, apply causal masking (for autoregressive models).
            dropout_p (float): Dropout probability. Set to 0.0 during evaluation.
            tile_size (int): Tile size used for processing the sequence.
        """
        super(FlashAttention, self).__init__()
        self.causal = causal
        self.dropout_p = dropout_p
        self.tile_size = tile_size

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FlashAttention module.

        Parameters:
            query, key, value (Tensor): Input tensors with shape (B, H, T, D), where:
                B - batch size,
                H - number of attention heads,
                T - sequence length,
                D - head dimension.

        Returns:
            Tensor: Output tensor of shape (B, H, T, D) containing the attention results.
        """
        B, H, T, D = query.shape
        output = torch.empty_like(query)

        # Process the sequence in tiles to reduce memory usage.
        for start in range(0, T, self.tile_size):
            end = min(start + self.tile_size, T)
            q_tile = query[:, :, start:end, :]  # (B, H, T_q, D)
            T_q = q_tile.size(2)

            # Initialize running statistics for the current query tile.
            running_max = torch.full((B, H, T_q), -float('inf'), device=query.device)
            running_sum = torch.zeros((B, H, T_q), device=query.device)
            output_acc = torch.zeros((B, H, T_q, D), device=query.device)

            # Iterate over key/value tiles.
            for k_start in range(0, T, self.tile_size):
                k_end = min(k_start + self.tile_size, T)
                k_tile = key[:, :, k_start:k_end, :]  # (B, H, T_k, D)
                v_tile = value[:, :, k_start:k_end, :]  # (B, H, T_k, D)
                T_k = k_tile.size(2)

                # Compute scaled dot-product scores.
                scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T_q, T_k)

                # Apply causal masking if needed.
                if self.causal:
                    q_indices = torch.arange(start, end, device=query.device).view(1, 1, T_q, 1)
                    k_indices = torch.arange(k_start, k_end, device=query.device).view(1, 1, 1, T_k)
                    mask = k_indices > q_indices
                    scores = scores.masked_fill(mask, -float('inf'))

                # Compute the maximum for numerical stability.
                tile_max = scores.max(dim=-1, keepdim=True)[0]  # (B, H, T_q, 1)
                new_max = torch.max(running_max.unsqueeze(-1), tile_max).squeeze(-1)  # (B, H, T_q)

                # Scaling factor for running statistics update.
                scaling_factor = torch.exp(running_max - new_max)  # (B, H, T_q)
                exp_scores = torch.exp(scores - new_max.unsqueeze(-1))  # (B, H, T_q, T_k)
                tile_sum = exp_scores.sum(dim=-1)  # (B, H, T_q)

                # Update running softmax denominator and weighted value accumulator.
                running_sum = running_sum * scaling_factor + tile_sum
                output_acc = output_acc * scaling_factor.unsqueeze(-1) + torch.matmul(exp_scores, v_tile)

                # Update running maximum.
                running_max = new_max

            # Normalize accumulated outputs.
            q_tile_out = output_acc / running_sum.unsqueeze(-1)
            if self.dropout_p > 0.0 and self.training:
                q_tile_out = F.dropout(q_tile_out, p=self.dropout_p, training=True)

            output[:, :, start:end, :] = q_tile_out

        return output

class Encoder(nn.Module):
    """
    Standard Transformer Encoder Layer using FlashAttention for self-attention.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            causal: bool = False,
            tile_size: int = 128
    ) -> None:
        """
        Initialize the encoder layer.

        Parameters:
            embed_dim (int): Dimensionality of the model embeddings.
            num_heads (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimensionality of the feed-forward network.
            dropout (float): Dropout probability.
            causal (bool): If True, apply causal masking.
            tile_size (int): Tile size for FlashAttention.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Projections for queries, keys, and values.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # FlashAttention module for self-attention.
        self.attention = FlashAttention(causal=causal, dropout_p=dropout, tile_size=tile_size)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward network.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Parameters:
            x (Tensor): Input tensor of shape (B, T, embed_dim).

        Returns:
            Tensor: Output tensor of shape (B, T, embed_dim).
        """
        B, T, E = x.shape

        # Self-attention sub-layer.
        x_norm = self.norm1(x)
        # Project and reshape for multi-head attention.
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self.attention(q, k, v)  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, E)
        x = x + self.dropout(self.out_proj(attn_output))

        # Feed-forward sub-layer.
        x = x + self.dropout(self.mlp(self.norm2(x)))
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
            tile_size: int = 128
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
            Encoder(embed_dim, num_heads, dropout=dropout, causal=causal, tile_size=tile_size)
            for _ in range(num_layers)
        ])

        # Final normalization and output projection.
        self.norm = nn.LayerNorm(embed_dim)
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
