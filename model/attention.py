"""
Attention — Multi-head attention mechanisms for the STT model.

Responsibility: Person 3 (Model Architecture)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    """Multi-Head Attention block for cross-attention between encoder and decoder.

    Implements scaled dot-product attention with multiple parallel heads,
    suitable for both self-attention and cross-attention use cases.

    Args:
        hidden_dim: Dimension of the model (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # TODO: Initialize projection layers
        # self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_q, hidden_dim).
            key: Key tensor of shape (batch, seq_k, hidden_dim).
            value: Value tensor of shape (batch, seq_k, hidden_dim).
            mask: Optional attention mask of shape (batch, seq_q, seq_k).

        Returns:
            Tuple of:
                - Attention output of shape (batch, seq_q, hidden_dim).
                - Attention weights of shape (batch, num_heads, seq_q, seq_k).
        """
        # TODO: Implement multi-head attention
        raise NotImplementedError(
            "MultiHeadAttentionBlock.forward() not yet implemented."
        )
