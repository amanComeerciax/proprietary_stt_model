"""
Decoder — Autoregressive decoder that converts encoder outputs into token sequences.

Responsibility: Person 3 (Model Architecture)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Autoregressive decoder for the STT model.

    Attends to encoder output and generates token predictions one step
    at a time during inference, or processes the full target sequence
    during training (with teacher forcing).

    Architecture:
        1. Token embedding layer.
        2. Positional encoding.
        3. Stack of Transformer decoder layers with cross-attention
           to encoder output.
        4. Linear projection to vocabulary logits.

    Args:
        vocab_size: Size of the output vocabulary.
        embedding_dim: Dimension of token embeddings.
        hidden_dim: Hidden dimension size (must match encoder hidden_dim).
        num_layers: Number of Transformer decoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # TODO: Implement embedding + positional encoding
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.pos_encoding = PositionalEncoding(...)

        # TODO: Implement Transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(...)
        # self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # TODO: Output projection
        # self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode encoder output given target tokens (teacher forcing).

        Args:
            targets: Target token IDs of shape (batch, target_len).
            encoder_output: Encoder output of shape (batch, src_len, hidden_dim).
            target_mask: Causal mask for target self-attention.
            memory_mask: Mask for cross-attention to encoder output.

        Returns:
            Logits over vocabulary of shape (batch, target_len, vocab_size).
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Decoder.forward() not yet implemented.")

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate a causal (upper-triangular) mask for autoregressive decoding.

        Args:
            size: Length of the target sequence.

        Returns:
            Boolean mask of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
