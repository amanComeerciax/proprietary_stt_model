"""
Encoder — Converts Mel spectrograms into high-level acoustic representations.

Architecture: Convolutional feature extractor → Transformer encoder layers.

Responsibility: Person 3 (Model Architecture)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Acoustic encoder for the STT model.

    Takes a Mel spectrogram and produces a sequence of hidden representations
    that capture the acoustic content of the speech signal.

    Architecture:
        1. Convolutional front-end for local feature extraction and
           downsampling the time dimension.
        2. Positional encoding for sequence position information.
        3. Stack of Transformer encoder layers for global context modeling.

    Args:
        input_dim: Number of input features (e.g., n_mels = 80).
        hidden_dim: Hidden dimension size for Transformer layers.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        conv_channels: Number of channels in convolutional front-end.
        conv_kernel_size: Kernel size for convolutional layers.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # TODO: Implement convolutional front-end
        # self.conv = nn.Sequential(...)

        # TODO: Implement positional encoding
        # self.pos_encoding = PositionalEncoding(...)

        # TODO: Implement Transformer encoder layers
        # encoder_layer = nn.TransformerEncoderLayer(...)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input spectrogram features.

        Args:
            features: Input Mel spectrogram of shape (batch, n_mels, time).
            feature_lengths: Lengths of each sequence in the batch (batch,).

        Returns:
            Tuple of:
                - Encoder output of shape (batch, time', hidden_dim).
                - Updated lengths after convolutional downsampling (batch,).
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Encoder.forward() not yet implemented.")
