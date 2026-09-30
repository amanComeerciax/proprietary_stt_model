"""
STT Model — Full end-to-end Speech-to-Text model.

Combines the Encoder, Attention, and Decoder into a single nn.Module.

Responsibility: Person 3 (Model Architecture)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class STTModel(nn.Module):
    """End-to-end Speech-to-Text model.

    Combines:
        - Encoder: Extracts high-level acoustic features from Mel spectrograms.
        - Decoder: Autoregressively generates token sequences from encoder output.

    Args:
        encoder_config: Dictionary of Encoder constructor arguments.
        decoder_config: Dictionary of Decoder constructor arguments.
    """

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass (training mode with teacher forcing).

        Args:
            features: Mel spectrogram batch of shape (batch, n_mels, time).
            feature_lengths: Lengths of each feature sequence (batch,).
            targets: Target token IDs of shape (batch, max_target_len).
            target_lengths: Lengths of each target sequence (batch,).

        Returns:
            Logits over vocabulary of shape (batch, target_len, vocab_size).
        """
        # TODO: Implement full forward pass
        # 1. Encode features
        # encoder_out, enc_lengths = self.encoder(features, feature_lengths)
        # 2. Decode with teacher forcing
        # logits = self.decoder(targets, encoder_out, ...)
        raise NotImplementedError("STTModel.forward() not yet implemented.")

    @torch.no_grad()
    def transcribe(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        max_decode_length: int = 300,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Inference-time transcription (greedy or beam search).

        Args:
            features: Mel spectrogram of shape (batch, n_mels, time).
            feature_lengths: Lengths of each feature sequence (batch,).
            max_decode_length: Maximum number of tokens to generate.
            beam_width: Beam width (1 = greedy decoding).

        Returns:
            Predicted token IDs of shape (batch, decoded_length).
        """
        # TODO: Implement greedy / beam search decoding
        raise NotImplementedError("STTModel.transcribe() not yet implemented.")

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "STTModel":
        """Construct an STTModel from a configuration dictionary.

        Args:
            config: Configuration dict with 'encoder' and 'decoder' keys.

        Returns:
            Instantiated STTModel.
        """
        return cls(
            encoder_config=config["encoder"],
            decoder_config=config["decoder"],
        )
