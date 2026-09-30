"""
STT Dataset — PyTorch Dataset for loading audio–transcript pairs.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class STTDataset(Dataset):
    """PyTorch Dataset for Speech-to-Text training.

    Each sample consists of:
        - Audio features (e.g., Mel spectrogram) as a tensor
        - Target token IDs as a tensor
        - Metadata (language, duration, file path)

    Attributes:
        manifest: List of dictionaries containing sample metadata.
        feature_extractor: FeatureExtractor instance for computing features.
        tokenizer: CharTokenizer instance for encoding transcriptions.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        feature_extractor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize the STT Dataset.

        Args:
            manifest_path: Path to the manifest file (CSV/JSON) listing
                audio file paths and their transcriptions.
            feature_extractor: Optional FeatureExtractor for computing
                spectrograms on-the-fly.
            tokenizer: Optional CharTokenizer for encoding transcriptions.
        """
        self.manifest_path = Path(manifest_path)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.manifest: List[Dict[str, Any]] = []

        # TODO: Load manifest from file
        # Expected manifest columns: audio_path, transcript, language, duration

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample as a dictionary of tensors.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary with keys:
                - "features": Tensor of shape (n_mels, time_steps)
                - "targets": Tensor of target token IDs
                - "feature_lengths": Scalar tensor with feature sequence length
                - "target_lengths": Scalar tensor with target sequence length
        """
        # TODO: Implement sample loading and processing
        raise NotImplementedError("STTDataset.__getitem__() not yet implemented.")

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Custom collate function for padding variable-length sequences.

        Args:
            batch: List of sample dictionaries from __getitem__.

        Returns:
            Batched and padded dictionary of tensors.
        """
        # TODO: Implement padding and batching logic
        raise NotImplementedError("STTDataset.collate_fn() not yet implemented.")
