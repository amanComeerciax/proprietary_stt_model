"""
STT Dataset — PyTorch Dataset and DataLoader collation for audio-transcript pairs.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.features import FeatureExtractor
from tokenizer.tokenizer import CharTokenizer


class STTDataset(Dataset):
    """PyTorch Dataset for Speech-to-Text training.

    Each sample consists of:
        - Audio Log-Mel spectrogram features as a FloatTensor
        - Target token IDs as a LongTensor
        - Sequence lengths for both features and targets
        - Metadata (language, file path, original transcript)
    """

    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/metadata/metadata.csv",
        feature_extractor: Optional[FeatureExtractor] = None,
        tokenizer: Optional[CharTokenizer] = None,
        augment: bool = False,
    ) -> None:
        """Initialize the STT Dataset.

        Args:
            manifest_path: Path to metadata CSV file.
            feature_extractor: FeatureExtractor instance for spectrogram extraction.
            tokenizer: CharTokenizer instance for encoding text into token IDs.
            augment: Whether to apply SpecAugment data augmentation.
        """
        self.manifest_path = Path(manifest_path)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.tokenizer = tokenizer or CharTokenizer()
        self.augment = augment
        self.manifest: List[Dict[str, str]] = []

        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load and validate samples from the metadata CSV."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("audio_path", "").strip()
                transcript = row.get("transcript", "").strip()
                lang = row.get("language", "en").strip()
                speaker = row.get("speaker", "amaan").strip()

                if path and os.path.exists(path) and transcript:
                    self.manifest.append({
                        "audio_path": path,
                        "transcript": transcript,
                        "language": lang,
                        "speaker": speaker,
                    })

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample.

        Args:
            idx: Index of sample.

        Returns:
            Dictionary with feature tensor, targets tensor, and lengths.
        """
        sample = self.manifest[idx]
        audio_path = sample["audio_path"]
        transcript = sample["transcript"]

        # Extract 80-channel Log-Mel spectrogram (shape: 80, time_steps)
        mel = self.feature_extractor.extract_from_file(audio_path)

        if self.augment:
            mel = self.feature_extractor.apply_spec_augment(mel)

        # Transpose to (time_steps, n_mels) which is standard for acoustic models
        features = torch.from_numpy(mel.T).float()  # shape: (T, 80)
        feature_length = features.shape[0]

        # Tokenize target text
        token_ids = self.tokenizer.encode(transcript)
        targets = torch.tensor(token_ids, dtype=torch.long)
        target_length = len(token_ids)

        return {
            "features": features,
            "targets": targets,
            "feature_length": feature_length,
            "target_length": target_length,
            "language": sample["language"],
            "audio_path": audio_path,
            "transcript": transcript,
        }

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Any]],
        pad_id: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Custom collate function for padding variable-length sequences in a batch.

        Args:
            batch: List of sample dictionaries from __getitem__.
            pad_id: Integer token ID used for padding targets (default: 1 for <PAD>).

        Returns:
            Dictionary containing:
                - "features": FloatTensor of shape (B, T_max, n_mels)
                - "targets": LongTensor of shape (B, U_max)
                - "feature_lengths": LongTensor of shape (B,) with unpadded frame counts
                - "target_lengths": LongTensor of shape (B,) with unpadded token counts
        """
        batch_size = len(batch)

        # Find maximum time_steps and target_length in this batch
        max_feat_len = max(item["feature_length"] for item in batch)
        max_target_len = max(item["target_length"] for item in batch)
        n_mels = batch[0]["features"].shape[1]

        # Allocate padded tensors
        padded_features = torch.zeros((batch_size, max_feat_len, n_mels), dtype=torch.float32)
        padded_targets = torch.full((batch_size, max_target_len), fill_value=pad_id, dtype=torch.long)

        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        target_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, item in enumerate(batch):
            f_len = item["feature_length"]
            t_len = item["target_length"]

            padded_features[i, :f_len, :] = item["features"]
            padded_targets[i, :t_len] = item["targets"]

            feature_lengths[i] = f_len
            target_lengths[i] = t_len

        return {
            "features": padded_features,
            "targets": padded_targets,
            "feature_lengths": feature_lengths,
            "target_lengths": target_lengths,
        }


def create_dataloader(
    manifest_path: str = "data/metadata/metadata.csv",
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = False,
) -> DataLoader:
    """Helper function to instantiate an STT DataLoader.

    Args:
        manifest_path: Path to metadata CSV.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data.
        augment: Whether to apply SpecAugment.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    tokenizer = CharTokenizer()
    vocab_path = "tokenizer/vocab.json"
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        tokenizer.build_from_metadata(manifest_path)

    extractor = FeatureExtractor()
    dataset = STTDataset(
        manifest_path=manifest_path,
        feature_extractor=extractor,
        tokenizer=tokenizer,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: STTDataset.collate_fn(b, pad_id=tokenizer.pad_id),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("STTDataset & DataLoader — Self Verification Test")
    print("=" * 60)

    loader = create_dataloader(
        manifest_path="data/metadata/metadata.csv",
        batch_size=4,
        shuffle=True,
    )

    print(f"Dataset Size: {len(loader.dataset)} samples")
    print(f"Number of Batches (batch_size=4): {len(loader)}")

    # Fetch 1 batch to verify shapes
    batch = next(iter(loader))
    features = batch["features"]
    targets = batch["targets"]
    feat_lens = batch["feature_lengths"]
    target_lens = batch["target_lengths"]

    print("\n--- Batch Inspection ---")
    print(f"  Features tensor shape    : {features.shape}  (Batch, Time, MelChannels)")
    print(f"  Targets tensor shape     : {targets.shape}   (Batch, MaxTokens)")
    print(f"  Feature lengths          : {feat_lens.tolist()}")
    print(f"  Target lengths           : {target_lens.tolist()}")
    print("  ✓ STTDataset and DataLoader verified successfully!")
    print("=" * 60)
