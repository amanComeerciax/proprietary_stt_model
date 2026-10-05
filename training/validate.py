"""
Validation — Runs model evaluation and WER/CER computation on validation sets.

Responsibility: Person 4 (Training + Validation + Optimization + Deployment)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_batch_wer, compute_batch_cer
from tokenizer.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    tokenizer: Optional[CharTokenizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Run a full evaluation pass over the validation set.

    Computes:
        - Validation Loss (Joint CTC + Attention)
        - CTC Loss
        - Character Error Rate (CER)
        - Word Error Rate (WER)

    Args:
        model: The STTModel to evaluate.
        val_loader: DataLoader for the validation set.
        tokenizer: Optional CharTokenizer for decoding predictions.
        device: Device to run validation on.

    Returns:
        Dictionary with validation metrics:
            - "val_loss": Average validation loss.
            - "val_ctc_loss": Average CTC loss.
            - "val_wer": Average Word Error Rate.
            - "val_cer": Average Character Error Rate.
    """
    model.eval()

    total_loss = 0.0
    total_ctc_loss = 0.0
    all_refs: List[str] = []
    all_hyps: List[str] = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            # Compute loss
            loss, metrics = model.compute_loss(
                features=features,
                feature_lengths=feature_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )

            total_loss += metrics["loss"]
            total_ctc_loss += metrics["ctc_loss"]

            # Decode predictions for WER/CER (sample first 40 items for speed)
            if tokenizer is not None and len(all_hyps) < 40:
                # Predictions via fast CTC transcribe
                preds = model.transcribe(features, feature_lengths, tokenizer=tokenizer)
                all_hyps.extend(preds)

                # References from target tokens
                for i in range(targets.shape[0]):
                    t_len = target_lengths[i].item()
                    ref_ids = targets[i, :t_len].tolist()
                    ref_text = tokenizer.decode(ref_ids, remove_special=True)
                    all_refs.append(ref_text)

    num_batches = max(len(val_loader), 1)
    avg_loss = total_loss / num_batches
    avg_ctc_loss = total_ctc_loss / num_batches

    wer = 0.0
    cer = 0.0
    if all_refs and all_hyps:
        wer = compute_batch_wer(all_refs, all_hyps)
        cer = compute_batch_cer(all_refs, all_hyps)

    return {
        "val_loss": avg_loss,
        "val_ctc_loss": avg_ctc_loss,
        "val_wer": wer,
        "val_cer": cer,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Validation Module — Verification Test")
    print("=" * 60)

    from model.stt_model import STTModel
    from dataset.stt_dataset import create_dataloader

    model = STTModel()
    loader = create_dataloader(batch_size=2)
    tokenizer = loader.dataset.tokenizer

    results = validate(model, loader, tokenizer=tokenizer)
    print(f"  Validation Loss     : {results['val_loss']:.4f}")
    print(f"  Validation CTC Loss : {results['val_ctc_loss']:.4f}")
    print(f"  Validation WER      : {results['val_wer']:.4f} ({results['val_wer'] * 100:.1f}%)")
    print(f"  Validation CER      : {results['val_cer']:.4f} ({results['val_cer'] * 100:.1f}%)")
    print("  ✓ Validation module verified successfully!")
    print("=" * 60)
