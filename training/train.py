"""
Training Loop — Production-ready training pipeline for the proprietary STT model.

Responsibility: Person 4 (Training + Validation + Optimization + Deployment)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.validate import validate
from tokenizer.tokenizer import CharTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    """Manages the end-to-end training process for the STT model.

    Features:
        - AdamW optimizer with weight decay
        - Cosine Annealing learning rate schedule
        - Gradient clipping to prevent gradient explosion
        - Automatic checkpointing (best model & latest checkpoint)
        - Validation with Word Error Rate (WER) and Character Error Rate (CER)
        - Early stopping support

    Args:
        model: The STTModel instance to train.
        config: TrainingConfig with hyperparameters.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        device: Device to train on ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        tokenizer: Optional[CharTokenizer] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader or train_loader
        self.tokenizer = tokenizer

        # 1. Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 2. Scheduler: Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6,
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_ce_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            loss, metrics = self.model.compute_loss(
                features=features,
                feature_lengths=feature_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Optimizer step
            self.optimizer.step()

            total_loss += metrics["loss"]
            total_ctc_loss += metrics["ctc_loss"]
            total_ce_loss += metrics["ce_loss"]

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(self.train_loader):
                logger.info(
                    f"  ↳ [Batch {batch_idx+1:02d}/{len(self.train_loader):02d}] "
                    f"Loss: {loss.item():.4f} (CTC: {metrics['ctc_loss']:.4f}, CE: {metrics['ce_loss']:.4f})"
                )

        num_batches = max(len(self.train_loader), 1)
        return {
            "loss": total_loss / num_batches,
            "ctc_loss": total_ctc_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
        }

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save a model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config.to_dict(),
        }

        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f" Saved new best model checkpoint to {best_path} (Val Loss: {val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Load model and optimizer state from checkpoint. Returns start_epoch."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}, starting from scratch.")
            return 1
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"✓ Resumed from {path} at epoch {start_epoch} (Previous Best Val Loss: {self.best_val_loss:.4f})")
        return start_epoch

    def train(self, max_epochs: Optional[int] = None, start_epoch: int = 1) -> Dict[str, Any]:
        """Run the full training loop across epochs."""
        num_epochs = max_epochs or self.config.num_epochs
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Total Epochs: {num_epochs} (Starting at {start_epoch}), Batch Size: {self.config.batch_size}")

        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_wer": [],
            "val_cer": [],
        }

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start = time.time()

            # 1. Training step
            train_metrics = self.train_epoch(epoch)

            # 2. Validation step
            val_metrics = validate(
                model=self.model,
                val_loader=self.val_loader,
                tokenizer=self.tokenizer,
                device=self.device,
            )

            # 3. Learning rate scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 4. Record history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_wer"].append(val_metrics["val_wer"])
            history["val_cer"].append(val_metrics["val_cer"])

            epoch_time = time.time() - epoch_start

            # 5. Check if best model
            val_loss = val_metrics["val_loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(epoch, val_loss, is_best=False)

            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} (CTC: {train_metrics['ctc_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"WER: {val_metrics['val_wer'] * 100:.1f}% | "
                f"CER: {val_metrics['val_cer'] * 100:.1f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # 6. Early stopping check
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.config.early_stopping_patience} "
                    f"epochs without improvement."
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes!")

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch,
            "history": history,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("STT Training Pipeline")
    print("=" * 60)

    import argparse
    from dataset.stt_dataset import create_dataloader
    from model.stt_model import STTModel

    parser = argparse.ArgumentParser(description="Train Proprietary STT Model")
    parser.add_argument("--epochs", type=int, default=30, help="Target total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--manifest", type=str, default="data/metadata/metadata.csv", help="Path to manifest CSV")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir="checkpoints/",
    )

    train_loader = create_dataloader(
        manifest_path=args.manifest,
        batch_size=config.batch_size,
        shuffle=True,
    )

    tokenizer = train_loader.dataset.tokenizer
    model = STTModel(decoder_config={"vocab_size": tokenizer.vocab_size})

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=train_loader,
        device=args.device,
        tokenizer=tokenizer,
    )

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        start_epoch = trainer.load_checkpoint(args.resume)

    print(f"Starting STT Training from epoch {start_epoch} to {args.epochs} on {trainer.device}...")
    result = trainer.train(max_epochs=args.epochs, start_epoch=start_epoch)
    print("=" * 60)
    print(f"✓ Training finished! Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"  Best checkpoint saved at: checkpoints/best_model.pt")
    print("=" * 60)
