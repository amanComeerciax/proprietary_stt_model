"""
Training Configuration — Dataclass-based training hyperparameters.

Responsibility: Person 4 (Training + Validation + Optimization + Deployment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths.

    Attributes:
        batch_size: Number of samples per training batch.
        num_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization factor.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        scheduler: Learning rate scheduler type (cosine, step, plateau).
        early_stopping_patience: Epochs to wait before early stopping.
        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory for training logs.
    """

    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 5.0
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        """Load training config from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance.
        """
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        training_section = full_config.get("training", {})
        return cls(**training_section)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a plain dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "scheduler": self.scheduler,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
        }
