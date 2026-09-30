"""
Validation — Runs model evaluation on the validation set during training.

Responsibility: Person 4 (Training + Validation + Optimization + Deployment)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run a full validation pass over the validation set.

    Args:
        model: The STTModel to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        Dictionary with:
            - "val_loss": Average validation loss.
            - Additional metrics as available.
    """
    # TODO: Implement validation loop
    raise NotImplementedError("validate() not yet implemented.")
