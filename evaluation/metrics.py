"""
Evaluation Metrics — Word Error Rate (WER) and Character Error Rate (CER).

Responsibility: Person 2 (Tokenizer + Text Processing + Evaluation Metrics)
"""

from typing import List


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis transcriptions.

    WER = (Substitutions + Insertions + Deletions) / Number of Reference Words

    Uses the Levenshtein distance at the word level.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model-predicted transcription.

    Returns:
        WER as a float between 0.0 and potentially > 1.0.
    """
    # TODO: Implement WER calculation using edit distance
    raise NotImplementedError("compute_wer() not yet implemented.")


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    CER = (Substitutions + Insertions + Deletions) / Number of Reference Characters

    Uses the Levenshtein distance at the character level.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model-predicted transcription.

    Returns:
        CER as a float between 0.0 and potentially > 1.0.
    """
    # TODO: Implement CER calculation using edit distance
    raise NotImplementedError("compute_cer() not yet implemented.")


def compute_batch_wer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Compute average WER over a batch of reference–hypothesis pairs.

    Args:
        references: List of ground truth transcriptions.
        hypotheses: List of model-predicted transcriptions.

    Returns:
        Average WER across the batch.

    Raises:
        ValueError: If lists have different lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Mismatched lengths: {len(references)} references vs "
            f"{len(hypotheses)} hypotheses."
        )

    # TODO: Implement batch WER
    raise NotImplementedError("compute_batch_wer() not yet implemented.")


def compute_batch_cer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Compute average CER over a batch of reference–hypothesis pairs.

    Args:
        references: List of ground truth transcriptions.
        hypotheses: List of model-predicted transcriptions.

    Returns:
        Average CER across the batch.

    Raises:
        ValueError: If lists have different lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Mismatched lengths: {len(references)} references vs "
            f"{len(hypotheses)} hypotheses."
        )

    # TODO: Implement batch CER
    raise NotImplementedError("compute_batch_cer() not yet implemented.")
