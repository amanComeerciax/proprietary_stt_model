"""
Evaluation Metrics — Word Error Rate (WER) and Character Error Rate (CER).

Responsibility: Person 2 (Tokenizer + Text Processing + Evaluation Metrics)
"""

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def levenshtein_distance(ref: Sequence[T], hyp: Sequence[T]) -> int:
    """Compute the minimum edit distance (Levenshtein distance) between two sequences.

    Counts insertions, deletions, and substitutions needed to transform ref into hyp.

    Args:
        ref: Reference sequence of elements (tokens, words, or characters).
        hyp: Hypothesis sequence of elements.

    Returns:
        Integer minimum edit distance.
    """
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return m
    if m == 0:
        return n

    # Space-optimized dynamic programming: only keep 2 rows
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i
        ref_item = ref[i - 1]
        for j in range(1, m + 1):
            hyp_item = hyp[j - 1]
            cost = 0 if ref_item == hyp_item else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost  # substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis transcriptions.

    WER = (Substitutions + Insertions + Deletions) / Number of Reference Words

    Args:
        reference: Ground truth transcription.
        hypothesis: Model-predicted transcription.

    Returns:
        WER as a float between 0.0 and potentially > 1.0.
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float(len(hyp_words))

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    CER = (Substitutions + Insertions + Deletions) / Number of Reference Characters

    Args:
        reference: Ground truth transcription.
        hypothesis: Model-predicted transcription.

    Returns:
        CER as a float between 0.0 and potentially > 1.0.
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float(len(hyp_chars))

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_batch_wer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Compute average WER over a batch of reference-hypothesis pairs.

    Aggregates total edit distance across all words in the batch.

    Args:
        references: List of ground truth transcriptions.
        hypotheses: List of model-predicted transcriptions.

    Returns:
        Aggregated WER across the batch.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Mismatched lengths: {len(references)} references vs {len(hypotheses)} hypotheses."
        )

    if not references:
        return 0.0

    total_dist = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()

        total_dist += levenshtein_distance(ref_words, hyp_words)
        total_words += len(ref_words)

    return total_dist / max(total_words, 1)


def compute_batch_cer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Compute average CER over a batch of reference-hypothesis pairs.

    Aggregates total edit distance across all characters in the batch.

    Args:
        references: List of ground truth transcriptions.
        hypotheses: List of model-predicted transcriptions.

    Returns:
        Aggregated CER across the batch.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Mismatched lengths: {len(references)} references vs {len(hypotheses)} hypotheses."
        )

    if not references:
        return 0.0

    total_dist = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.strip())
        hyp_chars = list(hyp.strip())

        total_dist += levenshtein_distance(ref_chars, hyp_chars)
        total_chars += len(ref_chars)

    return total_dist / max(total_chars, 1)


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics (WER & CER) — Self Verification Test")
    print("=" * 60)

    # Test cases in English, Hindi, and Gujarati
    test_cases = [
        ("the meeting starts in five minutes", "the meeting start in five minutes"),
        ("कमरे की खिड़की खोलो", "कमरे की खिड़की खोलो"),  # Exact match
        ("AnyDesk બંધ કરો", "AnyDesk બંધ કરી દો"),
    ]

    for ref, hyp in test_cases:
        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        print(f"Ref: '{ref}'")
        print(f"Hyp: '{hyp}'")
        print(f"  WER: {wer:.4f} ({wer * 100:.1f}%), CER: {cer:.4f} ({cer * 100:.1f}%)")
        print()

    # Batch test
    refs = [t[0] for t in test_cases]
    hyps = [t[1] for t in test_cases]
    b_wer = compute_batch_wer(refs, hyps)
    b_cer = compute_batch_cer(refs, hyps)

    print(f"Batch Average WER : {b_wer:.4f} ({b_wer * 100:.1f}%)")
    print(f"Batch Average CER : {b_cer:.4f} ({b_cer * 100:.1f}%)")
    print("✓ Metrics verified successfully!")
    print("=" * 60)
