"""
Audio Processor — Handles audio loading, resampling, and normalization.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class AudioProcessor:
    """Loads, resamples, normalizes, and validates raw audio files.

    Attributes:
        sample_rate: Target sample rate for all audio (default 16000 Hz).
        mono: Whether to convert all audio to mono.
        max_duration: Maximum allowed duration in seconds.
        min_duration: Minimum allowed duration in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.max_duration = max_duration
        self.min_duration = min_duration

    def load(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load an audio file and return (waveform, sample_rate).

        Args:
            file_path: Path to the audio file (.wav, .mp3, .flac).

        Returns:
            Tuple of (waveform as numpy array, sample rate).

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If audio duration is out of allowed bounds.
        """
        # TODO: Implement audio loading with torchaudio / soundfile
        raise NotImplementedError("AudioProcessor.load() not yet implemented.")

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample waveform to the target sample rate.

        Args:
            waveform: Input audio waveform.
            orig_sr: Original sample rate of the waveform.

        Returns:
            Resampled waveform as numpy array.
        """
        # TODO: Implement resampling logic
        raise NotImplementedError("AudioProcessor.resample() not yet implemented.")

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform amplitude to [-1.0, 1.0].

        Args:
            waveform: Input audio waveform.

        Returns:
            Normalized waveform.
        """
        # TODO: Implement peak / RMS normalization
        raise NotImplementedError("AudioProcessor.normalize() not yet implemented.")

    def validate(self, waveform: np.ndarray, sample_rate: int) -> bool:
        """Check if audio meets duration and quality constraints.

        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate of the waveform.

        Returns:
            True if the audio is valid, False otherwise.
        """
        duration = len(waveform) / sample_rate
        return self.min_duration <= duration <= self.max_duration

    def trim_silence(
        self, waveform: np.ndarray, top_db: float = 20.0
    ) -> np.ndarray:
        """Trim leading and trailing silence from the waveform.

        Args:
            waveform: Input audio waveform.
            top_db: Threshold (in dB) below peak to consider as silence.

        Returns:
            Trimmed waveform.
        """
        # TODO: Implement silence trimming with librosa
        raise NotImplementedError("AudioProcessor.trim_silence() not yet implemented.")
