"""
Feature Extractor — Computes spectrograms and acoustic features from waveforms.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from typing import Optional

import numpy as np


class FeatureExtractor:
    """Extracts Mel spectrograms, MFCCs, and other acoustic features.

    Attributes:
        n_mels: Number of Mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length for STFT.
        f_min: Minimum frequency for Mel filter bank.
        f_max: Maximum frequency for Mel filter bank.
        normalize: Whether to normalize features.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

    def compute_mel_spectrogram(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Compute log-Mel spectrogram from a waveform.

        Args:
            waveform: Audio waveform as numpy array.
            sample_rate: Sample rate of the waveform.

        Returns:
            Log-Mel spectrogram of shape (n_mels, time_steps).
        """
        # TODO: Implement using torchaudio or librosa
        raise NotImplementedError(
            "FeatureExtractor.compute_mel_spectrogram() not yet implemented."
        )

    def compute_mfcc(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        n_mfcc: int = 13,
    ) -> np.ndarray:
        """Compute MFCCs from a waveform.

        Args:
            waveform: Audio waveform as numpy array.
            sample_rate: Sample rate of the waveform.
            n_mfcc: Number of MFCC coefficients to extract.

        Returns:
            MFCC features of shape (n_mfcc, time_steps).
        """
        # TODO: Implement MFCC extraction
        raise NotImplementedError(
            "FeatureExtractor.compute_mfcc() not yet implemented."
        )

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply per-channel mean and variance normalization.

        Args:
            features: Feature matrix of shape (channels, time_steps).

        Returns:
            Normalized feature matrix.
        """
        # TODO: Implement CMVN (Cepstral Mean and Variance Normalization)
        raise NotImplementedError(
            "FeatureExtractor.normalize_features() not yet implemented."
        )
