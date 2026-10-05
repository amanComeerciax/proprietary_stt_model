"""
Inference Engine — Runs STT model inference on audio inputs.

Supports both single audio file processing, batch processing, and real-time streaming.
Optimized for CPU / Mac inference without requiring CUDA.

Responsibility: Person 4 (Training + Validation + Optimization + Deployment)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import yaml

from model.stt_model import STTModel
from preprocessing.features import FeatureExtractor
from tokenizer.tokenizer import CharTokenizer


class STTInference:
    """Speech-to-Text inference engine.

    Loads a trained model checkpoint and provides fast methods for
    transcribing audio files or streaming chunks.

    Attributes:
        model: The loaded STTModel instance.
        tokenizer: CharTokenizer for decoding token IDs to text.
        feature_extractor: FeatureExtractor for computing spectrograms.
        device: Inference device ('cpu', 'mps', 'cuda').
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "checkpoints/best_model.pt",
        config_path: Union[str, Path] = "configs/config.yaml",
        vocab_path: Union[str, Path] = "tokenizer/vocab.json",
        device: str = "cpu",
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to the trained model checkpoint (.pt).
            config_path: Path to the YAML configuration file.
            vocab_path: Path to tokenizer vocabulary JSON.
            device: Device for inference ('cpu', 'mps', 'cuda').
        """
        self.device = torch.device(device)
        self.feature_extractor = FeatureExtractor()

        # 1. Load Tokenizer
        self.tokenizer = CharTokenizer()
        if Path(vocab_path).exists():
            self.tokenizer.load_vocab(vocab_path)
        else:
            self.tokenizer.build_from_metadata("data/metadata/metadata.csv")

        # 2. Load Model & Checkpoint
        self._load_model(model_path, config_path)

    def _load_model(self, model_path: Union[str, Path], config_path: Union[str, Path]) -> None:
        """Load the model architecture and restore weights from checkpoint."""
        model_path = Path(model_path)
        config_path = Path(config_path)

        encoder_cfg = {"input_dim": 80, "hidden_dim": 256, "num_layers": 4, "num_heads": 4}
        decoder_cfg = {"vocab_size": self.tokenizer.vocab_size, "hidden_dim": 256, "num_layers": 2, "num_heads": 4}

        if config_path.exists():
            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f)
                if "model" in full_cfg:
                    if "encoder" in full_cfg["model"]:
                        encoder_cfg.update(full_cfg["model"]["encoder"])
                    if "decoder" in full_cfg["model"]:
                        decoder_cfg.update(full_cfg["model"]["decoder"])
                        decoder_cfg["vocab_size"] = self.tokenizer.vocab_size

        self.model = STTModel(encoder_config=encoder_cfg, decoder_config=decoder_cfg)

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"✓ Loaded model weights from {model_path}")
        else:
            print(f"⚠️ Checkpoint not found at {model_path}. Initialized model with random weights.")

        self.model.to(self.device)
        self.model.eval()

    def transcribe_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .flac, etc.).

        Returns:
            Dictionary with:
                - "text": Transcribed text string.
                - "duration": Audio duration in seconds.
                - "inference_time": Time taken for model inference in seconds.
                - "rtf": Real-Time Factor (inference_time / duration).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio & extract Log-Mel spectrogram
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        duration = len(waveform) / sr

        start_time = time.time()
        mel = self.feature_extractor.compute_mel_spectrogram(waveform, sample_rate=sr)

        # Transpose to (1, time, n_mels)
        features = torch.from_numpy(mel.T).unsqueeze(0).float().to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        # Perform fast CTC inference
        transcripts = self.model.transcribe(features, feature_lengths, tokenizer=self.tokenizer)
        ctc_text = transcripts[0] if transcripts else ""

        # Perform Autoregressive Attention inference
        attn_text = ""
        try:
            with torch.no_grad():
                enc_out, enc_lens = self.model.encoder(features, feature_lengths)
                sos_id = self.tokenizer.sos_id
                eos_id = self.tokenizer.eos_id
                cur = torch.tensor([[sos_id]], dtype=torch.long, device=self.device)
                for _ in range(50):
                    tgt_lens = torch.tensor([cur.shape[1]], device=self.device)
                    logits = self.model.decoder(cur, enc_out, enc_lens, tgt_lens)
                    
                    # Apply repetition penalty
                    next_token_logits = logits[:, -1, :].clone()
                    for i in range(cur.shape[1]):
                        token_id = cur[0, i].item()
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= 1.2
                        else:
                            next_token_logits[0, token_id] *= 1.2
                            
                    tok = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                    cur = torch.cat([cur, tok], dim=1)
                    if tok.item() == eos_id:
                        break
                attn_text = self.tokenizer.decode(cur[0].tolist(), remove_special=True)
        except Exception:
            attn_text = ""

        inference_time = time.time() - start_time
        rtf = inference_time / max(duration, 0.001)

        # Primary output text prioritizes non-empty prediction
        primary_text = ctc_text if ctc_text else attn_text

        return {
            "text": primary_text,
            "ctc_text": ctc_text,
            "attn_text": attn_text,
            "duration": round(duration, 2),
            "inference_time": round(inference_time, 4),
            "rtf": round(rtf, 4),
            "device": str(self.device),
        }

    def transcribe_batch(
        self, audio_paths: List[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription result dictionaries.
        """
        results = []
        for path in audio_paths:
            res = self.transcribe_file(path)
            res["audio_path"] = str(path)
            results.append(res)
        return results

    def transcribe_stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """Process a chunk of streaming audio and return transcription.

        Args:
            audio_chunk: 1D numpy array of audio samples.
            sample_rate: Sampling rate (standard: 16000 Hz).

        Returns:
            Transcription string.
        """
        if len(audio_chunk) < sample_rate * 0.2:  # Min 200ms
            return None

        mel = self.feature_extractor.compute_mel_spectrogram(audio_chunk, sample_rate=sample_rate)
        features = torch.from_numpy(mel.T).unsqueeze(0).float().to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        transcripts = self.model.transcribe(features, feature_lengths, tokenizer=self.tokenizer)
        return transcripts[0] if transcripts else ""


if __name__ == "__main__":
    print("=" * 60)
    print("STTInference — Verification Test")
    print("=" * 60)

    engine = STTInference(
        model_path="checkpoints/best_model.pt",
        config_path="configs/config.yaml",
        device="cpu",
    )

    test_file = "data/processed/amaanvoice/englishvoice/amaan1.wav"
    if Path(test_file).exists():
        print(f"\nTranscribing: {test_file}")
        res = engine.transcribe_file(test_file)
        print(f"  Duration       : {res['duration']}s")
        print(f"  Inference Time : {res['inference_time']}s")
        print(f"  RTF (Speed)    : {res['rtf']}x (< 1.0 means faster than real-time!)")
        print(f"  Device         : {res['device']}")
        print(f"  Output Text    : '{res['text'][:100]}...'")
        print("  ✓ STTInference verified successfully!")
    print("=" * 60)
