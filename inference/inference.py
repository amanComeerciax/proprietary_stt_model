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
        model_path: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path] = "configs/config.yaml",
        vocab_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to trained checkpoint (.pt) or HuggingFace directory.
            config_path: Path to the YAML configuration file.
            vocab_path: Path to tokenizer vocabulary JSON.
            device: Device for inference ('cpu', 'mps', 'cuda').
        """
        self.device = torch.device(device)
        self.feature_extractor = FeatureExtractor()

        # Auto-select best available model if none provided
        if model_path is None:
            if Path("checkpoints/stt_w2v/best").exists():
                model_path = "checkpoints/stt_w2v/best"
            else:
                model_path = "checkpoints/best_model.pt"

        self.model_path = Path(model_path)
        self.is_w2v = self.model_path.is_dir() and (
            (self.model_path / "model.safetensors").exists()
            or (self.model_path / "config.json").exists()
        )

        # 1. Load Tokenizer
        self.tokenizer = CharTokenizer()
        if vocab_path is None:
            if self.is_w2v and (self.model_path / "vocab.json").exists():
                vocab_path = self.model_path / "vocab.json"
            else:
                vocab_path = "tokenizer/vocab.json"

        if Path(vocab_path).exists():
            self.tokenizer.load_vocab(str(vocab_path))
        else:
            self.tokenizer.build_from_metadata("data/metadata/metadata.csv")

        # 2. Load Model Architecture & Checkpoint
        if self.is_w2v:
            from transformers import Wav2Vec2ForCTC
            self.w2v_model = Wav2Vec2ForCTC.from_pretrained(str(self.model_path)).to(self.device)
            self.w2v_model.eval()
            self.model = None
            print(f"✓ Loaded fine-tuned MMS-300M model from {self.model_path}")
        else:
            self.w2v_model = None
            self._load_scratch_model(self.model_path, config_path)

        # 3. Optional KenLM Beam Search Decoder
        self.lm_decoder = None
        lm_path = Path("data/lm/multilingual_3gram.arpa")
        unigrams_path = Path("data/lm/unigrams.txt")
        if lm_path.exists() and self.is_w2v:
            try:
                from pyctcdecode import build_ctcdecoder
                labels = [self.tokenizer.id_to_char.get(i, "") for i in range(self.tokenizer.vocab_size)]
                labels[self.tokenizer.blank_id] = ""
                # Replace multi-character special tokens (<PAD>, <SOS>, <EOS>, <UNK>) with single private chars
                for i, spec in enumerate(["<PAD>", "<SOS>", "<EOS>", "<UNK>"], start=1):
                    if i < len(labels) and labels[i] == spec:
                        labels[i] = chr(0xE000 + i)

                unigrams = None
                if unigrams_path.exists():
                    with open(unigrams_path, "r", encoding="utf-8") as f:
                        unigrams = [line.strip() for line in f if line.strip()]

                self.lm_decoder = build_ctcdecoder(
                    labels=labels,
                    kenlm_model_path=str(lm_path),
                    unigrams=unigrams,
                    alpha=0.55,
                    beta=1.2,
                )
                print(f"✓ Initialized KenLM 3-Gram Beam Search Decoder from {lm_path}")
            except Exception as e:
                print(f"[!] Warning: Could not initialize KenLM decoder: {e}")

    def _load_scratch_model(self, model_path: Path, config_path: Union[str, Path]) -> None:
        """Load the scratch model architecture and restore weights from checkpoint."""
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
            print(f"✓ Loaded scratch model weights from {model_path}")
        else:
            print(f"⚠️ Checkpoint not found at {model_path}. Initialized model with random weights.")

        self.model.to(self.device)
        self.model.eval()

    def transcribe_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .flac, etc.).

        Returns:
            Dictionary with transcription results, latency, and duration.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        duration = len(waveform) / sr

        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        start_time = time.time()

        if self.is_w2v:
            # 1. Resample to 16kHz
            if sr != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                sr = 16000

            # 2. Voice Activity Detection (VAD) / Silence Trimming
            # Cuts out leading/trailing dead silence (>25dB below peak energy)
            try:
                import librosa
                trimmed, _ = librosa.effects.trim(waveform, top_db=25)
                # Keep trimmed only if at least 0.2s of audio remains
                if len(trimmed) >= 3200:
                    waveform = trimmed
            except Exception:
                pass

            # 3. Auto-Gain Normalization (Normalize peak amplitude to studio level 0.95)
            max_val = np.max(np.abs(waveform))
            if max_val > 1e-4:
                waveform = (waveform / max_val) * 0.95

            # 4. Standardize audio (zero mean, unit variance) matching Wav2Vec2 training
            waveform = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)

            inp = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.w2v_model(inp).logits

            logits_np = logits[0].cpu().numpy()
            pred_ids = logits_np.argmax(-1).tolist()
            raw_text = self.tokenizer.decode_ctc(pred_ids)

            # 5. KenLM Beam Search decoding with width=50
            lm_text = ""
            if self.lm_decoder is not None:
                try:
                    lm_text = self.lm_decoder.decode(logits_np, beam_width=50)
                except Exception:
                    lm_text = raw_text
            else:
                lm_text = raw_text

            primary_text = lm_text if lm_text.strip() else raw_text

            inference_time = time.time() - start_time
            rtf = inference_time / max(duration, 0.001)

            return {
                "text": primary_text,
                "ctc_text": raw_text,
                "lm_text": lm_text,
                "attn_text": lm_text,
                "duration": round(duration, 2),
                "inference_time": round(inference_time, 4),
                "rtf": round(rtf, 4),
                "device": str(self.device),
                "model_name": "MMS-300M + KenLM 3-Gram LM",
            }

        # Otherwise use Scratch STTModel
        mel = self.feature_extractor.compute_mel_spectrogram(waveform, sample_rate=sr)
        features = torch.from_numpy(mel.T).unsqueeze(0).float().to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        # Fast CTC inference
        transcripts = self.model.transcribe(features, feature_lengths, tokenizer=self.tokenizer)
        ctc_text = transcripts[0] if transcripts else ""

        # Autoregressive Attention inference
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
        primary_text = ctc_text if ctc_text else attn_text

        return {
            "text": primary_text,
            "ctc_text": ctc_text,
            "attn_text": attn_text,
            "duration": round(duration, 2),
            "inference_time": round(inference_time, 4),
            "rtf": round(rtf, 4),
            "device": str(self.device),
            "model_name": "Proprietary Transformer + CTC",
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
