"""
Dataset Segmentation Pipeline — Splits 60s recordings into clean 2-8s sentence chunks.

Transforms 36 long recordings into ~400+ high-quality sentence-level audio samples.
Enables fast CTC alignment and rapid convergence on CPU.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import whisper

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def clean_text(text: str) -> str:
    """Normalize text and remove duplicate whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_dataset(
    manifest_path: str = "data/metadata/metadata.csv",
    output_dir: str = "data/processed/chunks",
    output_manifest: str = "data/metadata/metadata_chunks.csv",
    whisper_model_name: str = "base",
) -> List[Dict[str, Any]]:
    """Segment all recordings in the dataset into sentence chunks."""
    manifest_file = PROJECT_ROOT / manifest_path
    chunks_base_dir = PROJECT_ROOT / output_dir
    manifest_out_file = PROJECT_ROOT / output_manifest

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    print("=" * 65)
    print("🚀 STT Audio Segmentation Pipeline")
    print("=" * 65)
    print(f"Loading Whisper model '{whisper_model_name}' on CPU for alignment...")
    model = whisper.load_model(whisper_model_name, device="cpu")
    print("✓ Model loaded successfully!")

    with open(manifest_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} recordings in {manifest_path}.\n")

    chunk_records: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        rel_audio_path = row["audio_path"]
        full_audio_path = PROJECT_ROOT / rel_audio_path
        language = row.get("language", "en")
        speaker = row.get("speaker", "amaan")
        gt_transcript = row.get("transcript", "")

        if not full_audio_path.exists():
            print(f"[{idx}/{len(rows)}] ⚠️ File missing: {rel_audio_path}, skipping.")
            continue

        print(f"[{idx:02d}/{len(rows):02d}] Processing {rel_audio_path} ({language})...")

        # Load full waveform
        audio_data, sr = sf.read(str(full_audio_path), dtype="float32")
        total_dur = len(audio_data) / sr

        # Run whisper segmentation
        res = model.transcribe(str(full_audio_path), language=language)
        segments = res.get("segments", [])

        # Create language subdirectory for chunks
        lang_chunk_dir = chunks_base_dir / language
        lang_chunk_dir.mkdir(parents=True, exist_ok=True)
        file_stem = full_audio_path.stem

        created_chunks = 0
        for s_idx, seg in enumerate(segments, 1):
            start_s = max(0.0, seg["start"] - 0.05)
            end_s = min(total_dur, seg["end"] + 0.05)
            duration = end_s - start_s

            # Skip tiny noise blips (< 0.5s)
            if duration < 0.5:
                continue

            seg_text = clean_text(seg.get("text", ""))
            if not seg_text:
                continue

            # Slice waveform
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            chunk_waveform = audio_data[start_sample:end_sample]

            if len(chunk_waveform) == 0:
                continue

            chunk_filename = f"{file_stem}_seg_{s_idx:03d}.wav"
            chunk_full_path = lang_chunk_dir / chunk_filename
            chunk_rel_path = str(chunk_full_path.relative_to(PROJECT_ROOT))

            # Save 16kHz WAV chunk
            sf.write(str(chunk_full_path), chunk_waveform, sr)

            chunk_records.append({
                "audio_path": chunk_rel_path,
                "transcript": seg_text,
                "language": language,
                "speaker": speaker,
                "duration": round(duration, 2),
            })
            created_chunks += 1

        print(f"      ↳ Created {created_chunks} chunks (Duration: {total_dur:.1f}s)")

    # Write output manifest
    manifest_out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"]
        )
        writer.writeheader()
        writer.writerows(chunk_records)

    print("\n" + "=" * 65)
    print(f"🎉 Segmentation Complete!")
    print(f"Total chunks created: {len(chunk_records)}")
    print(f"Saved chunk manifest to: {manifest_out_file}")

    # Print language breakdown
    en_cnt = sum(1 for c in chunk_records if c["language"] == "en")
    hi_cnt = sum(1 for c in chunk_records if c["language"] == "hi")
    gu_cnt = sum(1 for c in chunk_records if c["language"] == "gu")
    print(f"  - English  : {en_cnt} clips")
    print(f"  - Hindi    : {hi_cnt} clips")
    print(f"  - Gujarati : {gu_cnt} clips")
    print("=" * 65)

    return chunk_records


if __name__ == "__main__":
    segment_dataset()
