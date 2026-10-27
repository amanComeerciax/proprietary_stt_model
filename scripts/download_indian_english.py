"""
Download & Ingest Indian English / Code-Switched Speech Dataset (OpenSLR 104).
OpenSLR 104: Hindi-English code-switched & Indian English accent speech dataset.

Downloads:
- Hindi-English_test.tar.gz (443 MB) from OpenSLR 104.
"""

import os
import sys
import csv
import tarfile
import argparse
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np


URL = "https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz"


def download_indian_english(
    raw_dir="data/raw/openslr104",
    output_dir="data/processed/indian_english",
    manifest_dir="data/metadata",
    max_samples=None,
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    manifest_path = Path(manifest_dir)

    raw_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    archive_path = raw_path / "Hindi-English_test.tar.gz"
    extract_folder = raw_path / "extracted_test"

    # 1. Download
    if not extract_folder.exists():
        if not archive_path.exists() or archive_path.stat().st_size < 400 * 1024 * 1024:
            print(f"[*] Downloading Indian English dataset ({URL})...")
            ret = subprocess.run(["curl", "-L", "-C", "-", "-o", str(archive_path), URL])
            if ret.returncode != 0:
                print(f"[!] Download failed with code {ret.returncode}")
                return

        # 2. Extract
        print(f"[*] Extracting {archive_path.name}...")
        extract_folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_folder)
        print(f"[+] Extracted into {extract_folder}")

    # 3. Locate Kaldi segments and text
    transcripts_dir = extract_folder / "test" / "transcripts"
    segments_file = transcripts_dir / "segments"
    text_file = transcripts_dir / "text"

    if not segments_file.exists() or not text_file.exists():
        print(f"[!] Error: segments or text file missing in {transcripts_dir}")
        return

    # Load text mapping: utt_id -> transcript
    print("[*] Reading Kaldi transcripts...")
    text_map = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                text_map[parts[0]] = parts[1]
    print(f"[+] Loaded {len(text_map):,} text entries.")

    # Load segments: utt_id -> (recording_id, start_sec, end_sec)
    print("[*] Reading Kaldi segments...")
    segments = []
    with open(segments_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                utt_id, rec_id, start_s, end_s = parts[0], parts[1], float(parts[2]), float(parts[3])
                if utt_id in text_map:
                    segments.append((utt_id, rec_id, start_s, end_s, text_map[utt_id]))
    print(f"[+] Loaded {len(segments):,} valid segments.")

    # 4. Slice segments from audio files
    audio_dir = extract_folder / "test"
    records = []
    count = 0

    # Cache open audio or read by segment
    print("[*] Slicing and saving audio segments...")
    for utt_id, rec_id, start_s, end_s, transcript in segments:
        if max_samples and count >= max_samples:
            break

        src_wav = audio_dir / f"{rec_id}.wav"
        if not src_wav.exists():
            continue

        duration = round(end_s - start_s, 3)
        if duration < 0.5 or duration > 15.0:
            continue

        try:
            # Sliced read using start/stop frames
            sr = 16000
            start_frame = int(start_s * sr)
            end_frame = int(end_s * sr)
            data, read_sr = sf.read(str(src_wav), start=start_frame, stop=end_frame, dtype="float32")

            if read_sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=read_sr, target_sr=16000)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            out_filename = f"openslr104_{utt_id}.wav"
            out_wav = out_path / out_filename
            sf.write(str(out_wav), data, 16000)

            records.append({
                "audio_path": f"data/processed/indian_english/{out_filename}",
                "transcript": transcript,
                "language": "en",
                "speaker": f"in_spk_{rec_id[:8]}",
                "duration": duration,
            })
            count += 1
            if count % 250 == 0:
                print(f"    -> Processed {count:,} / {len(segments):,} clips ({round(count * duration / 3600, 2)} hrs est)...", end="\r")
        except Exception:
            continue

    if records:
        out_csv = manifest_path / "indian_english.csv"
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
            writer.writeheader()
            writer.writerows(records)

        total_hours = sum(r["duration"] for r in records) / 3600.0
        print(f"\n[+] Created Indian English manifest: {out_csv}")
        print(f"    Total Clips: {len(records):,} | Total Hours: {total_hours:.2f} hrs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and ingest Indian English dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    args = parser.parse_args()

    download_indian_english(max_samples=args.max_samples)

