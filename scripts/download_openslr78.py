"""
Download & Ingest OpenSLR-78 Gujarati Speech Dataset
OpenSLR 78: Crowdsourced high-quality Gujarati multi-speaker speech dataset.

Downloads:
- Female: https://www.openslr.org/resources/78/gu_in_female.zip (~917 MB)
- Male:   https://www.openslr.org/resources/78/gu_in_male.zip (~825 MB)
"""

import os
import sys
import csv
import zipfile
import argparse
import urllib.request
from pathlib import Path
import soundfile as sf
import numpy as np

URLS = {
    "female": "https://www.openslr.org/resources/78/gu_in_female.zip",
    "male": "https://www.openslr.org/resources/78/gu_in_male.zip"
}


def download_progress_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        if count % 200 == 0:
            print(f"    -> Progress: {percent}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)", end="\r")


def download_file(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 500 * 1024 * 1024:
        print(f"[*] File already downloaded: {target_path}")
        return
    print(f"\n[*] Downloading {url} -> {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    ret = subprocess.run(["curl", "-L", "-C", "-", "-o", str(target_path), url])
    if ret.returncode != 0:
        print("[!] curl failed, falling back to urllib...")
        urllib.request.urlretrieve(url, target_path, reporthook=download_progress_hook)
    print(f"\n[+] Downloaded: {target_path}")


def download_and_process_openslr78(
    splits=("female", "male"),
    raw_dir="data/raw/openslr78",
    output_dir="data/processed/openslr78",
    manifest_dir="data/metadata",
    max_samples=None,
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    manifest_path = Path(manifest_dir)

    raw_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    records = []

    for split in splits:
        url = URLS.get(split)
        if not url:
            print(f"[!] Unknown split: {split}")
            continue

        zip_path = raw_path / f"gu_in_{split}.zip"
        extract_folder = raw_path / f"gu_in_{split}"

        # 1. Download ZIP if not exists
        if not extract_folder.exists():
            download_file(url, zip_path)

        # 2. Extract ZIP
        if not extract_folder.exists():
            print(f"[*] Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_folder)
            print(f"[+] Extracted into {extract_folder}")

        # 3. Read index TSV
        # Possible tsv names: line_index_female.tsv, line_index.tsv
        tsv_candidates = list(extract_folder.rglob("*.tsv"))
        if not tsv_candidates:
            tsv_candidates = list(raw_path.rglob(f"*{split}*.tsv"))

        if not tsv_candidates:
            print(f"[!] No TSV index found in {extract_folder}")
            continue

        index_file = tsv_candidates[0]
        print(f"[*] Using index file: {index_file}")

        # Format: FileID \t Text
        count = 0
        with open(index_file, mode="r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                file_id, text = parts[0].strip(), parts[1].strip()
                if not text:
                    continue

                if max_samples and count >= max_samples:
                    break

                # Locate audio file (handle with and without .wav)
                wav_name = file_id if file_id.endswith(".wav") else f"{file_id}.wav"
                candidate_wav = extract_folder / wav_name
                if not candidate_wav.exists():
                    matches = list(extract_folder.rglob(wav_name))
                    if matches:
                        candidate_wav = matches[0]
                    else:
                        continue

                try:
                    data, sr = sf.read(str(candidate_wav))
                    # Ensure mono
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)

                    if sr != 16000:
                        import librosa
                        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    duration = round(len(data) / sr, 3)
                    if duration < 0.5 or duration > 15.0:
                        continue

                    # Save to processed folder
                    out_filename = f"openslr_{split}_{file_id}.wav"
                    out_wav = out_path / out_filename
                    sf.write(str(out_wav), data.astype(np.float32), sr)

                    # Store clean relative path
                    rel_audio_path = f"data/processed/openslr78/{out_filename}"

                    records.append({
                        "audio_path": rel_audio_path,
                        "transcript": text,
                        "language": "gu",
                        "speaker": f"openslr_gu_{split}",
                        "duration": duration,
                    })
                    count += 1
                    if count % 200 == 0:
                        print(f"    -> Processed {count} clips ({round(count * duration / 3600, 2)} hrs est)...", end="\r")
                except Exception as e:
                    continue

        print(f"\n[+] Finished {count} clips from OpenSLR-78 ({split})")

    if records:
        out_csv = manifest_path / "openslr78_gu.csv"
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
            writer.writeheader()
            writer.writerows(records)

        total_hours = sum(r["duration"] for r in records) / 3600.0
        print(f"\n[+] Created OpenSLR-78 manifest: {out_csv}")
        print(f"    Total Clips: {len(records):,} | Total Hours: {total_hours:.2f} hrs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and ingest OpenSLR-78 Gujarati dataset")
    parser.add_argument("--splits", nargs="+", default=["female", "male"], choices=["female", "male"], help="Which splits to download")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    args = parser.parse_args()

    download_and_process_openslr78(splits=args.splits, max_samples=args.max_samples)

