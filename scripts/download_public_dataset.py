"""
Download & Ingest Google FLEURS Speech Dataset for Multi-Language STT
Languages: Gujarati (gu_in), Hindi (hi_in), English (en_us)

Extracts 16kHz mono WAV audio clips and builds CSV manifests aligned with
the proprietary STT model schema:
    audio_path,transcript,language,speaker,duration
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np

# Mapping of FLEURS configs to our model language tags
LANG_MAP = {
    "gu_in": "gu",
    "hi_in": "hi",
    "en_us": "en",
    "en_in": "en"
}


def download_and_process_fleurs(
    languages=("gu_in", "hi_in", "en_us"),
    splits=("train", "validation"),
    max_samples_per_lang=None,
    output_dir="data/processed/external_fleurs",
    manifest_dir="data/metadata",
    merge_with_team=True,
):
    from datasets import load_dataset

    output_path = Path(output_dir)
    manifest_path = Path(manifest_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    records_by_split = {split: [] for split in splits}

    for lang_config in languages:
        lang_code = LANG_MAP.get(lang_config, lang_config[:2])
        print(f"\n==================================================")
        print(f"[*] Processing Google FLEURS: {lang_config} (-> '{lang_code}')")
        print(f"==================================================")

        for split in splits:
            print(f"[*] Loading split '{split}' for {lang_config}...")
            try:
                from datasets import Audio
                ds = load_dataset("google/fleurs", lang_config, split=split)
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as e:
                print(f"[!] Error loading {lang_config} ({split}): {e}")
                continue

            total_available = len(ds)
            target_count = total_available if max_samples_per_lang is None else min(total_available, max_samples_per_lang)
            print(f"[*] Found {total_available} samples. Extracting {target_count} samples...")

            lang_out_dir = output_path / lang_code / split
            lang_out_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for idx, item in enumerate(ds):
                if count >= target_count:
                    break

                try:
                    import io
                    audio_info = item["audio"]
                    audio_bytes = audio_info.get("bytes")
                    transcription = (item.get("transcription") or "").strip()

                    if not transcription or not audio_bytes:
                        continue

                    raw_audio, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(raw_audio) == 0:
                        continue

                    # Resample to 16kHz if needed (FLEURS is 16kHz)
                    if sr != 16000:
                        import librosa
                        raw_audio = librosa.resample(raw_audio, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    duration = round(len(raw_audio) / sr, 3)

                    # Filter out outlier clips (< 0.5s or > 15s)
                    if duration < 0.5 or duration > 15.0:
                        continue

                    # Save WAV file
                    wav_filename = f"fleurs_{lang_code}_{split}_{idx:05d}.wav"
                    wav_filepath = lang_out_dir / wav_filename

                    sf.write(str(wav_filepath), raw_audio.astype(np.float32), sr)

                    # Record metadata
                    rel_audio_path = str(wav_filepath)
                    speaker_id = f"fleurs_{lang_code}_{item.get('speaker_id', 'spk')}"

                    records_by_split[split].append({
                        "audio_path": rel_audio_path,
                        "transcript": transcription,
                        "language": lang_code,
                        "speaker": speaker_id,
                        "duration": duration,
                    })
                    count += 1

                    if count % 200 == 0:
                        print(f"    -> Extracted {count}/{target_count} clips ({round(count * duration / 3600, 2)} hrs est)...")

                except Exception as ex:
                    print(f"[!] Warning: error processing sample {idx}: {ex}")
                    continue

            print(f"[+] Successfully extracted {count} clips for {lang_config} [{split}]")

    # Write manifests
    fleurs_manifests = {}
    for split, records in records_by_split.items():
        if not records:
            continue
        out_csv = manifest_path / f"fleurs_{split}.csv"
        fleurs_manifests[split] = out_csv
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
            writer.writeheader()
            writer.writerows(records)

        total_hours = sum(r["duration"] for r in records) / 3600.0
        print(f"\n[+] Created manifest: {out_csv}")
        print(f"    Total Samples: {len(records)} | Total Duration: {total_hours:.2f} hours")

    # Merge with team dataset if requested
    if merge_with_team:
        print("\n==================================================")
        print("[*] Merging with existing team datasets...")
        print("==================================================")
        merge_manifests(
            team_train_csv=manifest_path / "train.csv",
            team_val_csv=manifest_path / "val.csv",
            fleurs_train_csv=fleurs_manifests.get("train"),
            fleurs_val_csv=fleurs_manifests.get("validation"),
            out_combined_train=manifest_path / "combined_train.csv",
            out_combined_val=manifest_path / "combined_val.csv",
        )


def merge_manifests(
    team_train_csv,
    team_val_csv,
    fleurs_train_csv,
    fleurs_val_csv,
    out_combined_train,
    out_combined_val,
):
    def read_csv(path):
        if not path or not os.path.exists(path):
            return []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def write_csv(path, rows):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
            writer.writeheader()
            writer.writerows(rows)

    team_train = read_csv(team_train_csv)
    team_val = read_csv(team_val_csv)
    fleurs_train = read_csv(fleurs_train_csv)
    fleurs_val = read_csv(fleurs_val_csv)

    combined_train = team_train + fleurs_train
    combined_val = team_val + fleurs_val

    write_csv(out_combined_train, combined_train)
    write_csv(out_combined_val, combined_val)

    def print_summary(name, rows):
        total_sec = sum(float(r.get("duration", 0)) for r in rows)
        hours = total_sec / 3600.0
        by_lang = {}
        for r in rows:
            lang = r.get("language", "unknown")
            by_lang[lang] = by_lang.get(lang, 0) + 1
        print(f"\n[+] {name}:")
        print(f"    Total Samples: {len(rows):,}")
        print(f"    Total Hours:   {hours:.2f} hrs")
        print(f"    Breakdown by Language: {by_lang}")

    print_summary("Combined TRAIN Dataset", combined_train)
    print_summary("Combined VAL Dataset", combined_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and ingest Google FLEURS dataset")
    parser.add_argument("--languages", nargs="+", default=["gu_in", "hi_in", "en_us"], help="FLEURS language configs")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Dataset splits")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per language (default: all)")
    parser.add_argument("--output_dir", default="data/processed/external_fleurs", help="Output directory for audio")
    parser.add_argument("--manifest_dir", default="data/metadata", help="Output directory for manifests")
    parser.add_argument("--no_merge", action="store_true", help="Do not merge with team dataset")

    args = parser.parse_args()

    download_and_process_fleurs(
        languages=args.languages,
        splits=args.splits,
        max_samples_per_lang=args.max_samples,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        merge_with_team=not args.no_merge,
    )
