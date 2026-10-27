#!/usr/bin/env python3
"""
Create Colab Iteration 2 Training Package.
Bundles:
1. Manifests: combined_train_v2.csv, combined_val_v2.csv
2. Audio files: 17,609 clean 16kHz WAV files across all 4 processed dirs
3. Code modules: tokenizer, training, model, evaluation, preprocessing, dataset
4. Tokenizer vocab: vocab.json
5. Notebook: train_mms_iteration_2.ipynb

Output: colab_iteration_2.zip
"""

import os
import sys
import time
import zipfile
import pandas as pd
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    train_csv = "data/metadata/combined_train_v2.csv"
    val_csv = "data/metadata/combined_val_v2.csv"
    out_zip = "colab_iteration_2.zip"

    print("=" * 65)
    print("📦 CREATING COLAB ITERATION 2 TRAINING PACKAGE")
    print("=" * 65)

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"[!] Error: {train_csv} or {val_csv} missing!")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    total_df = pd.concat([train_df, val_df], ignore_index=True)

    audio_files = sorted(list(set(total_df["audio_path"].tolist())))
    print(f"[*] Found {len(audio_files):,} unique audio clips to package.")

    # Code directories and files
    code_dirs = [
        "tokenizer",
        "training",
        "model",
        "evaluation",
        "preprocessing",
        "dataset",
    ]
    meta_files = [
        train_csv,
        val_csv,
        "requirements.txt",
        "train_mms_iteration_2.ipynb",
    ]

    t0 = time.time()
    
    # Check if existing zip needs to be removed
    if os.path.exists(out_zip):
        print(f"[*] Removing existing {out_zip}...")
        os.remove(out_zip)

    print(f"[*] Building {out_zip} with fast DEFLATE compression...")

    # We use compresslevel=1 for high speed with good compression
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # 1. Add code files
        print("[*] Adding code modules and configurations...")
        for cdir in code_dirs:
            for dirpath, _, filenames in os.walk(cdir):
                for f in filenames:
                    if f.endswith((".py", ".json", ".md", ".yaml", ".txt")):
                        src_path = os.path.join(dirpath, f)
                        zf.write(src_path, src_path)

        # 2. Add metadata and notebook
        print("[*] Adding metadata manifests and notebook...")
        for mfile in meta_files:
            if os.path.exists(mfile):
                zf.write(mfile, mfile)

        # 3. Add audio files with progress indicator
        print("[*] Adding 17,609 audio files...")
        milestone = len(audio_files) // 10
        for i, afile in enumerate(audio_files):
            if os.path.exists(afile):
                zf.write(afile, afile)
            else:
                print(f"[!] Warning: Missing file {afile}")
            
            if (i + 1) % milestone == 0 or (i + 1) == len(audio_files):
                pct = int((i + 1) / len(audio_files) * 100)
                elapsed = time.time() - t0
                print(f"    - Packaged {i + 1:,}/{len(audio_files):,} audio clips ({pct}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    zip_size_bytes = os.path.getsize(out_zip)
    zip_size_gb = zip_size_bytes / (1024 ** 3)
    zip_size_mb = zip_size_bytes / (1024 ** 2)

    print("\n" + "=" * 65)
    print("✓ COLAB ITERATION 2 PACKAGE READY!")
    print("=" * 65)
    print(f"📁 Output File: {out_zip}")
    print(f"⚖️ Size:        {zip_size_gb:.2f} GB ({zip_size_mb:.1f} MB)")
    print(f"⏱️ Time taken:  {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
