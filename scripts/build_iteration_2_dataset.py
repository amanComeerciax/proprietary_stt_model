"""
Build Iteration 2 Multilingual Training Package for Google Colab.
Merges:
1. Team dataset + FLEURS (Existing combined_train/val)
2. OpenSLR-78 Gujarati (16-18 hours)
3. OpenSLR-104 Indian English (4.7 hours)

Generates:
- data/metadata/combined_train_v2.csv
- data/metadata/combined_val_v2.csv
- colab_package_iteration2.zip
"""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd


def build_iteration_2_dataset(seed: int = 42):
    print("=" * 65)
    print("🚀 BUILDING ITERATION 2 MULTILINGUAL DATASET & COLAB PACKAGE")
    print("=" * 65)

    dfs = []

    # 1. Base combined dataset (Team + FLEURS)
    base_train = Path("data/metadata/combined_train.csv")
    base_val = Path("data/metadata/combined_val.csv")
    for p in [base_train, base_val]:
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"[+] Loaded base manifest: {p.name} ({len(df):,} clips)")

    # 2. Indian English (OpenSLR 104)
    in_en_csv = Path("data/metadata/indian_english.csv")
    if in_en_csv.exists():
        df = pd.read_csv(in_en_csv)
        dfs.append(df)
        print(f"[+] Loaded Indian English manifest: {in_en_csv.name} ({len(df):,} clips)")

    # 3. OpenSLR-78 Gujarati
    openslr_csv = Path("data/metadata/openslr78_gu.csv")
    if openslr_csv.exists():
        df = pd.read_csv(openslr_csv)
        dfs.append(df)
        print(f"[+] Loaded OpenSLR-78 Gujarati manifest: {openslr_csv.name} ({len(df):,} clips)")
    else:
        print("[!] Note: openslr78_gu.csv not ready yet or still processing.")

    if not dfs:
        print("[!] No manifests found to merge!")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["transcript", "audio_path"])
    full_df["transcript"] = full_df["transcript"].astype(str).str.strip()
    full_df = full_df[full_df["transcript"].str.len() > 0]

    # Verify audio file existence
    print("\n[*] Verifying audio files on disk...")
    valid_mask = full_df["audio_path"].apply(lambda p: Path(p).exists())
    valid_df = full_df[valid_mask].copy()

    total_clips = len(valid_df)
    total_hours = valid_df["duration"].sum() / 3600.0 if "duration" in valid_df.columns else 0.0

    print(f"\n" + "=" * 55)
    print("📊 MERGED DATASET STATISTICS (ITERATION 2):")
    print(f"   Total Audio Clips : {total_clips:,}")
    print(f"   Total Audio Hours : {total_hours:.2f} hrs")
    print("\nLanguage breakdown:")
    for lang, count in valid_df["language"].value_counts().items():
        sub_hrs = valid_df[valid_df["language"] == lang]["duration"].sum() / 3600.0 if "duration" in valid_df.columns else 0
        print(f"   - {lang.upper():4s}: {count:,} clips ({sub_hrs:.2f} hrs)")
    print("=" * 55 + "\n")

    # Stratified 90% Train / 10% Validation Split
    print("[*] Performing 90% Train / 10% Validation split...")
    train_dfs = []
    val_dfs = []

    for lang, group in valid_df.groupby("language"):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(round(n * 0.9))
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:])

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    out_train = Path("data/metadata/combined_train_v2.csv")
    out_val = Path("data/metadata/combined_val_v2.csv")

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print(f"[+] Saved Train split: {out_train} ({len(train_df):,} clips)")
    print(f"[+] Saved Val split  : {out_val} ({len(val_df):,} clips)")


if __name__ == "__main__":
    build_iteration_2_dataset()
