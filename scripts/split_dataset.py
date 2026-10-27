"""
Stratified Train / Val / Test Dataset Splitter.

Splits metadata into:
  - 80% Train (~2,084 files)
  - 10% Validation (~260 files)
  - 10% Test (~261 files)

Stratification ensures:
  - Equal representation of all 4 team members (Amaan, Divya, Deep, Priyanshi)
  - Equal representation of all 3 languages (English, Hindi, Gujarati)
  - Reproducible random seed (42)
"""

import os
import argparse
import pandas as pd


def split_dataset(
    input_csv: str = "data/metadata/metadata_all.csv",
    out_dir: str = "data/metadata",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    print("=" * 60)
    print("Stratified Dataset Splitter (80 / 10 / 10)")
    print("=" * 60)
    print(f"Source metadata: {input_csv}")
    print(f"Output folder  : {out_dir}")

    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"Total samples  : {total}")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Stratify by (speaker, language)
    for (speaker, lang), group in df.groupby(["speaker", "language"]):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in val and test if possible
        if n >= 5:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))

        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])

    # Shuffle final splits
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("✓ SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train set (80%): {len(train_df):4d} files ({len(train_df)/total*100:.1f}%) -> {train_path}")
    print(f"Val set   (10%): {len(val_df):4d} files ({len(val_df)/total*100:.1f}%) -> {val_path}")
    print(f"Test set  (10%): {len(test_df):4d} files ({len(test_df)/total*100:.1f}%) -> {test_path}")

    print("\nSpeaker Representation:")
    sp_df = pd.DataFrame({
        "Train": train_df["speaker"].value_counts(),
        "Val": val_df["speaker"].value_counts(),
        "Test": test_df["speaker"].value_counts(),
    }).fillna(0).astype(int)
    print(sp_df.to_string())

    print("\nLanguage Representation:")
    lang_df = pd.DataFrame({
        "Train": train_df["language"].value_counts(),
        "Val": val_df["language"].value_counts(),
        "Test": test_df["language"].value_counts(),
    }).fillna(0).astype(int)
    print(lang_df.to_string())
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split STT Dataset")
    parser.add_argument("--input", default="data/metadata/metadata_all.csv", help="Input CSV")
    parser.add_argument("--out_dir", default="data/metadata", help="Output directory")
    args = parser.parse_args()

    split_dataset(input_csv=args.input, out_dir=args.out_dir)
