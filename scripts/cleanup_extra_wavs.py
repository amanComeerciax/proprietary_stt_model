import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

metadata_path = PROJECT_ROOT / "data" / "metadata" / "metadata_chunks.csv"
chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks"

# 1. Get list of valid audio paths from the 52-row CSV
valid_paths = set()
with open(metadata_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # absolute path
        abs_path = PROJECT_ROOT / row["audio_path"]
        valid_paths.add(str(abs_path.resolve()))

# 2. Iterate through all wav files in chunks directory and delete extras
deleted_count = 0
for root, _, files in os.walk(chunks_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_path = Path(root) / file
            if str(wav_path.resolve()) not in valid_paths:
                os.remove(wav_path)
                deleted_count += 1
                # print(f"Deleted extra file: {file}")

print(f"Cleanup complete! Deleted {deleted_count} extra/garbage voice files.")
