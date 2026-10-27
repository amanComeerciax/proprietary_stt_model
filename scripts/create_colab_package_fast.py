#!/usr/bin/env python3
"""
Create Fast Lightweight Colab Package (~210 MB).
Instead of uploading 3.1 GB over slow home internet, packages:
1. Team's proprietary speech audio (data/processed/all_speakers/) ~200MB
2. All code modules (training, model, tokenizer, scripts, evaluation, dataset)
3. Metadata and requirements

Colab's 1 Gbps Google Cloud connection downloads the public datasets (FLEURS, OpenSLR)
in just 2 minutes directly in the cloud.
"""

import os
import sys
import time
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    out_zip = "colab_iteration_2_fast.zip"
    if os.path.exists(out_zip):
        os.remove(out_zip)

    print("=" * 65)
    print("⚡ CREATING LIGHTWEIGHT FAST COLAB PACKAGE (~210 MB)")
    print("=" * 65)

    include_dirs = [
        "tokenizer",
        "training",
        "model",
        "evaluation",
        "preprocessing",
        "dataset",
        "scripts",
        "data/metadata",
        "data/processed/all_speakers",
    ]

    t0 = time.time()
    total_files = 0

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for item in include_dirs:
            if not os.path.exists(item):
                print(f"[!] Warning: {item} does not exist!")
                continue
            print(f"[*] Packaging {item}...")
            if os.path.isfile(item):
                zf.write(item, item)
                total_files += 1
            else:
                for dirpath, _, filenames in os.walk(item):
                    for f in filenames:
                        if not f.startswith(".") and not f.endswith((".pyc", ".DS_Store")):
                            full = os.path.join(dirpath, f)
                            zf.write(full, full)
                            total_files += 1

        for extra in ["requirements.txt", "train_mms_iteration_2.ipynb"]:
            if os.path.exists(extra):
                zf.write(extra, extra)
                total_files += 1

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_zip) / (1024 ** 2)

    print("\n" + "=" * 65)
    print(f"✓ Created {out_zip}!")
    print(f"⚖️ Size:       {size_mb:.1f} MB (vs 3,100 MB earlier — 93% smaller!)")
    print(f"📁 Files:      {total_files:,}")
    print(f"⏱️ Time taken: {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
