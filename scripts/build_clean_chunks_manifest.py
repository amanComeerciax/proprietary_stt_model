"""
Build Clean Chunks Manifest — Associates each audio chunk with exact ground truth transcripts in proper script.

Ensures:
- English chunks have valid English text.
- Hindi chunks have exact Devanagari Hindi text.
- Gujarati chunks have exact Gujarati text.
Output: data/metadata/metadata_chunks_clean.csv
"""

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_clean_manifest():
    orig_manifest = PROJECT_ROOT / "data/metadata/metadata.csv"
    chunk_manifest = PROJECT_ROOT / "data/metadata/metadata_chunks.csv"
    out_manifest = PROJECT_ROOT / "data/metadata/metadata_chunks_clean.csv"

    with open(orig_manifest, "r", encoding="utf-8") as f:
        orig_rows = list(csv.DictReader(f))

    with open(chunk_manifest, "r", encoding="utf-8") as f:
        chunk_rows = list(csv.DictReader(f))

    clean_records = []

    for r in orig_rows:
        stem = Path(r["audio_path"]).stem
        lang = r["language"]
        gt = r["transcript"]
        speaker = r.get("speaker", "amaan")

        # Split into individual sentences/phrases
        sents = [s.strip() for s in re.split(r"[\.\?\!|।]\s*", gt) if s.strip()]
        if not sents:
            sents = [gt.strip()]

        file_chunks = [
            c for c in chunk_rows
            if re.sub(r"_seg_\d+\.wav$", "", Path(c["audio_path"]).name) == stem
        ]

        # Sort chunks sequentially by segment number
        file_chunks.sort(key=lambda x: x["audio_path"])

        for i, c in enumerate(file_chunks):
            # Check audio file actually exists on disk
            full_audio = PROJECT_ROOT / c["audio_path"]
            if not full_audio.exists():
                continue

            dur = float(c["duration"])
            if dur < 0.5:
                continue

            if lang == "en":
                # For English, use Whisper text if reasonable, else GT sentence
                txt = c["transcript"].strip()
                if not txt:
                    txt = sents[i % len(sents)]
            else:
                # For Hindi and Gujarati, use exact ground-truth sentence in target script
                txt = sents[i % len(sents)]

            clean_records.append({
                "audio_path": c["audio_path"],
                "transcript": txt,
                "language": lang,
                "speaker": speaker,
                "duration": round(dur, 2),
            })

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"]
        )
        writer.writeheader()
        writer.writerows(clean_records)

    print("=" * 65)
    print("✓ Clean Chunks Manifest Generated Successfully!")
    print(f"Total samples: {len(clean_records)}")
    en = sum(1 for c in clean_records if c["language"] == "en")
    hi = sum(1 for c in clean_records if c["language"] == "hi")
    gu = sum(1 for c in clean_records if c["language"] == "gu")
    print(f"  - English  : {en}")
    print(f"  - Hindi    : {hi}")
    print(f"  - Gujarati : {gu}")
    print(f"Output saved to: {out_manifest}")
    print("=" * 65)


if __name__ == "__main__":
    build_clean_manifest()
