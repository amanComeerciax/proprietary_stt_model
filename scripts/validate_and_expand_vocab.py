"""
Vocabulary Validation & Safe Expansion Utility for Multi-Language STT

Scans transcripts from one or more manifests, verifies that all characters
exist in tokenizer/vocab.json, and safely appends missing characters to the
vocabulary without disrupting existing token IDs.
"""

import json
import csv
import argparse
from pathlib import Path


def check_and_expand_vocab(manifest_paths, vocab_path="tokenizer/vocab.json", expand=False):
    vocab_file = Path(vocab_path)
    if not vocab_file.exists():
        print(f"[!] Vocab file not found: {vocab_file}")
        return

    with open(vocab_file, mode="r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    char_to_id = vocab_data.get("char_to_id", {})
    id_to_char = vocab_data.get("id_to_char", {})

    missing_chars = set()
    total_lines = 0

    for m_path in manifest_paths:
        m_file = Path(m_path)
        if not m_file.exists():
            continue
        with open(m_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_lines += 1
                transcript = row.get("transcript", "")
                for char in transcript:
                    if char not in char_to_id:
                        missing_chars.add(char)

    print(f"[*] Scanned {total_lines:,} transcripts across {len(manifest_paths)} manifests.")
    print(f"[*] Existing Vocab Size: {len(char_to_id)}")

    if not missing_chars:
        print("[✓] 100% of characters are already in tokenizer/vocab.json! No missing tokens.")
        return

    sorted_missing = sorted(list(missing_chars))
    print(f"[!] Found {len(sorted_missing)} missing characters: {sorted_missing}")

    if expand:
        next_id = max(int(i) for i in char_to_id.values()) + 1 if char_to_id else 0
        added = 0
        for char in sorted_missing:
            char_to_id[char] = next_id
            id_to_char[str(next_id)] = char
            next_id += 1
            added += 1

        with open(vocab_file, mode="w", encoding="utf-8") as f:
            json.dump({"char_to_id": char_to_id, "id_to_char": id_to_char}, f, ensure_ascii=False, indent=2)

        print(f"[✓] Successfully added {added} new characters to {vocab_file}!")
        print(f"[✓] New Vocab Size: {len(char_to_id)}")
    else:
        print("[i] Run with --expand to automatically add these characters to vocab.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and expand tokenizer vocabulary")
    parser.add_argument("manifests", nargs="+", help="Path to manifest CSV files")
    parser.add_argument("--vocab", default="tokenizer/vocab.json", help="Path to vocab.json")
    parser.add_argument("--expand", action="store_true", help="Automatically add missing characters")
    args = parser.parse_args()

    check_and_expand_vocab(args.manifests, vocab_path=args.vocab, expand=args.expand)
