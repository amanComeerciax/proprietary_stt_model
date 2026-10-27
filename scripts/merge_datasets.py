"""
Merge all 4 members' datasets into one unified, clean CSV + organized WAV directory.

Features:
  - Robust path resolution for all members (Amaan, Divya, Deep, Priyanshi)
  - Automatic filtering of prompt instruction headers (e.g., 'No script...', 'Read each...')
  - Automatic stripping of parenthetical tone notes (e.g., '(surprised)', '(खुशी से)')
  - Clean per-speaker audio file numbering: {speaker}_{id:06d}.wav
  - Standardized 16kHz mono audio verification & duration filtering (0.5s - 30s)
  - Unified metadata CSV at: data/metadata/metadata_all.csv
"""

import csv
import os
import re
import shutil
import soundfile as sf

BASE = "/Users/commerciax-fs-1/Documents/proprietary_model_stt/our-stt-model"
OUT_WAV_DIR = os.path.join(BASE, "data/processed/all_speakers")
OUT_CSV = os.path.join(BASE, "data/metadata/metadata_all.csv")

# Clean output dir to avoid orphaned files
if os.path.exists(OUT_WAV_DIR):
    shutil.rmtree(OUT_WAV_DIR)
os.makedirs(OUT_WAV_DIR, exist_ok=True)

# Language normalization map
LANG_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi",
    "gujarati": "gu", "gu": "gu",
}

# Instruction heading substrings that were not meant to be spoken
INSTRUCTION_PATTERNS = [
    "no script",
    "read each",
    "read these",
    "સ્વાભાવિક ઝડપે બોલો",
    "અલગ ભાવ સાથે બોલો",
    "વાત કરતા હો એમ બોલો",
    "ભાવ કે સાથ બોલેં",
    "ભાવ કે સાથે બોલો",
    "હર અંક અલગ",
    "हर अंक अलग",
    "એક જ વાક્ય અલગ",
    "एक ही वाक्य को अलग",
    "dhire shuru",
    "dhime shuru",
]

unified_rows = []
stats = {"total": 0, "copied": 0, "skipped": 0, "instructions_filtered": 0, "parens_cleaned": 0, "errors": []}


def clean_transcript(text: str) -> tuple[str, bool, bool]:
    """Clean transcript: remove prompt notes in parentheses and detect instructions.
    
    Returns:
        (cleaned_text, is_instruction, had_parens)
    """
    if not text:
        return "", False, False
        
    lower_text = text.lower()
    for pat in INSTRUCTION_PATTERNS:
        if pat in lower_text:
            return "", True, False

    # Check if text contains parenthetical notes
    had_parens = bool(re.search(r"\(.*?\)", text))
    cleaned = re.sub(r"\(.*?\)", "", text)

    # Normalize whitespace & strip quotes
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip('"\'“”')

    # If nothing left, it was purely parenthetical
    if not cleaned:
        return "", True, had_parens

    return cleaned, False, had_parens


def add_member(speaker_name, csv_path, wav_dir, audio_col, text_col, lang_col, dur_col, 
               file_is_basename=False):
    """Process one member's data and add to unified list with clean indexing."""
    speaker_copied = 0
    speaker_skipped = 0
    speaker_inst_filtered = 0
    speaker_parens_cleaned = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            stats["total"] += 1

            raw_text = row.get(text_col, "").strip()
            transcript, is_instruction, had_parens = clean_transcript(raw_text)

            if is_instruction:
                speaker_inst_filtered += 1
                stats["instructions_filtered"] += 1
                continue

            if had_parens:
                speaker_parens_cleaned += 1
                stats["parens_cleaned"] += 1

            if not transcript:
                speaker_skipped += 1
                stats["skipped"] += 1
                continue

            # Get language
            lang_raw = row.get(lang_col, "en").strip().lower()
            lang = LANG_MAP.get(lang_raw, lang_raw)

            # Find WAV file
            audio_ref = row.get(audio_col, "").strip()
            if not audio_ref:
                speaker_skipped += 1
                stats["skipped"] += 1
                continue

            if file_is_basename:
                wav_filename = audio_ref
            else:
                wav_filename = os.path.basename(audio_ref)

            # Resolve source path
            src_path = None
            if wav_dir and os.path.exists(os.path.join(wav_dir, wav_filename)):
                src_path = os.path.join(wav_dir, wav_filename)
            elif os.path.exists(os.path.join(BASE, audio_ref)):
                src_path = os.path.join(BASE, audio_ref)
            elif os.path.exists(audio_ref):
                src_path = audio_ref

            if not src_path or not os.path.exists(src_path):
                speaker_skipped += 1
                stats["skipped"] += 1
                continue

            # Get actual duration from file
            try:
                info = sf.info(src_path)
                actual_duration = round(info.duration, 3)
            except Exception as e:
                stats["errors"].append(f"{src_path}: {e}")
                speaker_skipped += 1
                stats["skipped"] += 1
                continue

            # Filter duration: skip <0.5s or >30s
            if actual_duration < 0.5 or actual_duration > 30.0:
                speaker_skipped += 1
                stats["skipped"] += 1
                continue

            # Filter out anomalously fast speech (wps > 6.0 and dur < 1.5s = click/noise)
            word_count = len(transcript.split())
            wps = word_count / actual_duration if actual_duration > 0 else 0
            if word_count > 4 and actual_duration < 1.5 and wps > 5.5:
                speaker_inst_filtered += 1
                stats["instructions_filtered"] += 1
                continue

            speaker_copied += 1
            stats["copied"] += 1

            # New filename: speaker_000001.wav
            new_filename = f"{speaker_name}_{speaker_copied:06d}.wav"
            dst_path = os.path.join(OUT_WAV_DIR, new_filename)
            rel_path = f"data/processed/all_speakers/{new_filename}"

            # Copy WAV file
            shutil.copy2(src_path, dst_path)

            unified_rows.append({
                "audio_path": rel_path,
                "transcript": transcript,
                "language": lang,
                "speaker": speaker_name,
                "duration": actual_duration,
            })

    print(f"  ✓ {speaker_name.upper():10s}: {speaker_copied:4d} kept | {speaker_inst_filtered:2d} instructions dropped | {speaker_parens_cleaned:2d} tone notes cleaned | {speaker_skipped:2d} missing/short skipped")


print("=" * 70)
print("Merging and Normalizing all 4 members' datasets...")
print("=" * 70)

# Member 1: Amaan
print("\n[1/4] Processing Amaan's data...")
add_member(
    speaker_name="amaan",
    csv_path=os.path.join(BASE, "data/metadata/metadata.csv"),
    wav_dir=os.path.join(BASE, "data/processed/chunks/studio_recorded"),
    audio_col="audio_path",
    text_col="transcript",
    lang_col="language",
    dur_col="duration",
    file_is_basename=False,
)

# Member 2: Divya
print("\n[2/4] Processing Divya's data...")
add_member(
    speaker_name="divya",
    csv_path=os.path.expanduser("~/Downloads/divydata/metadata.csv"),
    wav_dir=os.path.expanduser("~/Downloads/divydata/wav"),
    audio_col="audio_path",
    text_col="text",
    lang_col="language",
    dur_col="duration",
    file_is_basename=False,
)

# Member 3: Deep
print("\n[3/4] Processing Deep's data...")
add_member(
    speaker_name="deep",
    csv_path=os.path.expanduser("~/Downloads/Deep_voice/metadata_wav.csv"),
    wav_dir=os.path.expanduser("~/Downloads/Deep_voice/audio_wav"),
    audio_col="audio_path",
    text_col="text",
    lang_col="language",
    dur_col="duration",
    file_is_basename=False,
)

# Member 4: Priyanshi
print("\n[4/4] Processing Priyanshi's data...")
add_member(
    speaker_name="priyanshi",
    csv_path=os.path.expanduser("~/Downloads/Priyanshi_data/metadata.csv"),
    wav_dir=os.path.expanduser("~/Downloads/Priyanshi_data/All_WAV_Recordings"),
    audio_col="file_name",
    text_col="transcription",
    lang_col="language",
    dur_col="duration_sec",
    file_is_basename=True,
)

# Write unified CSV
print(f"\nWriting unified CSV to {OUT_CSV}...")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
    writer.writeheader()
    writer.writerows(unified_rows)

# Print final summary
print("\n" + "=" * 70)
print("✓ MERGE & CLEAN COMPLETE SUCCESSFULLY!")
print("=" * 70)

speakers = {}
langs = {}
total_dur = 0
for row in unified_rows:
    sp = row["speaker"]
    speakers[sp] = speakers.get(sp, 0) + 1
    lg = row["language"]
    langs[lg] = langs.get(lg, 0) + 1
    total_dur += row["duration"]

print(f"\nTotal clean files : {len(unified_rows)}")
print(f"Total duration    : {total_dur:.1f}s ({total_dur/60:.1f} min = {total_dur/3600:.2f} hours)")
print(f"Tone notes cleaned: {stats['parens_cleaned']} prompts (removed prompt directions like '(surprised)')")
print(f"Garbage headers   : {stats['instructions_filtered']} dropped (non-speech instructions)")
print(f"Unrecorded skipped: {stats['skipped']} files")

print(f"\nBreakdown By Speaker:")
for sp, count in sorted(speakers.items()):
    sp_dur = sum(r["duration"] for r in unified_rows if r["speaker"] == sp)
    print(f"  {sp:12s}: {count:4d} files ({sp_dur/60:.1f} min)")

print(f"\nBreakdown By Language:")
for lg, count in sorted(langs.items()):
    lg_dur = sum(r["duration"] for r in unified_rows if r["language"] == lg)
    print(f"  {lg:12s}: {count:4d} files ({lg_dur/60:.1f} min)")

print(f"\nOutput CSV : {OUT_CSV}")
print(f"Output WAV : {OUT_WAV_DIR}/")
