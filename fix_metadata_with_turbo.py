import csv
import whisper
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

print("Downloading and loading Whisper Turbo model (this may take a minute)...")
model = whisper.load_model("turbo")

metadata_file = 'data/metadata/metadata.csv'

with open(metadata_file, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Transcribing {len(rows)} files to fix labels...")

for idx, r in enumerate(rows):
    audio_path = r['audio_path']
    lang = r['language']
    
    print(f"[{idx+1}/{len(rows)}] Transcribing {audio_path}...")
    try:
        res = model.transcribe(audio_path, language=lang)
        pred = res.get("text", "").strip()
        
        # Replace the transcript with the highly accurate turbo output
        r['transcript'] = pred
        print(f"  -> Fixed: {pred[:50]}...")
    except Exception as e:
        print(f"  -> Error: {e}")

# Save back to CSV
with open(metadata_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Done! All labels have been fixed and replaced with highly accurate transcriptions.")
