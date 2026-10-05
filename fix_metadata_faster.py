import csv
from faster_whisper import WhisperModel
import time

print("Loading Faster-Whisper Turbo model...")
start_time = time.time()
# CPU inference with int8 quantization is super fast!
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

metadata_file = 'data/metadata/metadata.csv'

with open(metadata_file, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Transcribing {len(rows)} files...")

for idx, r in enumerate(rows):
    audio_path = r['audio_path']
    lang = r['language']
    
    print(f"[{idx+1}/{len(rows)}] Transcribing {audio_path}...")
    t0 = time.time()
    try:
        segments, info = model.transcribe(audio_path, language=lang, condition_on_previous_text=False)
        pred = "".join([segment.text for segment in segments]).strip()
        
        r['transcript'] = pred
        print(f"  -> Fixed in {time.time()-t0:.1f}s: {pred[:60]}...")
    except Exception as e:
        print(f"  -> Error: {e}")

with open(metadata_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Done! All labels have been fixed.")
