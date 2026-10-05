import csv
import os
import re
from pathlib import Path
from faster_whisper import WhisperModel
import soundfile as sf
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    manifest_file = PROJECT_ROOT / "data/metadata/metadata.csv"
    chunks_base_dir = PROJECT_ROOT / "data/processed/chunks"
    manifest_out_file = PROJECT_ROOT / "data/metadata/metadata_chunks.csv"
    
    print("Loading Faster-Whisper large-v3 for alignment...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    
    with open(manifest_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    chunk_records = []
    
    for idx, row in enumerate(rows, 1):
        rel_audio_path = row["audio_path"]
        full_audio_path = PROJECT_ROOT / rel_audio_path
        language = row.get("language", "en")
        speaker = row.get("speaker", "amaan")
        
        print(f"[{idx:02d}/{len(rows):02d}] Segmenting {rel_audio_path}...")
        
        audio_data, sr = sf.read(str(full_audio_path), dtype="float32")
        total_dur = len(audio_data) / sr
        
        t0 = time.time()
        segments_gen, _ = model.transcribe(str(full_audio_path), language=language, condition_on_previous_text=False)
        segments = list(segments_gen)
        
        lang_chunk_dir = chunks_base_dir / language
        lang_chunk_dir.mkdir(parents=True, exist_ok=True)
        file_stem = full_audio_path.stem
        
        created_chunks = 0
        for s_idx, seg in enumerate(segments, 1):
            start_s = max(0.0, seg.start - 0.05)
            end_s = min(total_dur, seg.end + 0.05)
            duration = end_s - start_s
            
            if duration < 0.5: continue
            
            seg_text = clean_text(seg.text)
            if not seg_text: continue
                
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            chunk_waveform = audio_data[start_sample:end_sample]
            if len(chunk_waveform) == 0: continue
                
            chunk_filename = f"{file_stem}_seg_{s_idx:03d}.wav"
            chunk_full_path = lang_chunk_dir / chunk_filename
            chunk_rel_path = str(chunk_full_path.relative_to(PROJECT_ROOT))
            
            sf.write(str(chunk_full_path), chunk_waveform, sr)
            chunk_records.append({
                "audio_path": chunk_rel_path,
                "transcript": seg_text,
                "language": language,
                "speaker": speaker,
                "duration": round(duration, 2),
            })
            created_chunks += 1
            
        print(f"  -> Created {created_chunks} chunks in {time.time()-t0:.1f}s")

    with open(manifest_out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
        writer.writeheader()
        writer.writerows(chunk_records)
        
    print(f"Done! Created {len(chunk_records)} total chunks.")

if __name__ == "__main__":
    main()
