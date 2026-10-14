# -*- coding: utf-8 -*-
import csv
import os
import re
from pathlib import Path
from faster_whisper import WhisperModel
import soundfile as sf
import time
from difflib import SequenceMatcher

PROJECT_ROOT = Path("/Users/commerciax-fs-1/Documents/proprietary_model_stt/our-stt-model")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_sentences(text: str) -> list:
    # Split by ., ?, !, । but keep the punctuation
    # \u0964 is DEVANAGARI DANDA (।)
    sentences = re.split(r'(?<=[.!?\u0964])\s+', text.strip())
    return [s for s in sentences if s.strip()]

def similarity(a, b):
    # Remove punctuation for comparison
    a_clean = re.sub(r'[^\w\s]', '', a).lower()
    b_clean = re.sub(r'[^\w\s]', '', b).lower()
    return SequenceMatcher(None, a_clean, b_clean).ratio()

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
        gt_transcript = row.get("transcript", "")
        
        gt_sentences = split_into_sentences(gt_transcript)
        
        print(f"\n[{idx:02d}/{len(rows):02d}] Segmenting {rel_audio_path}")
        print(f"  -> Target sentences: {len(gt_sentences)}")
        
        audio_data, sr = sf.read(str(full_audio_path), dtype="float32")
        total_dur = len(audio_data) / sr
        
        t0 = time.time()
        # We use vad_filter to get cleaner pauses
        segments_gen, _ = model.transcribe(str(full_audio_path), language=language, condition_on_previous_text=False, vad_filter=True)
        segments = list(segments_gen)
        
        lang_chunk_dir = chunks_base_dir / language
        lang_chunk_dir.mkdir(parents=True, exist_ok=True)
        file_stem = full_audio_path.stem
        
        # Heuristic alignment
        # We will try to map each GT sentence to one or more Whisper segments
        aligned_chunks = []
        seg_idx = 0
        
        for gt_sent in gt_sentences:
            best_sim = 0
            best_end_idx = seg_idx
            current_text = ""
            
            # Look ahead to combine segments and find the best match for this GT sentence
            for j in range(seg_idx, min(seg_idx + 4, len(segments))):
                current_text += " " + segments[j].text
                sim = similarity(gt_sent, current_text)
                if sim > best_sim:
                    best_sim = sim
                    best_end_idx = j
            
            # If we found a reasonable match
            if best_sim > 0.5 and seg_idx < len(segments):
                start_s = max(0.0, segments[seg_idx].start - 0.1)
                end_s = min(total_dur, segments[best_end_idx].end + 0.1)
                
                aligned_chunks.append({
                    "start": start_s,
                    "end": end_s,
                    "text": gt_sent # USE EXACT GROUND TRUTH TEXT
                })
                seg_idx = best_end_idx + 1
            else:
                print(f"  -> WARNING: Could not align sentence: '{gt_sent[:30]}...'")
        
        # If alignment missed sentences, we fallback to just saving the whole file as 1 chunk to preserve exact data
        if len(aligned_chunks) != len(gt_sentences):
            print(f"  -> ALIGNMENT MISMATCH (Found {len(aligned_chunks)}, Expected {len(gt_sentences)}). Saving whole file as 1 chunk.")
            chunk_filename = f"{file_stem}_full.wav"
            chunk_full_path = lang_chunk_dir / chunk_filename
            chunk_rel_path = str(chunk_full_path.relative_to(PROJECT_ROOT))
            
            sf.write(str(chunk_full_path), audio_data, sr)
            chunk_records.append({
                "audio_path": chunk_rel_path,
                "transcript": gt_transcript,
                "language": language,
                "speaker": speaker,
                "duration": round(total_dur, 2),
            })
        else:
            # Save the perfectly aligned chunks
            for i, chunk in enumerate(aligned_chunks, 1):
                duration = chunk["end"] - chunk["start"]
                if duration <= 0: continue
                
                start_sample = int(chunk["start"] * sr)
                end_sample = int(chunk["end"] * sr)
                chunk_waveform = audio_data[start_sample:end_sample]
                
                chunk_filename = f"{file_stem}_seg_{i:03d}.wav"
                chunk_full_path = lang_chunk_dir / chunk_filename
                chunk_rel_path = str(chunk_full_path.relative_to(PROJECT_ROOT))
                
                sf.write(str(chunk_full_path), chunk_waveform, sr)
                chunk_records.append({
                    "audio_path": chunk_rel_path,
                    "transcript": chunk["text"],
                    "language": language,
                    "speaker": speaker,
                    "duration": round(duration, 2),
                })
            print(f"  -> Created {len(aligned_chunks)} exact chunks in {time.time()-t0:.1f}s")

    # Write output manifest
    with open(manifest_out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker", "duration"])
        writer.writeheader()
        writer.writerows(chunk_records)
        
    print(f"\nDone! Created {len(chunk_records)} total chunks.")

if __name__ == "__main__":
    main()
