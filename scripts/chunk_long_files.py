import os
import pandas as pd
import librosa
import soundfile as sf
from faster_whisper import WhisperModel

def main():
    metadata_path = "data/metadata/metadata_chunks.csv"
    output_csv = "data/metadata/metadata_final_chunks.csv"
    
    print("Loading Whisper model (small) for faster AI transcription...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    df = pd.read_csv(metadata_path)
    final_rows = []
    
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        
        # Jo pehle se chhote chunk hain, unko waisa hi rakh lo
        if not audio_path.endswith("_full.wav"):
            final_rows.append(row.to_dict())
            continue
            
        print(f"\nProcessing long file: {audio_path}")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            final_rows.append(row.to_dict())
            continue
            
        # Jab insaan bolte hue chup hota hai (silence), tabhi kaato (top_db=35 means 35 decibels quiet)
        intervals = librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512)
        
        if len(intervals) <= 1:
            print(f"Could not split {audio_path} further on silence.")
            final_rows.append(row.to_dict())
            continue
            
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.basename(audio_path).replace("_full.wav", "")
        
        for i, (start_i, end_i) in enumerate(intervals):
            # Aawaz ekdum na kat jaye isliye aage aur peeche thoda margin de do (200ms)
            start_i = max(0, start_i - int(sr*0.2))
            end_i = min(len(y), end_i + int(sr*0.2))
            
            chunk_y = y[start_i:end_i]
            duration = len(chunk_y) / sr
            
            # Agar tukda 1 second se bhi chhota hai, toh faltu hai
            if duration < 1.0:
                continue
                
            chunk_path = os.path.join(base_dir, f"{base_name}_split_{i:03d}.wav")
            sf.write(chunk_path, chunk_y, sr)
            
            # Ab Whisper AI se is naye chunk ka transcript nikalwa lo
            segments, _ = model.transcribe(chunk_path, language=row['language'], vad_filter=True)
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            
            if len(text) > 2:
                final_rows.append({
                    "audio_path": chunk_path,
                    "transcript": text,
                    "language": row['language'],
                    "speaker": row['speaker'],
                    "duration": round(duration, 2)
                })
                print(f"  -> Created chunk {i}: {text} ({round(duration, 2)}s)")
            else:
                # Agar AI ko kuch samajh nahi aaya, toh delete kardo
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\nDone! Created {len(out_df)} perfect smart chunks at {output_csv}")

if __name__ == "__main__":
    main()
