import csv
import whisper
import warnings
from pathlib import Path
from difflib import SequenceMatcher
warnings.filterwarnings("ignore")

print("Loading Whisper model (base)...")
model = whisper.load_model("base")

with open('data/metadata/metadata.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Gather all ground truth texts
all_gts = [r['transcript'] for r in rows]

print(f"Checking {len(rows)} files...")
for idx, r in enumerate(rows):
    audio_path = r['audio_path']
    gt = r['transcript']
    
    # Transcribe audio
    res = model.transcribe(audio_path)
    pred = res.get("text", "").strip()
    
    # Compare with its CURRENT ground truth
    sim_current = SequenceMatcher(None, pred, gt).ratio()
    
    # Find BEST matching ground truth among all
    best_sim = 0
    best_gt = ""
    for cand_gt in all_gts:
        sim = SequenceMatcher(None, pred, cand_gt).ratio()
        if sim > best_sim:
            best_sim = sim
            best_gt = cand_gt
            
    # Check lengths
    gt_len = len(gt)
    pred_len = len(pred)
    
    print(f"\n--- {audio_path.split('/')[-1]} ---")
    print(f"Whisper output length : {pred_len} chars")
    print(f"Current GT length     : {gt_len} chars")
    
    if best_gt == gt:
        print(f"Status: MATCH (Similarity: {sim_current:.2f})")
    else:
        print(f"Status: MISMATCH!")
        print(f"  Current similarity: {sim_current:.2f}")
        print(f"  Best similarity   : {best_sim:.2f}")
        print(f"  Best GT text      : {best_gt[:100]}...")

