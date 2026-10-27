import os
import csv
import soundfile as sf
import http.server
import socketserver
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
metadata_path = PROJECT_ROOT / "data" / "metadata" / "metadata_chunks.csv"
manual_chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks" / "manual"
html_path = PROJECT_ROOT / "manual_annotation.html"

os.makedirs(manual_chunks_dir, exist_ok=True)

def slice_audio_and_generate_ui():
    print("Slicing long files into 10-second chunks...")
    
    manual_records = []
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = PROJECT_ROOT / row["audio_path"]
            
            # Only process the long uncut files
            if "_full.wav" in row["audio_path"]:
                try:
                    data, sr = sf.read(str(audio_path))
                    total_samples = len(data)
                    chunk_samples = 10 * sr # 10 seconds
                    
                    file_stem = audio_path.stem
                    
                    for i, start in enumerate(range(0, total_samples, chunk_samples)):
                        end = min(start + chunk_samples, total_samples)
                        chunk_data = data[start:end]
                        duration = (end - start) / sr
                        
                        # Skip if chunk is too tiny (less than 1 second)
                        if duration < 1.0:
                            continue
                            
                        chunk_filename = f"{file_stem}_part{i+1:02d}.wav"
                        chunk_rel_path = f"data/processed/chunks/manual/{chunk_filename}"
                        chunk_full_path = manual_chunks_dir / chunk_filename
                        
                        sf.write(str(chunk_full_path), chunk_data, sr)
                        
                        manual_records.append({
                            "audio_path": chunk_rel_path,
                            "original_text": row["transcript"],
                            "language": row["language"],
                            "speaker": row["speaker"],
                            "duration": round(duration, 2)
                        })
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

    print(f"Created {len(manual_records)} 10-second chunks!")

    print("Generating HTML UI...")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Manual Annotation UI</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }
        h1 { color: #38bdf8; text-align: center; }
        .card { background: #1e293b; padding: 15px; margin-bottom: 15px; border-radius: 8px; border: 1px solid #334155; display: flex; gap: 20px; align-items: center; }
        .audio-container { flex: 1; }
        .input-container { flex: 2; }
        textarea { width: 100%; height: 60px; background: #0f172a; color: white; border: 1px solid #38bdf8; border-radius: 4px; padding: 10px; font-size: 16px; }
        button.export { display: block; width: 100%; padding: 15px; background: #0284c7; color: white; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; position: sticky; bottom: 20px; }
        button.export:hover { background: #0369a1; }
        .hint { font-size: 12px; color: #94a3b8; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Audio Splitter & Annotation UI</h1>
    <p style="text-align:center;">Sun kar text likhiye. Jab sab ho jaye toh Export daba dijiye!</p>
    
    <div id="data-container">
"""
    
    for idx, rec in enumerate(manual_records):
        html += f"""
        <div class="card" data-path="{rec['audio_path']}" data-lang="{rec['language']}" data-speaker="{rec['speaker']}" data-dur="{rec['duration']}">
            <div class="audio-container">
                <p><strong>Part {idx+1}</strong> ({rec['duration']}s)</p>
                <audio controls src="/{rec['audio_path']}"></audio>
            </div>
            <div class="input-container">
                <textarea placeholder="Type what you hear here..."></textarea>
                <div class="hint">Original Full Text: {rec['original_text'][:100]}...</div>
            </div>
        </div>
        """
        
    html += """
    </div>
    <button class="export" onclick="exportCSV()">💾 Save & Download CSV</button>

    <script>
        function exportCSV() {
            let csvContent = "audio_path,transcript,language,speaker,duration\\n";
            const cards = document.querySelectorAll('.card');
            
            cards.forEach(card => {
                const path = card.getAttribute('data-path');
                const lang = card.getAttribute('data-lang');
                const speaker = card.getAttribute('data-speaker');
                const dur = card.getAttribute('data-dur');
                let text = card.querySelector('textarea').value.trim();
                
                // Escape quotes
                text = text.replace(/"/g, '""');
                
                if(text !== "") {
                    csvContent += `${path},"${text}",${lang},${speaker},${dur}\\n`;
                }
            });
            
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "manual_annotations.csv");
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def start_server():
    PORT = 8081
    os.chdir(PROJECT_ROOT)
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✅ Annotation UI Ready! Serving at port {PORT}")
        print(f"Open http://localhost:{PORT}/manual_annotation.html in your browser.")
        httpd.serve_forever()

if __name__ == "__main__":
    slice_audio_and_generate_ui()
    start_server()
