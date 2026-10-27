import os
import csv
import re
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "data" / "metadata" / "prompts"
STUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "chunks" / "studio_recorded"
STUDIO_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata_studio.csv"

os.makedirs(STUDIO_DIR, exist_ok=True)

app = Flask(__name__)

# Initialize CSV if not exists
if not STUDIO_CSV.exists():
    with open(STUDIO_CSV, "w", encoding="utf-8") as f:
        f.write("audio_path,transcript,language,speaker,duration\n")

def get_all_sentences():
    sentences = []
    for md_file in PROMPTS_DIR.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip markdown formatting, empty lines, and headers
                if not line or line.startswith("#") or line.startswith("---") or line.startswith("==="):
                    continue
                # Split line into sentences if needed, or just yield the line
                # Let's split by simple delimiters like | or just take the line
                clean = re.sub(r'^\d+\.\s*', '', line) # remove numbering like 1. 2. 
                clean = clean.replace('*', '').replace('-', '').strip()
                if len(clean) > 2:
                    sentences.append(clean)
    
    # Deduplicate and return
    return list(dict.fromkeys(sentences))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Studio Recording Tool</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; padding: 40px 20px; box-sizing: border-box; }
        .container { background: #1e293b; padding: 40px; border-radius: 12px; border: 1px solid #334155; width: 100%; max-width: 800px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5); margin-bottom: 20px; }
        h1 { color: #38bdf8; margin-bottom: 5px; }
        .progress { color: #94a3b8; font-size: 14px; margin-bottom: 30px; }
        .sentence-box { font-size: 32px; font-weight: bold; margin-bottom: 40px; line-height: 1.4; color: #f1f5f9; min-height: 100px; display: flex; align-items: center; justify-content: center; }
        .btn { padding: 15px 40px; font-size: 20px; border: none; border-radius: 50px; cursor: pointer; transition: 0.2s; font-weight: bold; }
        .btn-record { background: #ef4444; color: white; }
        .btn-record:hover { background: #dc2626; }
        .btn-stop { background: #eab308; color: black; display: none; }
        .btn-stop:hover { background: #ca8a04; }
        .btn-skip { background: #334155; color: white; margin-left: 20px; }
        .btn-skip:hover { background: #475569; }
        #status { margin-top: 20px; font-size: 16px; color: #10b981; height: 20px; }
        .recording-indicator { display: none; width: 20px; height: 20px; background: #ef4444; border-radius: 50%; margin: 0 auto 20px auto; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 100% { transform: scale(1.5); opacity: 0; } }
        
        .history-container { margin-top: 20px; width: 100%; max-width: 800px; text-align: left; }

        .history-title { color: #38bdf8; font-size: 20px; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 20px; }
        .history-item { background: #1e293b; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
        .history-text { flex: 1; font-size: 16px; color: #f8fafc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Studio Recorder</h1>
        <div class="progress" id="progress">Loading...</div>
        
        <div class="recording-indicator" id="indicator"></div>
        <div class="sentence-box" id="sentence">Please wait...</div>
        
        <button class="btn btn-record" id="recordBtn" onclick="startRecording()">🎤 Start Recording</button>
        <button class="btn btn-stop" id="stopBtn" onclick="stopRecording()">⏹️ Stop</button>
        <button class="btn btn-skip" id="skipBtn" onclick="nextSentence()">⏭️ Skip</button>
        
        <div id="reviewSection" style="display:none; margin-top: 20px;">
            <audio id="reviewAudio" controls style="margin-bottom: 20px;"></audio><br>
            <button class="btn" style="background:#10b981; color:white;" onclick="uploadAudio()">✅ Save & Next</button>
            <button class="btn" style="background:#f59e0b; color:black; margin-left:10px;" onclick="cancelRecording()">🔄 Re-record</button>
        </div>
        
        <div id="status"></div>
        
        <div style="margin-top: 40px; border-top: 1px solid #334155; padding-top: 20px;">
            <a href="/download_csv" class="btn" style="background:#0284c7; color:white; text-decoration:none; display:inline-block; font-size:16px; padding: 10px 20px;">📥 Download Final CSV</a>
            <p style="font-size:12px; color:#94a3b8; margin-top:10px;">(Don't worry, every recording is auto-saved to your hard drive instantly!)</p>
        </div>
    </div>

    <div class="history-container">
        <div class="history-title">Recent Recordings</div>
        <div id="historyList"></div>
    </div>

    <script>
        let sentences = {{ sentences|tojson }};
        let currentIndex = {{ start_index }}; // Resumes from where you left off
        let mediaRecorder;
        let audioChunks = [];
        let startTime = 0;

        const sentenceEl = document.getElementById("sentence");
        const progressEl = document.getElementById("progress");
        const recordBtn = document.getElementById("recordBtn");
        const stopBtn = document.getElementById("stopBtn");
        const skipBtn = document.getElementById("skipBtn");
        const indicator = document.getElementById("indicator");
        const statusEl = document.getElementById("status");
        const historyList = document.getElementById("historyList");
        const reviewSection = document.getElementById("reviewSection");
        const reviewAudio = document.getElementById("reviewAudio");

        function updateUI() {
            if (currentIndex >= sentences.length) {
                sentenceEl.innerHTML = "🎉 All done! Dataset complete.";
                recordBtn.style.display = "none";
                skipBtn.style.display = "none";
                reviewSection.style.display = "none";
                progressEl.innerHTML = `Completed ${sentences.length} / ${sentences.length}`;
                return;
            }
            sentenceEl.innerText = sentences[currentIndex];
            progressEl.innerText = `Sentence ${currentIndex + 1} of ${sentences.length}`;
            statusEl.innerText = "";
            recordBtn.style.display = "inline-block";
            skipBtn.style.display = "inline-block";
            reviewSection.style.display = "none";
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = e => {
                    if(e.data.size > 0) audioChunks.push(e.data);
                };
                
                mediaRecorder.start();
                startTime = Date.now();
                
                recordBtn.style.display = "none";
                skipBtn.style.display = "none";
                stopBtn.style.display = "inline-block";
                indicator.style.display = "block";
                statusEl.innerText = "Recording...";
                statusEl.style.color = "#ef4444";
            } catch(e) {
                alert("Microphone access denied or not available!");
            }
        }

        function stopRecording() {
            if(mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                recordBtn.style.display = "none";
                stopBtn.style.display = "none";
                indicator.style.display = "none";
                
                statusEl.innerText = "Review your recording...";
                statusEl.style.color = "#38bdf8";
                
                setTimeout(() => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const url = URL.createObjectURL(audioBlob);
                    reviewAudio.src = url;
                    reviewSection.style.display = "block";
                }, 300);
            }
        }

        function cancelRecording() {
            audioChunks = [];
            reviewSection.style.display = "none";
            recordBtn.style.display = "inline-block";
            skipBtn.style.display = "inline-block";
            statusEl.innerText = "";
        }

        function uploadAudio() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            const duration = (Date.now() - startTime) / 1000.0;
            const currentText = sentences[currentIndex];
            
            reviewSection.style.display = "none";
            statusEl.innerText = "Saving to disk...";
            statusEl.style.color = "#eab308";
            
            formData.append("audio", audioBlob, "audio.webm");
            formData.append("text", currentText);
            formData.append("duration", duration);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            }).then(response => response.json())
              .then(data => {
                  if(data.success) {
                      statusEl.innerText = "✅ Saved!";
                      statusEl.style.color = "#10b981";
                      
                      const item = document.createElement("div");
                      item.className = "history-item";
                      item.innerHTML = `
                          <div class="history-text">${currentText}</div>
                          <div><audio controls src="${data.audio_url}"></audio></div>
                      `;
                      historyList.prepend(item);
                      
                      setTimeout(() => {
                          currentIndex++;
                          updateUI();
                      }, 500);
                  } else {
                      statusEl.innerText = "❌ Error saving!";
                      statusEl.style.color = "#ef4444";
                      reviewSection.style.display = "block";
                  }
              });
        }
        
        function nextSentence() {
            currentIndex++;
            updateUI();
        }

        // Init
        updateUI();
    </script>
</body>
</html>
"""

from flask import send_file

@app.route("/")
def index():
    sentences = get_all_sentences()
    
    # Auto-resume feature: Check how many are already saved in CSV
    start_index = 0
    if STUDIO_CSV.exists():
        with open(STUDIO_CSV, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Ignore header line
            if len(lines) > 1:
                start_index = len(lines) - 1
                
    return render_template_string(HTML_TEMPLATE, sentences=sentences, start_index=start_index)

# Add route for serving audio files statically
from flask import send_from_directory
@app.route("/data/<path:path>")
def send_data(path):
    return send_from_directory(PROJECT_ROOT / "data", path)

@app.route("/download_csv")
def download_csv():
    if STUDIO_CSV.exists():
        return send_file(STUDIO_CSV, as_attachment=True, download_name="metadata_studio.csv")
    return "No recordings yet!", 404

@app.route("/upload", methods=["POST"])
def upload():
    audio = request.files.get("audio")
    text = request.form.get("text")
    duration = request.form.get("duration")
    
    if not audio or not text:
        return jsonify(success=False), 400
        
    # Get the next file number
    count = len(list(STUDIO_DIR.glob("*.wav"))) + 1
    wav_filename = f"studio_rec_{count:04d}.wav"
    wav_full_path = STUDIO_DIR / wav_filename
    
    # Save as webm first
    temp_webm = STUDIO_DIR / "temp.webm"
    audio.save(temp_webm)
    
    # Convert webm to wav using ffmpeg (via subprocess)
    subprocess.run(["ffmpeg", "-y", "-i", str(temp_webm), "-ar", "16000", "-ac", "1", str(wav_full_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if temp_webm.exists():
        temp_webm.unlink()
        
    rel_path = f"data/processed/chunks/studio_recorded/{wav_filename}"
    
    # detect simple language (if contains hindi characters -> hi, gujarati -> gu, else en)
    lang = "en"
    if re.search(r'[\u0900-\u097F]', text):
        lang = "hi"
    elif re.search(r'[\u0A80-\u0AFF]', text):
        lang = "gu"
        
    with open(STUDIO_CSV, "a", encoding="utf-8") as f:
        f.write(f'"{rel_path}","{text}","{lang}","user","{duration}"\n')
        
    return jsonify(success=True, audio_url="/" + rel_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
