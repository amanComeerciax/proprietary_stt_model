"""
Proprietary STT Model — Interactive Web Studio & Real-Time Tester.

Provides an interactive studio supporting:
- 601 clean 3-5s sentence clips across English, Hindi, and Gujarati
- Full original 36 audio files
- Joint CTC & Attention dual-head decoding display
- Ultra-fast Mac CPU inference (< 0.1x RTF)
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Ensure our-stt-model root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.inference import STTInference

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎙️ Proprietary STT Model — Interactive Studio</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(20, 27, 45, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-en: #38bdf8;
      --accent-hi: #f59e0b;
      --accent-gu: #10b981;
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.35);
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      padding: 2rem;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--card-border);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .badge {
      background: linear-gradient(135deg, #6366f1, #a855f7);
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    h1 { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1.25fr;
      gap: 1.75rem;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 1.25rem;
      padding: 1.5rem;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Dataset Mode Toggle */
    .mode-switch {
      display: flex;
      background: rgba(0, 0, 0, 0.3);
      padding: 0.25rem;
      border-radius: 0.65rem;
      border: 1px solid var(--card-border);
      margin-bottom: 1rem;
      gap: 0.25rem;
    }
    .mode-btn {
      flex: 1;
      padding: 0.5rem 0.75rem;
      border-radius: 0.5rem;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.2s;
    }
    .mode-btn.active {
      background: var(--primary);
      color: white;
      box-shadow: 0 4px 12px var(--primary-glow);
    }

    /* Tabs */
    .filter-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    .tab-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.35rem 0.75rem;
      border-radius: 0.5rem;
      font-size: 0.8rem;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn.active, .tab-btn:hover {
      background: rgba(99, 102, 241, 0.25);
      color: white;
      border-color: var(--primary);
    }

    .samples-list {
      max-height: 480px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      padding-right: 0.25rem;
    }

    .sample-item {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem 0.9rem;
      border-radius: 0.75rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s ease;
    }
    .sample-item:hover, .sample-item.selected {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--primary);
      transform: translateY(-1px);
    }
    .sample-name { font-weight: 500; font-size: 0.9rem; }
    .sample-dur { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem; }
    .lang-pill {
      font-size: 0.7rem;
      padding: 0.2rem 0.5rem;
      border-radius: 0.35rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .lang-en { background: rgba(56, 189, 248, 0.15); color: var(--accent-en); border: 1px solid rgba(56, 189, 248, 0.3); }
    .lang-hi { background: rgba(245, 158, 11, 0.15); color: var(--accent-hi); border: 1px solid rgba(245, 158, 11, 0.3); }
    .lang-gu { background: rgba(16, 185, 129, 0.15); color: var(--accent-gu); border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Player & Transcribe Area */
    .player-box {
      margin-bottom: 1.25rem;
      background: rgba(0, 0, 0, 0.25);
      padding: 1.25rem;
      border-radius: 1rem;
      border: 1px solid var(--card-border);
    }
    audio { width: 100%; outline: none; margin-top: 0.75rem; filter: invert(0.9) hue-rotate(180deg); }

    .transcribe-btn {
      width: 100%;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border: none;
      color: white;
      padding: 0.9rem;
      border-radius: 0.75rem;
      font-weight: 600;
      font-size: 1rem;
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 10px 25px var(--primary-glow);
      transition: all 0.2s;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }
    .transcribe-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 15px 30px var(--primary-glow);
    }

    .output-box {
      margin-top: 1.25rem;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.25rem;
      min-height: 120px;
      transition: border-color 0.3s;
    }
    .output-header {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .output-text {
      font-size: 1.25rem;
      line-height: 1.6;
      font-weight: 500;
    }

    .dual-head-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    .head-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 0.85rem;
    }
    .head-title {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 0.35rem;
    }
    .head-val {
      font-size: 1.05rem;
      font-weight: 500;
      color: #e2e8f0;
      word-break: break-word;
    }

    .metrics-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-top: 1.25rem;
    }
    .metric-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      padding: 0.75rem;
      border-radius: 0.75rem;
      text-align: center;
    }
    .metric-val { font-size: 1.1rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }
    .metric-lbl { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; margin-top: 0.2rem; }
    .status-badge {
      display: inline-block;
      font-size: 0.75rem;
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
      padding: 0.2rem 0.5rem;
      border-radius: 0.35rem;
      font-weight: 600;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="badge">Proprietary STT</div>
      <div>
        <h1>Multilingual Speech-to-Text Studio</h1>
        <div class="subtitle">English 🇬🇧 • Hindi 🇮🇳 • Gujarati 🇮🇳 | Mac CPU Real-Time Inference</div>
      </div>
    </div>
  </header>

  <div class="grid">
    <!-- Left: Dataset Samples Explorer -->
    <div class="card">
      <div class="card-title">
        <span>🎧 Audio Samples</span>
        <span id="sampleCountBadge" style="font-size: 0.8rem; color: var(--text-muted);">601 clips</span>
      </div>

      <!-- Mode Switch Removed -->

      <div class="filter-tabs">
        <button class="tab-btn active" onclick="filterLang('all')">All</button>
        <button class="tab-btn" onclick="filterLang('en')">English</button>
        <button class="tab-btn" onclick="filterLang('hi')">Hindi</button>
        <button class="tab-btn" onclick="filterLang('gu')">Gujarati</button>
      </div>

      <div class="samples-list" id="samplesList">
        <!-- Dynamically filled -->
      </div>
    </div>

    <!-- Right: Player & Real-Time Transcriber -->
    <div class="card">
      <div class="card-title">
        <span>⚡ Real-Time Model Inference</span>
        <span id="liveStatusBadge" style="display: none;" class="status-badge">✓ Complete</span>
      </div>

      <div class="player-box">
        <div style="font-weight: 500; font-size: 0.9rem; color: var(--text-muted);" id="currentAudioName">Select an audio sample from the left</div>
        <audio id="audioPlayer" controls style="display: none;"></audio>
      </div>

      <div style="display: flex; gap: 0.75rem;">
        <button class="transcribe-btn" id="transcribeBtn" onclick="runInference()" disabled style="flex: 1;">
          ▶ Transcribe on Mac CPU
        </button>
        <button class="transcribe-btn" id="recordBtn" onclick="toggleRecording()" style="flex: 1; background: linear-gradient(135deg, #ef4444, #b91c1c);">
          🎤 Record Mic
        </button>
      </div>

      <div class="metrics-row" id="metricsRow" style="display: none;">
        <div class="metric-card">
          <div class="metric-val" id="rtfVal">-</div>
          <div class="metric-lbl">Speed (RTF)</div>
        </div>
        <div class="metric-card">
          <div class="metric-val" id="timeVal">-</div>
          <div class="metric-lbl">CPU Latency</div>
        </div>
        <div class="metric-card">
          <div class="metric-val" id="durVal">-</div>
          <div class="metric-lbl">Audio Length</div>
        </div>
      </div>

      <div class="output-box" id="outputCard">
        <div class="output-header">
          <span>Primary Prediction (Speech-to-Text)</span>
          <span id="latencyTag" style="font-size: 0.75rem; color: #38bdf8;"></span>
        </div>
        <div class="output-text" id="outputText" style="color: var(--text-muted); font-style: italic;">
          Select audio and click "Transcribe on Mac CPU" above...
        </div>

        <div class="dual-head-grid">
          <div class="head-card">
            <div class="head-title">⚡ Fast CTC Head (Single-Pass)</div>
            <div class="head-val" id="ctcText">-</div>
          </div>
          <div class="head-card">
            <div class="head-title">🧠 Seq2Seq Attention Head</div>
            <div class="head-val" id="attnText">-</div>
          </div>
        </div>
      </div>

      <div class="output-box" style="margin-top: 1rem;">
        <div class="output-header">
          <span>Ground Truth Reference (Actual Script)</span>
          <button id="saveGtBtn" onclick="saveGroundTruth()" style="display:none; background:var(--accent-en); color:#0f172a; border:none; padding:0.25rem 0.75rem; border-radius:0.25rem; font-size:0.75rem; font-weight:bold; cursor:pointer;">💾 Save</button>
        </div>
        <textarea id="groundTruthText" oninput="document.getElementById('saveGtBtn').style.display='inline-block'" style="width: 100%; min-height: 60px; background: rgba(0,0,0,0.3); color: #cbd5e1; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.1); border-radius: 0.5rem; padding: 0.75rem; font-family: inherit; resize: vertical;" placeholder="Select a clip to see text..."></textarea>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'full';
    let currentLang = 'all';
    let allSamples = [];
    let selectedSample = null;

    async function loadSamples() {
      const res = await fetch('/api/samples?mode=' + currentMode);
      allSamples = await res.json();
      document.getElementById('sampleCountBadge').innerText = allSamples.length + ' files';
      applyFilter();
    }

    function switchMode(mode) {
      // Feature disabled, using full mode only
      currentMode = 'full';
      loadSamples();
    }

    function renderSamples(list) {
      const container = document.getElementById('samplesList');
      container.innerHTML = '';
      list.forEach(sample => {
        const item = document.createElement('div');
        item.className = 'sample-item' + (selectedSample && selectedSample.audio_path === sample.audio_path ? ' selected' : '');
        item.onclick = () => selectSample(sample);
        
        const fname = sample.audio_path.split('/').pop();
        const dur = sample.duration ? parseFloat(sample.duration).toFixed(1) + 's' : '';
        item.innerHTML = `
          <div>
            <div class="sample-name">${fname}</div>
            <div class="sample-dur">${dur}</div>
          </div>
          <span class="lang-pill lang-${sample.language}">${sample.language}</span>
        `;
        container.appendChild(item);
      });
    }

    function filterLang(lang) {
      currentLang = lang;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      applyFilter();
    }

    function applyFilter() {
      if (currentLang === 'all') {
        renderSamples(allSamples);
      } else {
        renderSamples(allSamples.filter(s => s.language === currentLang));
      }
    }

    function selectSample(sample) {
      selectedSample = sample;
      document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
      applyFilter();

      const fname = sample.audio_path.split('/').pop();
      document.getElementById('currentAudioName').innerText = fname + ' (' + sample.language.toUpperCase() + ')';
      
      const player = document.getElementById('audioPlayer');
      player.src = '/audio/' + encodeURIComponent(sample.audio_path);
      player.style.display = 'block';
      player.play();

      document.getElementById('transcribeBtn').disabled = false;
      document.getElementById('groundTruthText').value = sample.transcript;
      document.getElementById('saveGtBtn').style.display = 'none';
      document.getElementById('outputText').innerText = 'Ready to transcribe. Click "Transcribe on Mac CPU" above.';
      document.getElementById('outputText').style.color = 'var(--text-muted)';
      document.getElementById('outputText').style.fontStyle = 'italic';
      document.getElementById('ctcText').innerText = '-';
      document.getElementById('attnText').innerText = '-';
      document.getElementById('metricsRow').style.display = 'none';
      document.getElementById('liveStatusBadge').style.display = 'none';
      document.getElementById('latencyTag').innerText = '';
    }

    async function runInference() {
      if (!selectedSample) return;

      const btn = document.getElementById('transcribeBtn');
      btn.innerText = '⚡ Transcribing on Mac CPU...';
      btn.disabled = true;

      const res = await fetch('/api/transcribe?path=' + encodeURIComponent(selectedSample.audio_path));
      const data = await res.json();

      document.getElementById('outputText').style.color = '#f8fafc';
      document.getElementById('outputText').style.fontStyle = 'normal';
      document.getElementById('outputText').innerText = data.text || '(Silence / Empty)';

      document.getElementById('ctcText').innerText = data.ctc_text || '(Empty)';
      document.getElementById('attnText').innerText = data.attn_text || '(Empty)';

      document.getElementById('rtfVal').innerText = data.rtf + 'x';
      document.getElementById('timeVal').innerText = data.inference_time + 's';
      document.getElementById('durVal').innerText = data.duration + 's';
      document.getElementById('metricsRow').style.display = 'grid';

      document.getElementById('liveStatusBadge').style.display = 'inline-block';
      document.getElementById('latencyTag').innerText = '⚡ ' + data.inference_time + 's (' + data.rtf + 'x RTF)';

      btn.innerText = '▶ Transcribe on Mac CPU';
      btn.disabled = false;
    }

    let mediaRecorder;
    let audioChunks = [];

    async function toggleRecording() {
      const btn = document.getElementById('recordBtn');
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        btn.innerHTML = '🎤 Record Mic';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #b91c1c)';
        document.getElementById('outputText').innerText = 'Processing recording...';
      } else {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
          };
          mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks);
            audioChunks = [];
            
            document.getElementById('transcribeBtn').disabled = true;
            document.getElementById('outputText').innerText = '⚡ Transcribing your voice...';
            document.getElementById('outputText').style.color = 'var(--text-muted)';
            document.getElementById('ctcText').innerText = '-';
            document.getElementById('attnText').innerText = '-';

            const response = await fetch('/api/transcribe_upload', {
              method: 'POST',
              body: audioBlob
            });
            const data = await response.json();
            
            document.getElementById('outputText').style.color = '#f8fafc';
            document.getElementById('outputText').style.fontStyle = 'normal';
            document.getElementById('outputText').innerText = data.text || '(Silence / Empty)';
            document.getElementById('ctcText').innerText = data.ctc_text || '(Empty)';
            document.getElementById('attnText').innerText = data.attn_text || '(Empty)';
            document.getElementById('rtfVal').innerText = data.rtf + 'x';
            document.getElementById('timeVal').innerText = data.inference_time + 's';
            document.getElementById('durVal').innerText = data.duration + 's';
            document.getElementById('metricsRow').style.display = 'grid';
            document.getElementById('liveStatusBadge').style.display = 'inline-block';
            document.getElementById('latencyTag').innerText = '⚡ ' + data.inference_time + 's (' + data.rtf + 'x RTF)';
            document.getElementById('groundTruthText').innerText = '(Live Microphone Recording)';
          };
          
          audioChunks = [];
          mediaRecorder.start();
          btn.innerHTML = '⏹ Stop & Transcribe';
          btn.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
          
          // Clear current selection visual
          document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
          document.getElementById('currentAudioName').innerText = 'Live Microphone Recording...';
          document.getElementById('audioPlayer').style.display = 'none';
          document.getElementById('audioPlayer').pause();
          document.getElementById('transcribeBtn').disabled = true;
          document.getElementById('groundTruthText').innerText = '-';
          
        } catch (err) {
          alert('Microphone access denied or not available.');
        }
      }
    }

    loadSamples();
    async function saveGroundTruth() {
      if(!selectedSample) return;
      const newText = document.getElementById('groundTruthText').value;
      const btn = document.getElementById('saveGtBtn');
      btn.innerText = 'Saving...';
      const res = await fetch('/api/update_transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_path: selectedSample.audio_path,
          transcript: newText,
          mode: currentMode
        })
      });
      if(res.ok) {
        btn.innerText = '✅ Saved!';
        selectedSample.transcript = newText;
        setTimeout(() => { btn.style.display = 'none'; btn.innerText = '💾 Save'; }, 1500);
      } else {
        btn.innerText = '❌ Error';
      }
    }
  </script>
</body>
</html>
"""

class STTStudioHandler(BaseHTTPRequestHandler):
    engine: Optional[STTInference] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif path == "/api/samples":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", ["chunks"])[0]

            if mode == "full":
                metadata_file = PROJECT_ROOT / "data/metadata/metadata.csv"
            else:
                metadata_file = PROJECT_ROOT / "data/metadata/metadata_chunks_clean.csv"
                if not metadata_file.exists():
                    metadata_file = PROJECT_ROOT / "data/metadata/metadata.csv"

            samples = []
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    samples = list(csv.DictReader(f))
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(samples, ensure_ascii=False).encode("utf-8"))

        elif path.startswith("/audio/"):
            raw_path = self.path[len("/audio/"):]
            from urllib.parse import unquote
            audio_rel_path = unquote(raw_path)
            audio_file = PROJECT_ROOT / audio_rel_path

            if audio_file.exists() and audio_file.is_file():
                self.send_response(200)
                mime = mimetypes.guess_type(str(audio_file))[0] or "audio/wav"
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(audio_file.stat().st_size))
                self.end_headers()
                with open(audio_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Audio not found")

        elif path == "/api/transcribe":
            qs = parse_qs(parsed.query)
            target_path = qs.get("path", [""])[0]

            if not target_path:
                self.send_error(400, "Missing path query parameter")
                return

            full_audio_path = PROJECT_ROOT / target_path
            if not full_audio_path.exists():
                self.send_error(404, f"File not found: {target_path}")
                return

            if STTStudioHandler.engine is None:
                STTStudioHandler.engine = STTInference(device="cpu")

            result = STTStudioHandler.engine.transcribe_file(full_audio_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/transcribe_upload":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            import tempfile
            import subprocess
            
            # Save raw upload
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
                f_in.write(post_data)
                upload_path = f_in.name
            
            # Convert to clean WAV format for soundfile to read
            wav_path = upload_path.replace(".webm", ".wav")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", upload_path, 
                    "-ar", "16000", "-ac", "1", wav_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                target_audio = wav_path
            except FileNotFoundError:
                # ffmpeg not installed, try reading raw upload (might fail for webm but good fallback)
                target_audio = upload_path
            
            if STTStudioHandler.engine is None:
                STTStudioHandler.engine = STTInference(device="cpu")
                
            try:
                result = STTStudioHandler.engine.transcribe_file(target_audio)
            except Exception as e:
                result = {"text": f"Error: {e}", "ctc_text": "", "attn_text": "", "duration": 0, "inference_time": 0, "rtf": 0}
            
            # Cleanup
            if os.path.exists(upload_path): os.remove(upload_path)
            if os.path.exists(wav_path): os.remove(wav_path)
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            
        elif parsed.path == "/api/update_transcript":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))
            
            target_path = payload.get("audio_path")
            new_transcript = payload.get("transcript")
            mode = payload.get("mode", "chunks")
            
            if mode == "chunks":
                metadata_file = PROJECT_ROOT / "data/metadata/metadata_chunks_clean.csv"
            else:
                metadata_file = PROJECT_ROOT / "data/metadata/metadata.csv"
                
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                
                for row in rows:
                    if row["audio_path"] == target_path:
                        row["transcript"] = new_transcript
                        break
                        
                with open(metadata_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            
        else:
            self.send_error(404, "Not Found")


def run_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), STTStudioHandler)
    print(f"🚀 STT Studio Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
