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
      <div class="badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);">MMS-300M Fine-Tuned • Best CER: 9.1% (Gujarati 6.5%)</div>
      <div>
        <h1>Multilingual Speech-to-Text Studio</h1>
        <div class="subtitle">Gujarati (6.5% CER) • Hindi (8.4% CER) • English (12.6% CER) | Ultra-Fast Mac CPU</div>
      </div>
    </div>
  </header>

  <div class="grid">
    <!-- Left: Dataset Samples Explorer -->
    <div class="card">
      <div class="card-title">
        <span>🎧 Audio Validation Benchmarks</span>
        <span id="sampleCountBadge" style="font-size: 0.8rem; color: var(--text-muted);">1,242 clips</span>
      </div>

      <div class="filter-tabs">
        <button class="tab-btn active" onclick="filterLang('all')">All</button>
        <button class="tab-btn" onclick="filterLang('gu')">Gujarati</button>
        <button class="tab-btn" onclick="filterLang('hi')">Hindi</button>
        <button class="tab-btn" onclick="filterLang('en')">English</button>
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

      <!-- Language Mode Priority Selector -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem; padding: 0.55rem 0.85rem; background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); border-radius: 0.65rem;">
        <span style="font-size: 0.82rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em;">Language Priority:</span>
        <div style="display: flex; gap: 0.4rem;" id="langModeSelector">
          <button type="button" class="lang-pill" id="pill_gu" onclick="setLanguageMode('gu')" style="padding: 0.32rem 0.8rem; border-radius: 20px; border: 1px solid #10b981; background: #10b981; color: #fff; font-weight: 600; font-size: 0.8rem; cursor: pointer;">🟢 Gujarati</button>
          <button type="button" class="lang-pill" id="pill_hi" onclick="setLanguageMode('hi')" style="padding: 0.32rem 0.8rem; border-radius: 20px; border: 1px solid var(--card-border); background: rgba(255,255,255,0.05); color: #cbd5e1; font-weight: 500; font-size: 0.8rem; cursor: pointer;">🟠 Hindi</button>
          <button type="button" class="lang-pill" id="pill_en" onclick="setLanguageMode('en')" style="padding: 0.32rem 0.8rem; border-radius: 20px; border: 1px solid var(--card-border); background: rgba(255,255,255,0.05); color: #cbd5e1; font-weight: 500; font-size: 0.8rem; cursor: pointer;">🔵 English</button>
          <button type="button" class="lang-pill" id="pill_auto" onclick="setLanguageMode('auto')" style="padding: 0.32rem 0.8rem; border-radius: 20px; border: 1px solid var(--card-border); background: rgba(255,255,255,0.05); color: #cbd5e1; font-weight: 500; font-size: 0.8rem; cursor: pointer;">⚪ Auto</button>
        </div>
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
          🎤 Record Mic (Lossless WAV)
        </button>
      </div>

      <!-- Hardware Mic Enhancements Checkboxes -->
      <div style="display: flex; gap: 0.9rem; align-items: center; justify-content: center; margin-top: 0.65rem; padding: 0.45rem 0.85rem; background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); border-radius: 0.5rem; font-size: 0.78rem; color: #94a3b8;">
        <span style="font-weight: 600; color: #cbd5e1;">Hardware Mic:</span>
        <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer; user-select: none;">
          <input type="checkbox" id="chkEcho" checked style="accent-color: #10b981;"> Echo Cancel
        </label>
        <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer; user-select: none;">
          <input type="checkbox" id="chkNoise" checked style="accent-color: #10b981;"> Noise Suppress
        </label>
        <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer; user-select: none;">
          <input type="checkbox" id="chkAgc" checked style="accent-color: #10b981;"> Auto Gain
        </label>
      </div>


      <!-- Dedicated Live Audio VU Level Meter -->
      <div id="liveMeterContainer" style="display: none; margin-top: 0.75rem; background: rgba(0,0,0,0.5); border-radius: 0.65rem; padding: 0.65rem 0.9rem; border: 1px solid rgba(255,255,255,0.15);">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.4rem;">
          <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ef4444;"></span>
            <strong style="color: #f8fafc;">Live Mic Volume</strong>
          </span>
          <span id="liveMeterPct" style="font-weight: 700; color: #10b981; font-size: 0.9rem;">0%</span>
        </div>
        <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
          <div id="liveMeterBar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #10b981 0%, #f59e0b 70%, #ef4444 100%); transition: width 0.08s ease;"></div>
        </div>
      </div>

      <div class="metrics-row" id="metricsRow" style="display: none;">
        <div class="metric-card">
          <div class="metric-val" id="cerVal" style="color: #10b981;">-</div>
          <div class="metric-lbl">Accuracy / CER</div>
        </div>
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
            <div class="head-title">⚡ Raw CTC Output (Acoustic)</div>
            <div class="head-val" id="ctcText">-</div>
          </div>
          <div class="head-card">
            <div class="head-title">🧠 KenLM Language Model (Corrected)</div>
            <div class="head-val" id="attnText" style="color: #10b981;">-</div>
          </div>
        </div>
      </div>

      <!-- Word Timestamps & Interactive Subtitles (SRT) Card -->
      <div class="output-box" id="timestampsCard" style="display: none; margin-top: 1rem; border-color: rgba(16, 185, 129, 0.35);">
        <div class="output-header">
          <span style="display: flex; align-items: center; gap: 0.5rem; color: #10b981; font-weight: 600;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            Word-Level Timestamps (Click word to play audio from that exact second)
          </span>
          <span id="wordCountBadge" style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 0.2rem 0.5rem; border-radius: 12px; font-weight: 600;">0 words</span>
        </div>
        
        <!-- Interactive Word Chips Container -->
        <div id="wordChipsContainer" style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; max-height: 180px; overflow-y: auto; padding-right: 0.25rem;">
        </div>

        <!-- SubRip (SRT) Subtitle Block -->
        <div style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.75rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.82rem; font-weight: 600; color: #cbd5e1;">📄 SRT Subtitle Format (YouTube / Premiere / VLC Ready)</span>
            <div style="display: flex; gap: 0.5rem;">
              <button onclick="copySrtToClipboard()" style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); color: #38bdf8; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; cursor: pointer;">📋 Copy SRT</button>
              <button onclick="downloadSrtFile()" style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #10b981; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; cursor: pointer;">⬇ Download .srt</button>
            </div>
          </div>
          <pre id="srtTextContainer" style="background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); border-radius: 0.4rem; padding: 0.75rem; font-family: monospace; font-size: 0.8rem; color: #94a3b8; max-height: 140px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4; margin: 0;"></pre>
        </div>
      </div>

      <div class="output-box" style="margin-top: 1rem;">
        <div class="output-header">
          <span>Ground Truth Reference (Actual Script)</span>
          <button id="saveGtBtn" onclick="saveGroundTruth()" style="display:none; background:var(--accent-en); color:#0f172a; border:none; padding:0.25rem 0.75rem; border-radius:0.25rem; font-size:0.75rem; font-weight:bold; cursor:pointer;">💾 Save</button>
        </div>
        <textarea id="groundTruthText" oninput="updateLiveCer(); document.getElementById('saveGtBtn').style.display='inline-block'" style="width: 100%; min-height: 60px; background: rgba(0,0,0,0.3); color: #cbd5e1; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.1); border-radius: 0.5rem; padding: 0.75rem; font-family: inherit; resize: vertical;" placeholder="Type or view ground truth script here..."></textarea>
      </div>

      <!-- Audio Preprocessing & Pipeline Diagnostics Panel -->
      <div class="output-box" id="audioMetricsCard" style="display: none; margin-top: 1rem; border-color: rgba(56, 189, 248, 0.35); background: rgba(15, 23, 42, 0.75);">
        <div class="output-header" style="cursor: pointer; user-select: none;" onclick="toggleMetricsBody()">
          <span style="display: flex; align-items: center; gap: 0.5rem; color: #38bdf8; font-weight: 600;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5v14M7 8v8M2 11v2M22 11v2"/></svg>
            Audio Preprocessing & Pipeline Diagnostics
          </span>
          <span id="metricsToggleIcon" style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">▲ Collapse</span>
        </div>
        <div id="metricsBody" style="display: block; margin-top: 0.75rem;">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.65rem; font-size: 0.82rem;">
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Audio Duration:</span>
              <span id="mAudioDuration" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Original Sample Rate:</span>
              <span id="mOrigSr" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Final Sample Rate:</span>
              <span id="mFinalSr" style="font-weight: 600; color: #10b981;">16,000 Hz</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Number of Channels:</span>
              <span id="mChannels" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">RMS Before Processing:</span>
              <span id="mRmsBefore" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">RMS After Processing:</span>
              <span id="mRmsAfter" style="font-weight: 600; color: #38bdf8;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Peak Amplitude:</span>
              <span id="mPeakAmp" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Duration After Trimming:</span>
              <span id="mTrimmedDur" style="font-weight: 600; color: #f1f5f9;">-</span>
            </div>
            <div style="grid-column: 1 / -1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 0.4rem; padding: 0.5rem 0.65rem; display: flex; justify-content: space-between; align-items: center;">
              <span style="color: #94a3b8; font-size: 0.78rem;">Waveform Shape:</span>
              <span id="mWaveformShape" style="font-family: monospace; font-weight: 600; color: #f59e0b;">-</span>
            </div>
            <div style="grid-column: 1 / -1; background: rgba(0,0,0,0.4); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 0.4rem; padding: 0.65rem 0.8rem;">
              <span style="display: block; font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.25rem;">Model Output:</span>
              <span id="mModelOutput" style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">-</span>
            </div>
          </div>
        </div>
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
      const tsCard = document.getElementById('timestampsCard');
      if (tsCard) tsCard.style.display = 'none';
    }

    function computeCer(ref, hyp) {
      if (!ref || ref.length === 0) return '0.0';
      const d = [];
      for (let i = 0; i <= ref.length; i++) d[i] = [i];
      for (let j = 0; j <= hyp.length; j++) d[0][j] = j;
      for (let i = 1; i <= ref.length; i++) {
        for (let j = 1; j <= hyp.length; j++) {
          const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
          d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
        }
      }
      return ((d[ref.length][hyp.length] / ref.length) * 100).toFixed(1);
    }

    function toggleMetricsBody() {
      const body = document.getElementById('metricsBody');
      const icon = document.getElementById('metricsToggleIcon');
      if (body.style.display === 'none') {
        body.style.display = 'block';
        icon.innerText = '▲ Collapse';
      } else {
        body.style.display = 'none';
        icon.innerText = '▼ Expand';
      }
    }

    function updateAudioMetrics(metrics, modelText) {
      const card = document.getElementById('audioMetricsCard');
      if (!card) return;
      if (!metrics) {
        card.style.display = 'none';
        return;
      }
      card.style.display = 'block';
      const body = document.getElementById('metricsBody');
      if (body) body.style.display = 'block';
      const icon = document.getElementById('metricsToggleIcon');
      if (icon) icon.innerText = '▲ Collapse';

      document.getElementById('mAudioDuration').innerText = metrics.audio_duration || '-';
      document.getElementById('mOrigSr').innerText = metrics.original_sample_rate || '-';
      document.getElementById('mFinalSr').innerText = metrics.final_sample_rate || '16,000 Hz';
      document.getElementById('mChannels').innerText = metrics.num_channels || '-';
      document.getElementById('mRmsBefore').innerText = metrics.rms_before || '-';
      document.getElementById('mRmsAfter').innerText = metrics.rms_after || '-';
      document.getElementById('mPeakAmp').innerText = metrics.peak_amplitude || '-';
      document.getElementById('mTrimmedDur').innerText = metrics.duration_after_trim || '-';
      document.getElementById('mWaveformShape').innerText = metrics.waveform_shape || '-';
      document.getElementById('mModelOutput').innerText = modelText || '-';
    }

    let currentLangMode = 'gu';
    function setLanguageMode(lang) {
      currentLangMode = lang;
      const pills = ['gu', 'hi', 'en', 'auto'];
      pills.forEach(p => {
        const el = document.getElementById('pill_' + p);
        if (el) {
          el.style.background = 'rgba(255,255,255,0.05)';
          el.style.color = '#cbd5e1';
          el.style.borderColor = 'var(--card-border)';
          el.style.fontWeight = '500';
        }
      });
      const activeEl = document.getElementById('pill_' + lang);
      if (activeEl) {
        activeEl.style.fontWeight = '600';
        activeEl.style.color = '#fff';
        if (lang === 'gu') { activeEl.style.background = '#10b981'; activeEl.style.borderColor = '#10b981'; }
        else if (lang === 'hi') { activeEl.style.background = '#f59e0b'; activeEl.style.borderColor = '#f59e0b'; }
        else if (lang === 'en') { activeEl.style.background = '#38bdf8'; activeEl.style.borderColor = '#38bdf8'; }
        else { activeEl.style.background = '#6366f1'; activeEl.style.borderColor = '#6366f1'; }
      }
    }

    async function runInference() {
      if (!selectedSample) return;

      const btn = document.getElementById('transcribeBtn');
      btn.innerText = '⚡ Transcribing on Mac CPU...';
      btn.disabled = true;

      const res = await fetch('/api/transcribe?path=' + encodeURIComponent(selectedSample.audio_path) + '&lang=' + currentLangMode);
      const data = await res.json();

      document.getElementById('outputText').style.color = '#f8fafc';
      document.getElementById('outputText').style.fontStyle = 'normal';
      document.getElementById('outputText').innerText = data.text || '(Silence / Empty)';

      document.getElementById('ctcText').innerText = data.ctc_text || '(Empty)';
      document.getElementById('attnText').innerText = data.lm_text || data.text || '(Empty)';

      if (selectedSample && selectedSample.transcript) {
        const cer = computeCer(selectedSample.transcript.trim(), (data.text || '').trim());
        document.getElementById('cerVal').innerText = cer + '%';
      } else {
        document.getElementById('cerVal').innerText = 'N/A';
      }

      document.getElementById('rtfVal').innerText = data.rtf + 'x';
      document.getElementById('timeVal').innerText = data.inference_time + 's';
      document.getElementById('durVal').innerText = data.duration + 's';
      document.getElementById('metricsRow').style.display = 'grid';

      document.getElementById('liveStatusBadge').style.display = 'inline-block';
      document.getElementById('latencyTag').innerText = '⚡ ' + data.inference_time + 's (' + data.rtf + 'x RTF)';

      if (data.audio_metrics) {
        updateAudioMetrics(data.audio_metrics, data.text);
      }

      renderTimestampsAndSrt(data.word_timestamps, data.srt_subtitles);

      btn.innerText = '▶ Transcribe on Mac CPU';
      btn.disabled = false;
    }

    let latestSrtText = '';

    function renderTimestampsAndSrt(wordTimestamps, srtSubtitles) {
      const card = document.getElementById('timestampsCard');
      const container = document.getElementById('wordChipsContainer');
      const srtContainer = document.getElementById('srtTextContainer');
      const badge = document.getElementById('wordCountBadge');

      if (!wordTimestamps || wordTimestamps.length === 0) {
        if (card) card.style.display = 'none';
        return;
      }

      if (card) card.style.display = 'block';
      if (badge) badge.innerText = wordTimestamps.length + ' words';
      latestSrtText = srtSubtitles || '';

      if (container) {
        container.innerHTML = '';
        wordTimestamps.forEach(item => {
          const chip = document.createElement('div');
          chip.className = 'word-time-chip';
          chip.style.cssText = 'background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; padding: 0.35rem 0.65rem; font-size: 0.85rem; cursor: pointer; transition: all 0.15s ease; display: inline-flex; flex-direction: column; align-items: flex-start; user-select: none;';
          chip.title = `Click to play audio from ${item.start}s`;
          chip.onmouseover = () => { chip.style.background = 'rgba(16, 185, 129, 0.2)'; chip.style.borderColor = '#10b981'; };
          chip.onmouseout = () => { chip.style.background = 'rgba(255,255,255,0.05)'; chip.style.borderColor = 'rgba(255,255,255,0.12)'; };
          chip.onclick = () => {
            const player = document.getElementById('audioPlayer');
            if (player) {
              player.currentTime = item.start;
              player.play();
            }
          };
          chip.innerHTML = `
            <span style="font-weight: 600; color: #f8fafc;">${item.word}</span>
            <span style="font-size: 0.68rem; color: #10b981; font-family: monospace; margin-top: 0.15rem;">${item.start.toFixed(2)}s - ${item.end.toFixed(2)}s</span>
          `;
          container.appendChild(chip);
        });
      }

      if (srtContainer) {
        srtContainer.innerText = latestSrtText || '(No SRT generated)';
      }
    }

    function copySrtToClipboard() {
      if (!latestSrtText) return;
      navigator.clipboard.writeText(latestSrtText).then(() => {
        alert('✓ SRT Subtitles copied to clipboard!');
      });
    }

    function downloadSrtFile() {
      if (!latestSrtText) return;
      const blob = new Blob([latestSrtText], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'subtitles.srt';
      a.click();
      URL.revokeObjectURL(url);
    }

    let isRecording = false;
    let micStream = null;
    let audioCtx = null;
    let scriptProcessor = null;
    let pcmChunks = [];
    let animId = null;

    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      }
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM
      view.setUint16(22, 1, true); // mono
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * 1 channel * 2 bytes)
      view.setUint16(32, 2, true); // block align
      view.setUint16(34, 16, true); // 16-bit
      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return new Blob([view], { type: 'audio/wav' });
    }

    function downsampleBuffer(buffer, sampleRate, outSampleRate) {
      if (outSampleRate === sampleRate) return buffer;
      if (outSampleRate > sampleRate) return buffer;
      const sampleRateRatio = sampleRate / outSampleRate;
      const newLength = Math.round(buffer.length / sampleRateRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0;
      let offsetBuffer = 0;
      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = count > 0 ? accum / count : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    async function toggleRecording() {
      const btn = document.getElementById('recordBtn');
      const meterContainer = document.getElementById('liveMeterContainer');
      const meterBar = document.getElementById('liveMeterBar');
      const meterPct = document.getElementById('liveMeterPct');

      if (isRecording) {
        isRecording = false;
        btn.innerHTML = '🎤 Record Mic (Lossless WAV)';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #b91c1c)';
        document.getElementById('outputText').innerText = 'Processing lossless recording...';

        if (animId) {
          cancelAnimationFrame(animId);
          animId = null;
        }
        if (meterContainer) meterContainer.style.display = 'none';

        if (scriptProcessor) {
          scriptProcessor.disconnect();
          scriptProcessor = null;
        }
        if (micStream) {
          micStream.getTracks().forEach(t => t.stop());
          micStream = null;
        }

        const sampleRate = audioCtx ? audioCtx.sampleRate : 16000;
        if (audioCtx) {
          audioCtx.close();
          audioCtx = null;
        }

        let totalLen = pcmChunks.reduce((acc, c) => acc + c.length, 0);
        if (totalLen === 0) {
          document.getElementById('outputText').innerText = 'No audio detected. Please try speaking again.';
          return;
        }
        let merged = new Float32Array(totalLen);
        let offset = 0;
        for (let c of pcmChunks) {
          merged.set(c, offset);
          offset += c.length;
        }
        pcmChunks = [];

        let final16k = merged;
        if (sampleRate !== 16000) {
          final16k = downsampleBuffer(merged, sampleRate, 16000);
        }
        const audioBlob = encodeWAV(final16k, 16000);

        // Load into player so user can immediately hear what was recorded
        const audioUrl = URL.createObjectURL(audioBlob);
        const player = document.getElementById('audioPlayer');
        player.src = audioUrl;
        player.style.display = 'block';
        document.getElementById('currentAudioName').innerText = '🎤 Your Mic Recording (Lossless WAV — click Play below)';

        document.getElementById('transcribeBtn').disabled = true;
        document.getElementById('outputText').innerText = '⚡ Transcribing studio audio...';
        document.getElementById('outputText').style.color = 'var(--text-muted)';
        document.getElementById('ctcText').innerText = '-';
        document.getElementById('attnText').innerText = 'MMS-300M + KenLM + Smart Decoder';

        const response = await fetch('/api/transcribe_upload?lang=' + currentLangMode + '&orig_sr=' + sampleRate, {
          method: 'POST',
          headers: { 'Content-Type': 'audio/wav' },
          body: audioBlob
        });
        const data = await response.json();

        document.getElementById('outputText').style.color = '#f8fafc';
        document.getElementById('outputText').style.fontStyle = 'normal';
        document.getElementById('outputText').innerText = data.text || '(Silence / Empty)';
        document.getElementById('ctcText').innerText = data.ctc_text || '(Empty)';
        document.getElementById('attnText').innerText = data.lm_text || data.text || '(Empty)';

        const refText = document.getElementById('groundTruthText').value.trim();
        if (refText && refText !== '(Your live microphone voice)' && refText !== 'Recording in progress...') {
          const cer = computeCer(refText, data.text || '');
          document.getElementById('cerVal').innerText = cer + '%';
        } else {
          document.getElementById('cerVal').innerText = '0.0% CER';
        }

        document.getElementById('rtfVal').innerText = data.rtf + 'x';
        document.getElementById('timeVal').innerText = data.inference_time + 's';
        document.getElementById('durVal').innerText = data.duration + 's';
        document.getElementById('metricsRow').style.display = 'grid';
        document.getElementById('liveStatusBadge').style.display = 'inline-block';
        document.getElementById('latencyTag').innerText = '⚡ ' + data.inference_time + 's (' + data.rtf + 'x RTF)';
        document.getElementById('groundTruthText').value = data.text || '';

        if (data.audio_metrics) {
          updateAudioMetrics(data.audio_metrics, data.text);
        }

        renderTimestampsAndSrt(data.word_timestamps, data.srt_subtitles);
      } else {
        try {
          const echoVal = document.getElementById('chkEcho') ? document.getElementById('chkEcho').checked : true;
          const noiseVal = document.getElementById('chkNoise') ? document.getElementById('chkNoise').checked : true;
          const agcVal = document.getElementById('chkAgc') ? document.getElementById('chkAgc').checked : true;

          micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: { ideal: 16000 },
              echoCancellation: echoVal,
              noiseSuppression: noiseVal,
              autoGainControl: agcVal
            }
          });

          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          if (audioCtx.state === 'suspended') {
            await audioCtx.resume();
          }

          const source = audioCtx.createMediaStreamSource(micStream);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 256;
          source.connect(analyser);

          scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
          pcmChunks = [];
          scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const channelData = e.inputBuffer.getChannelData(0);
            pcmChunks.push(new Float32Array(channelData));
          };
          source.connect(scriptProcessor);
          scriptProcessor.connect(audioCtx.destination);

          isRecording = true;
          btn.innerHTML = '⏹ Stop & Transcribe';
          btn.style.background = 'linear-gradient(135deg, #10b981, #059669)';

          // Clear current selection visual
          document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
          document.getElementById('currentAudioName').innerText = '🔴 Recording lossless voice... Speak now!';
          document.getElementById('audioPlayer').style.display = 'none';
          document.getElementById('audioPlayer').pause();
          document.getElementById('transcribeBtn').disabled = true;
          document.getElementById('groundTruthText').value = 'Recording in progress...';

          if (meterContainer) {
            meterContainer.style.display = 'block';
            meterBar.style.width = '0%';
            meterPct.innerText = '0% Volume';
          }

          const pcmData = new Uint8Array(analyser.frequencyBinCount);
          function updateMeter() {
            if (!isRecording) return;
            analyser.getByteFrequencyData(pcmData);
            let sum = 0;
            for (let i = 0; i < pcmData.length; i++) sum += pcmData[i];
            const avg = sum / pcmData.length;
            const pct = Math.min(100, Math.round((avg / 64) * 100));
            if (meterBar) meterBar.style.width = pct + '%';
            if (meterPct) meterPct.innerText = pct + '% Volume';
            animId = requestAnimationFrame(updateMeter);
          }
          updateMeter();
        } catch(e) {
          alert("Microphone Access Error: " + e.message);
        }
      }
    }

    function updateLiveCer() {
      const ref = document.getElementById('groundTruthText').value.trim();
      const hyp = document.getElementById('outputText').innerText.trim();
      if (ref && hyp && ref !== '(Your live microphone voice)' && ref !== 'Recording in progress...') {
        const cer = computeCer(ref, hyp);
        document.getElementById('cerVal').innerText = cer + '%';
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
            metadata_file = PROJECT_ROOT / "data/metadata/combined_val.csv"
            if not metadata_file.exists():
                metadata_file = PROJECT_ROOT / "data/metadata/metadata_all.csv"
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
            lang = qs.get("lang", ["gu"])[0]

            if not target_path:
                self.send_error(400, "Missing path query parameter")
                return

            full_audio_path = PROJECT_ROOT / target_path
            if not full_audio_path.exists():
                self.send_error(404, f"File not found: {target_path}")
                return

            if STTStudioHandler.engine is None:
                STTStudioHandler.engine = STTInference(device="cpu")

            result = STTStudioHandler.engine.transcribe_file(full_audio_path, language=lang, source="file")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/transcribe_upload":
            qs = parse_qs(parsed.query)
            lang = qs.get("lang", ["gu"])[0]
            try:
                orig_sr_hint = int(qs.get("orig_sr", [0])[0]) or None
            except Exception:
                orig_sr_hint = None
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            import tempfile
            import subprocess
            
            # Save raw upload
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                f_in.write(post_data)
                upload_path = f_in.name
            
            # Convert to clean WAV format for soundfile to read
            wav_path = upload_path + ".wav"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", upload_path, 
                    "-ar", "16000", "-ac", "1", wav_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                target_audio = wav_path if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44 else upload_path
            except Exception:
                target_audio = upload_path
            
            if STTStudioHandler.engine is None:
                STTStudioHandler.engine = STTInference(device="cpu")
                
            try:
                # source="mic" activates enhanced DSP + adaptive decoding pipeline
                result = STTStudioHandler.engine.transcribe_file(target_audio, language=lang, source="mic", orig_sr_hint=orig_sr_hint)
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
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("0.0.0.0", port), STTStudioHandler)
    print(f"🚀 STT Studio Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
