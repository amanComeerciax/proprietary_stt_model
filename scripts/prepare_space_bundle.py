import os
import shutil
from pathlib import Path

BUNDLE = Path("hf_space_bundle")
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
BUNDLE.mkdir(parents=True)

# 1. Copy required directories
for d in ["inference", "model", "preprocessing", "tokenizer", "configs"]:
    shutil.copytree(d, BUNDLE / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp"))

# Ensure clean vocab.json is in tokenizer
shutil.copy2("checkpoints/stt_w2v/best/vocab.json", BUNDLE / "tokenizer/vocab.json")

# 2. Copy test wav
shutil.copy2("unseen_test.wav", BUNDLE / "unseen_test.wav")

# 3. Write packages.txt
packages_content = """ffmpeg
libsndfile1
build-essential
cmake
libboost-all-dev
"""
(BUNDLE / "packages.txt").write_text(packages_content, encoding="utf-8")

# 4. Write requirements.txt
requirements_content = """torch>=2.0.0
transformers>=4.35.0
soundfile>=0.12.1
librosa>=0.10.1
pyctcdecode>=0.5.0
scipy>=1.11.0
numpy>=1.24.0
pyyaml>=6.0.1
huggingface_hub
https://github.com/kpu/kenlm/archive/master.zip
"""
(BUNDLE / "requirements.txt").write_text(requirements_content, encoding="utf-8")

# 5. Write README.md with YAML metadata
readme_content = """---
title: Multilingual Speech-to-Text Studio
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Gujarati, Hindi, and English STT Model (CER 5.4%)
---

# 🎙️ Multilingual Speech-to-Text Studio (Gujarati, Hindi, English)

Proprietary Multilingual Speech-to-Text system powered by **fine-tuned MMS-300M (V3 - CER 5.4%)**, **KenLM 3-gram language model**, and an **adaptive smart post-processor with Timestamps & Punctuation**.

### Features:
- **State-of-the-art Accuracy**: 5.4% CER (4.2% Gujarati, 4.6% Hindi, 7.3% English)
- **Word-Level Millisecond Timestamps**: Word start & end times
- **SRT Subtitles Generation**: YouTube / VLC / Premiere ready
- **Smart Multilingual Punctuation & Capitalization**
"""
(BUNDLE / "README.md").write_text(readme_content, encoding="utf-8")

# 6. Write app.py
app_py_content = '''import os
import sys
from pathlib import Path
import gradio as gr
import torch
from huggingface_hub import snapshot_download

try:
    import spaces
except ImportError:
    spaces = None

# Ensure current directory is on python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Automatically download model weights & KenLM from Hugging Face Model Hub
MODEL_REPO = "GlamourAmaan/multilingual-stt-w2v"
print(f"📦 Checking & downloading model assets from {MODEL_REPO} ...")
model_dir = snapshot_download(repo_id=MODEL_REPO)
print(f"✓ Model assets ready at: {model_dir}")

os.environ["STT_CHECKPOINT_DIR"] = model_dir
os.environ["STT_LM_PATH"] = str(Path(model_dir) / "multilingual_3gram.arpa")
os.environ["STT_UNIGRAMS_PATH"] = str(Path(model_dir) / "unigrams.txt")

from inference.inference import STTInference

# Initialize inference engine explicitly with downloaded model directory
print("🚀 Initializing Multilingual STT Engine (V3 - CER 5.4%) ...")
engine = STTInference(model_path=model_dir, device="cpu")
print(f"✓ Engine Loaded Successfully! (is_w2v={engine.is_w2v}, KenLM={engine.lm_decoder is not None})")

if not engine.is_w2v:
    raise RuntimeError(f"CRITICAL: Failed to load MMS-300M model from {model_dir}")
if engine.lm_decoder is None:
    print("⚠️ Warning: KenLM decoder is None, checking arpa path...")

def _predict(audio_path, language):
    if not audio_path:
        return "⚠️ Please speak into the microphone or upload an audio file.", "", "", "", [], {}

    lang_map = {
        "Gujarati (ગુજરાતી)": "gu",
        "Hindi (हिन्दी)": "hi",
        "English": "en",
        "Auto-Detect (સ્વચાલિત / स्वचालित)": "auto"
    }
    target_lang = lang_map.get(language, "gu")

    try:
        # source="mic" activates enhanced DSP + adaptive decoding pipeline
        result = engine.transcribe_file(audio_path, language=target_lang, source="mic")
        
        primary_text = result.get("text", "")
        ctc_text = result.get("ctc_text", "")
        lm_text = result.get("lm_text", "") or primary_text
        srt_subtitles = result.get("srt_subtitles", "")
        word_timestamps = result.get("word_timestamps", [])
        
        metrics = result.get("audio_metrics", {})
        diagnostics = {
            "Audio Duration": metrics.get("audio_duration", f"{result.get('duration', 0)}s"),
            "Input Sample Rate": metrics.get("original_sample_rate", "Unknown"),
            "Model Sample Rate": metrics.get("final_sample_rate", "16,000 Hz"),
            "Channels": metrics.get("num_channels", "1 (Mono)"),
            "RMS Loudness (Pre)": metrics.get("rms_before", "-"),
            "RMS Loudness (Post)": metrics.get("rms_after", "-"),
            "Peak Amplitude": metrics.get("peak_amplitude", "-"),
            "Speech Duration (VAD)": metrics.get("duration_after_trim", "-"),
            "Waveform Tensor Shape": metrics.get("waveform_shape", "-"),
            "Inference Latency": f"{result.get('inference_time', 0):.3f}s",
            "Real-Time Factor (RTF)": f"{result.get('rtf', 0)}x",
            "Hardware Device": result.get("device", "cpu"),
            "Engine Pipeline": "MMS-300M + KenLM + Smart Decoder [V3 - CER 5.4%]"
        }

        return primary_text, ctc_text, lm_text, srt_subtitles, word_timestamps, diagnostics
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", "", "", [], {}

if spaces is not None:
    @spaces.GPU(duration=60)
    def transcribe(audio_path, language):
        if torch.cuda.is_available() and engine.is_w2v and engine.w2v_model is not None:
            engine.w2v_model.to("cuda")
            engine.device = torch.device("cuda")
            try:
                res = _predict(audio_path, language)
            finally:
                engine.w2v_model.to("cpu")
                engine.device = torch.device("cpu")
            return res
        return _predict(audio_path, language)
else:
    def transcribe(audio_path, language):
        return _predict(audio_path, language)

custom_css = """
.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.main-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(56, 189, 248, 0.12) 100%);
    border-radius: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
"""

with gr.Blocks(title="Multilingual STT Studio") as demo:
    with gr.Column():
        gr.HTML("""
        <div class="main-header">
            <h1 style="font-size: 2.1rem; margin-bottom: 0.4rem; font-weight: 800; background: linear-gradient(135deg, #10b981, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎙️ Multilingual Speech-to-Text Studio
            </h1>
            <p style="color: #64748b; font-size: 1rem; margin: 0;">
                Proprietary Speech-to-Text for <strong>Gujarati</strong>, <strong>Hindi</strong>, and <strong>English</strong> (CER 5.4% | Timestamps | Punctuation)
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                lang_selector = gr.Radio(
                    choices=[
                        "Gujarati (ગુજરાતી)",
                        "Hindi (हिन्दी)",
                        "English",
                        "Auto-Detect (સ્વચાલિત / स्वचालित)"
                    ],
                    value="Gujarati (ગુજરાતી)",
                    label="🌐 Target Language Mode"
                )

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎤 Record from Microphone or Upload Audio (.wav, .mp3, .m4a)"
                )

                with gr.Row():
                    clear_btn = gr.Button("Reset")
                    submit_btn = gr.Button("⚡ Transcribe Speech", variant="primary", scale=2)

            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="📝 Final Transcription (Punctuated)",
                    lines=3,
                    placeholder="Your transcribed text will appear here..."
                )

                with gr.Accordion("⏱️ Word-Level Millisecond Timestamps & Subtitles (SRT)", open=True):
                    srt_out = gr.Textbox(label="📄 SRT Subtitles (YouTube / VLC ready)", lines=4)
                    words_out = gr.JSON(label="Word Timestamps List")

                with gr.Accordion("🔍 Acoustic vs Language Model Outputs", open=False):
                    raw_ctc = gr.Textbox(label="⚡ Raw CTC Acoustic Output", lines=2)
                    lm_out = gr.Textbox(label="🧠 KenLM Language Model Output", lines=2)

                with gr.Accordion("📊 Audio Preprocessing & Pipeline Diagnostics", open=False):
                    diag_json = gr.JSON(label="Pipeline Metrics")

        clear_btn.click(lambda: None, None, audio_input)
        submit_btn.click(
            fn=transcribe,
            inputs=[audio_input, lang_selector],
            outputs=[output_text, raw_ctc, lm_out, srt_out, words_out, diag_json]
        )
        audio_input.stop_recording(
            fn=transcribe,
            inputs=[audio_input, lang_selector],
            outputs=[output_text, raw_ctc, lm_out, srt_out, words_out, diag_json]
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=custom_css)
'''
(BUNDLE / "app.py").write_text(app_py_content, encoding="utf-8")

print("✓ hf_space_bundle created successfully!")
