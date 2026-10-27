"""
Quick Multilingual Inference Tester with KenLM Beam Search Decoder.

Usage:
    python3 test_w2v.py <audio_wav_path>
    python3 test_w2v.py unseen_test.wav
"""

import sys
from pathlib import Path
from inference.inference import STTInference


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_w2v.py <audio_wav_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    lang = "gu"
    if len(sys.argv) > 2:
        lang = sys.argv[2]
    if not Path(audio_path).exists():
        print(f"[!] File not found: {audio_path}")
        sys.exit(1)

    engine = STTInference(device="cpu")
    print(f"\n[*] Transcribing audio: {audio_path} (Language Mode: {lang.upper()})")
    res = engine.transcribe_file(audio_path, language=lang)

    print("\n" + "=" * 65)
    print("🎙️ TRANSCRIPTION RESULT (Multilingual STT):")
    print(f"   ⚡ Raw CTC Output       : \"{res.get('ctc_text', '')}\"")
    print(f"   🧠 KenLM Beam Search   : \"{res.get('lm_text', res.get('text', ''))}\"")
    print(f"   ⏱️ Audio Duration       : {res['duration']}s")
    print(f"   ⚡ CPU Latency          : {res['inference_time']}s (RTF: {res['rtf']}x)")
    print(f"   🏷️ Active Model         : {res.get('model_name', 'MMS-300M')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
