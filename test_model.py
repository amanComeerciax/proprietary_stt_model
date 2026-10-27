import sys
from inference.inference import STTInference

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bhai, aise run karo: PYTHONPATH=. python3 test_model.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    
    engine = STTInference(device="cpu")
    
    print(f"\nTesting on: {audio_file}")
    print("-" * 50)
    
    try:
        result = engine.transcribe_file(audio_file)
        print(f"Model Output Text: {result['text']}")
    except Exception as e:
        print(f"Error aagaya: {e}")
