import os
import sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    try:
        from huggingface_hub import HfFolder
        TOKEN = HfFolder.get_token()
    except Exception:
        pass

REPO_ID = "GlamourAmaan/multilingual-stt-w2v"
BUNDLE_DIR = "hf_model_bundle"

print(f"📡 Connecting to Hugging Face Model Hub: {REPO_ID} ...")
api = HfApi(token=TOKEN)

print(f"📦 Uploading model weights & LM files to Model Repo '{REPO_ID}' (1.3 GB) ...")
print("Uploading in progress...")

future = api.upload_folder(
    folder_path=BUNDLE_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload MMS-300M Fine-Tuned Checkpoints & KenLM 3-Gram"
)

print(f"🎉 Model Upload Complete! URL: https://huggingface.co/{REPO_ID}")
