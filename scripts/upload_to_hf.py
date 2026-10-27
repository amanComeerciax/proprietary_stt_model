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

REPO_ID = "GlamourAmaan/multilingual-speech-to-text"
BUNDLE_DIR = "hf_space_bundle"

print(f"📡 Connecting to Hugging Face Space: {REPO_ID} ...")
api = HfApi(token=TOKEN)

user = api.whoami()
print(f"✓ Authenticated as: {user['name']}")

print(f"📦 Uploading bundle directory '{BUNDLE_DIR}' to Space (1.3 GB model + code) ...")
print("This may take 1-3 minutes depending on network speed...")

future = api.upload_folder(
    folder_path=BUNDLE_DIR,
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Deploy Multilingual Speech-to-Text Model to Hugging Face Spaces"
)

print(f"🎉 Upload Complete! Commit info: {future}")
print(f"🚀 Space is now building at: https://huggingface.co/spaces/{REPO_ID}")
