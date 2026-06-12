"""
Download Kronos model weights to local ./models/ folder.
Run this ONCE on a network WITHOUT a firewall blocking HuggingFace.
After this, the forecast app/script works offline.

Usage: python download_models.py
"""
import os
from huggingface_hub import snapshot_download

MODELS = [
    "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-small",
    # Uncomment if you want other sizes:
    # "NeoQuasar/Kronos-Tokenizer-2k",
    # "NeoQuasar/Kronos-mini",
    # "NeoQuasar/Kronos-base",
]

local_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(local_dir, exist_ok=True)

for repo in MODELS:
    name = repo.split("/")[-1]
    target = os.path.join(local_dir, name)
    print(f"Downloading {repo} -> {target} ...")
    snapshot_download(repo_id=repo, local_dir=target)
    print(f"  Done: {name}")

print("\nAll models downloaded to ./models/")
print("The forecast app will now use these offline.")
