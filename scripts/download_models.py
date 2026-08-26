#!/usr/bin/env python
"""Download the paper's models into the shared HF cache.

Sequential and resumable: snapshot_download skips blobs already present, so
re-running after an interruption is safe and cheap.

gemma-3-12b-it is deliberately absent -- it is gated=manual and needs an HF
token from an account that has accepted Google's Gemma licence. Add it here
once HF_TOKEN is set.

Duplicate weight formats are skipped: OLMo-2-13B-DPO ships both .safetensors
and .bin (27 GB each) and transformers prefers safetensors.
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

MODELS = [
    "tiiuae/Falcon3-10B-Instruct",
    "Qwen/Qwen1.5-14B-Chat",
    "allenai/OLMo-2-1124-13B-DPO",
    "Qwen/Qwen1.5-32B-Chat",
    # judge model, CLAUDE.md section 5
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
]

IGNORE = ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.gguf", "*.onnx"]

token = os.environ.get("HF_TOKEN") or None

failed = []
for repo in MODELS:
    print(f"\n=== {repo}", flush=True)
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            p = snapshot_download(
                repo_id=repo,
                token=token,
                ignore_patterns=IGNORE,
                max_workers=8,
            )
            print(f"    OK  {p}  ({time.time() - t0:.0f}s)", flush=True)
            break
        except Exception as e:
            print(f"    attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            if attempt == 3:
                failed.append(repo)
            else:
                time.sleep(10)

print("\n=== summary")
for repo in MODELS:
    print(f"  {'FAILED' if repo in failed else 'ok    '}  {repo}")
sys.exit(1 if failed else 0)
