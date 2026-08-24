#!/usr/bin/env bash
# Pre-download every checkpoint this repo needs into $HF_HOME (scratch, not $HOME).
#
#   source env.sh && bash scripts/fetch_models.sh                # 5 paper models, ~166 GB
#   source env.sh && WITH_JUDGE=1 bash scripts/fetch_models.sh   # + 70B judge, ~+40 GB
#
# Resumable and content-addressed: re-running skips what is already present.
# gemma-3-12b-it is GATED -- HF_TOKEN must be set (env.sh does it).
# phi-4 is deliberately absent; it was dropped from the paper (CLAUDE.md s7).
#
# --exclude "*.bin": only allenai/OLMo-2-1124-13B-DPO ships both formats, and its
# .bin copy is a redundant 27.4 GB. Every model here has safetensors, which is
# what transformers loads. NOTE --exclude takes ONE pattern per flag; passing
# several after a single --exclude silently reinterprets the extras as positional
# FILENAMES, and `hf download` then "succeeds" having fetched nothing.
#
# Exit code is NOT evidence of work -- that is exactly how the bug above hid --
# so every repo is verified against the Hub's own file list before we move on.
set -euo pipefail

: "${HF_HOME:?source env.sh first}"
: "${HF_TOKEN:?source env.sh first}"

models=(
  tiiuae/Falcon3-10B-Instruct
  Qwen/Qwen1.5-14B-Chat
  Qwen/Qwen1.5-32B-Chat
  google/gemma-3-12b-it
  allenai/OLMo-2-1124-13B-DPO
)
if [ "${WITH_JUDGE:-0}" = "1" ]; then
  models+=( unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit )
fi

for m in "${models[@]}"; do
  echo "=== [$(date -Is)] $m ==="
  hf download "$m" --exclude "*.bin" || { echo "FAILED download: $m"; exit 1; }

  python - "$m" <<'PY' || { echo "FAILED verify: $m"; exit 1; }
import sys, os
from huggingface_hub import HfApi, snapshot_download
repo = sys.argv[1]
want = {f.rfilename: (f.size or 0)
        for f in HfApi().model_info(repo, files_metadata=True).siblings
        if not f.rfilename.endswith(".bin")}
root = snapshot_download(repo, allow_patterns=[], local_files_only=True)
missing, short = [], []
for name, size in want.items():
    p = os.path.join(root, name)
    if not os.path.exists(p):
        missing.append(name)
    elif size and os.path.getsize(p) != size:
        short.append(name)
have = sum(os.path.getsize(os.path.join(root, n))
           for n in want if os.path.exists(os.path.join(root, n)))
print(f"    verified {len(want)-len(missing)-len(short)}/{len(want)} files, "
      f"{have/1e9:.1f} GB at {root}")
if missing or short:
    print(f"    MISSING {len(missing)}: {missing[:5]}")
    print(f"    WRONG SIZE {len(short)}: {short[:5]}")
    sys.exit(1)
PY
done

echo "=== [$(date -Is)] all repos complete ==="
du -sh "$HF_HOME"
