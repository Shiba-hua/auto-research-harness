#!/usr/bin/env bash
# Download Qwen/Qwen3-32B-AWQ into /dev/shm/models (visible as /models in the PRoot guest).
# huggingface.co is unreachable from this host; hf-mirror.com works (verified 200, 0.36s).
# Resumable: huggingface_hub re-uses already-complete files.
set -uo pipefail

EV=/root/dsh-harness/evidence/model-download
mkdir -p "$EV"
LOG="$EV/download.log"
: > "$LOG"

export HF_ENDPOINT=https://hf-mirror.com

ts() { date -u +%FT%TZ; }
echo "[$(ts)] start Qwen/Qwen3-32B-AWQ -> /models/Qwen3-32B-AWQ (HF_ENDPOINT=$HF_ENDPOINT)" | tee -a "$LOG"

/root/dsh-harness/bin/vllm-run python - >>"$LOG" 2>&1 <<'PY'
import time, os, sys
from huggingface_hub import snapshot_download

t0 = time.time()
try:
    p = snapshot_download(
        repo_id="Qwen/Qwen3-32B-AWQ",
        local_dir="/models/Qwen3-32B-AWQ",
        max_workers=8,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.py",
            "tokenizer*", "vocab*", "merges*", "*.model",
        ],
    )
    print("SNAPSHOT", p)
except Exception as e:
    print("FAILED", type(e).__name__, e)
    sys.exit(1)

tot = 0
for root, _, files in os.walk("/models/Qwen3-32B-AWQ"):
    for f in files:
        try:
            tot += os.path.getsize(os.path.join(root, f))
        except OSError:
            pass
print(f"BYTES {tot}  ({tot/1024**3:.2f} GiB)")
print(f"ELAPSED {time.time()-t0:.0f}s")
PY
rc=$?
echo "[$(ts)] exit=$rc" | tee -a "$LOG"
echo "$rc" > "$EV/download.exit"
echo "$(ts)" > "$EV/download.finished"
