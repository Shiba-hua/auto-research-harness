#!/usr/bin/env bash
# 拉模型到 /dev/shm（可重入）
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
HF=/root/vllm018-conda/bin/hf
D=/dev/shm/models
mkdir -p "$D"
fetch() {
  local repo="$1" dir="$2" n=0
  echo "=== $repo -> $D/$dir ==="
  while [ $n -lt 4 ]; do
    n=$((n+1))
    "$HF" download "$repo" --local-dir "$D/$dir" --max-workers 8 2>&1 | tail -2
    rc=${PIPESTATUS[0]}
    echo "  attempt $n rc=$rc size=$(du -sh "$D/$dir" 2>/dev/null | cut -f1)"
    [ "$rc" = "0" ] && break
    sleep 8
  done
}
fetch openbmb/MiniCPM5-2B-GGUF        mcpm5-2b-gguf
fetch cyankiwi/Qwen3.8-27B-AWQ-INT4   qwen38-27b-awq-int4
fetch z-lab/Qwen3.8-27B-DFlash2       draft-qwen38-dflash2
fetch RadixArk/Qwen3.8-27B-DSpark     draft-qwen38-dspark
echo "PHASE1 DONE $(date -Is)"
du -sh $D/* 2>/dev/null
