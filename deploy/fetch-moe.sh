#!/usr/bin/env bash
# 第二批：Qwen3.6-35B-A3B（MoE）基座与草稿
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
# FP8：官方且人气最高（按"配对优先，无配对取人气最高"原则）
fetch Qwen/Qwen3.6-35B-A3B-FP8          moe-35b-a3b-fp8
# AWQ-4bit：落在用户要求的 4-6bit 区间
fetch cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit moe-35b-a3b-awq4bit
# 两条草稿
fetch z-lab/Qwen3.6-35B-A3B-DFlash       draft-moe-dflash
fetch RedHatAI/Qwen3.6-35B-A3B-speculator.dspark draft-moe-dspark
echo "PHASE2 DONE $(date -Is)"
du -sh $D/* 2>/dev/null
