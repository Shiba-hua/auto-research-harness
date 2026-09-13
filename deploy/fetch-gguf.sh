#!/usr/bin/env bash
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
HF=/root/vllm018-conda/bin/hf
D=/dev/shm/models
echo "=== 基座 GGUF: unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_M 15.33GiB) ==="
"$HF" download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf \
  --local-dir $D/base-qwen38-gguf --max-workers 8 2>&1 | tail -2
echo "GGUF DONE $(date -Is)"
du -sh $D/base-qwen38-gguf $D/draft-qwen38-dflash2-gguf 2>/dev/null
