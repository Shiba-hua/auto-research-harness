#!/usr/bin/env bash
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
HF=/root/vllm018-conda/bin/hf
"$HF" download openbmb/MiniCPM5-2B --local-dir /dev/shm/models/mcpm5-2b-st --max-workers 8 2>&1 | tail -3
echo "MCPM_ST_DONE $(date -Is)"
du -sh /dev/shm/models/mcpm5-2b-st
