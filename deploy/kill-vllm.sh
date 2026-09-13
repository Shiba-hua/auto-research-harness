#!/usr/bin/env bash
# 彻底停 vLLM。EngineCore 是独立进程且失败后会残留占显存，必须多模式覆盖。
set -uo pipefail
for pat in "[v]llm serve" "[V]LLM::EngineCore" "[V]LLM::EngineCor" \
           "[r]esource_tracker" "[v]llm025-venv/bin/python" "[v]llm025-venv/bin/vllm"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null && echo "  killed $pid ($pat)"
  done
done
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  GPU /'
