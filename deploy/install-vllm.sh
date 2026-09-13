#!/usr/bin/env bash
# 装 cu129 版 torch 2.11 + vLLM 0.25.0。约 10-20 分钟。
set -uo pipefail
D=/root/dsh-harness
G=$D/bin/guest
PY=/opt/vllm025-venv/bin/python
LOG=$D/logs/install-vllm.log
: > "$LOG"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

log "=== A. 升级 pip/wheel/setuptools ==="
$G $PY -m pip install --no-cache-dir -q --upgrade pip wheel setuptools 2>&1 | tail -3 | tee -a "$LOG"

log "=== B. torch 2.11.0+cu129 ==="
$G env PIP_INDEX_URL=https://download.pytorch.org/whl/cu129 $PY -m pip install --no-cache-dir \
  "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" 2>&1 | tail -6 | tee -a "$LOG"
log "  torch rc=$?"

log "=== C. vLLM 0.25.0+cu129（必须用 cu129 索引，PyPI 默认是 cu130）==="
$G env PIP_INDEX_URL=https://wheels.vllm.ai/0.25.0/cu129/ \
      PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  $PY -m pip install --no-cache-dir "vllm==0.25.0" 2>&1 | tail -10 | tee -a "$LOG"
log "  vllm rc=$?"

log "=== D. 移除 cu130 的 torchcodec（会抛 OSError 逃逸 vLLM 的 ImportError 守卫）==="
$G /bin/bash -c 'VE=/opt/vllm025-venv/lib/python3.11/site-packages; if [ -d $VE/torchcodec ]; then mv $VE/torchcodec $VE/.torchcodec.disabled && echo moved; fi' 2>&1 | tail -2 | tee -a "$LOG"

log "=== E. 版本核对 ==="
$G $PY - <<'PY' 2>&1 | tee -a "$LOG"
import importlib.metadata as md
for p in ("torch","torchvision","torchaudio","vllm","transformers"):
    try: print("  %-14s %s" % (p, md.version(p)))
    except Exception as e: print("  %-14s (missing)" % p)
PY

log "=== F. ★ 闸门：CUDA 可用性 ==="
$G $PY - <<'PY' 2>&1 | tee -a "$LOG"
import json, torch
o = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    o["n_gpu"] = torch.cuda.device_count()
    o["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    o["capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.randn(1024,1024, device="cuda"); y = float((x@x).sum())
    o["matmul_ok"] = True
print("GATE " + json.dumps(o, ensure_ascii=False))
PY

log "=== G. vLLM 架构支持 ==="
$G $PY - <<'PY' 2>&1 | tee -a "$LOG"
import vllm
print("  vllm", vllm.__version__)
from vllm.model_executor.models.registry import ModelRegistry
reg = set(ModelRegistry.get_supported_archs())
for a in ["Qwen3_5ForConditionalGeneration","Qwen3_5MTP","DFlashDraftModel","DSparkDraftModel",
          "Qwen3_5MoeForConditionalGeneration","Qwen3_5MoeMTP","DFlash2DraftModel"]:
    print("  %-36s %s" % (a, "YES" if a in reg else "no"))
PY
echo "INSTALL_DONE" >> "$LOG"
