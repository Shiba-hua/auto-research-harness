#!/usr/bin/env bash
# llama.cpp CUDA 版 —— 用从 NVIDIA 官方 deb 提取的 CUDA 12.9 工具链（pip 版 nvcc 有缺陷）。
set -uo pipefail
D=/root/dsh-harness
LOG=$D/logs/build-llama-cuda.log
: > "$LOG"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
CM=$D/build/cmake/bin/cmake; SRC=$D/build/llama.cpp
FLAGS="-I/opt/cuda12/include -L/opt/cuda12/lib64"
rm -rf $SRC/build-cuda
log "=== configure ==="
$D/bin/guest $CM -S $SRC -B $SRC/build-cuda -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda12/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS="$FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cuda12/lib64 -Wl,-rpath,/opt/cuda12/lib64:/opt/nvidia-driver" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/cuda12/lib64 -Wl,-rpath,/opt/cuda12/lib64:/opt/nvidia-driver" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON > /tmp/cfg.log 2>&1
log "  configure rc=$?"; grep -E "CMake Error" /tmp/cfg.log | head -4 | tee -a "$LOG"
log "=== build ==="
$D/bin/guest $CM --build $SRC/build-cuda --target llama-server llama-cli -j 96 > /tmp/build-full.log 2>&1
log "  build rc=$?"
grep -E "error:|FAILED|fatal" /tmp/build-full.log | head -6 | tee -a "$LOG"
log "=== 产物 ==="
ls -lh $SRC/build-cuda/bin/llama-server 2>/dev/null | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $D/bin/guest $SRC/build-cuda/bin/llama-server --list-devices 2>&1 | head -8 | tee -a "$LOG"
echo "CUDA_BUILD_DONE" >> "$LOG"
