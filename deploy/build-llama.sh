#!/usr/bin/env bash
# 在 guest（gcc/g++ 11.4）内编 llama.cpp CUDA 版。
# 注意：/usr/local/cuda/include/cuda_runtime.h 是指向 /opt/conda/envs/chatglm3/... 的
# 符号链接，所以那个 conda 环境也必须 bind，否则头文件悬空。
set -uo pipefail
D=/root/dsh-harness
LOG=$D/logs/build-llama.log
: > "$LOG"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

export GR_EXTRA_BIND="/usr/local/cuda:/usr/local/cuda /opt/conda/envs/chatglm3:/opt/conda/envs/chatglm3"
CM=$D/build/cmake/bin/cmake
SRC=$D/build/llama.cpp
rm -rf $SRC/build-cuda

log "=== nvcc 自检（带 bind）==="
$D/bin/guest /bin/bash -c '
cat > /tmp/t.cu <<EOF
__global__ void k(){}
int main(){k<<<1,1>>>();return 0;}
EOF
/usr/local/cuda/bin/nvcc -arch=sm_89 /tmp/t.cu -o /tmp/t.out 2>&1 | head -5
if [ -x /tmp/t.out ]; then echo "  nvcc 可用 (sm_89)"; else echo "  nvcc 仍不可用"; fi
gcc --version|head -1; g++ --version|head -1; ninja --version' 2>&1 | tee -a "$LOG"

log "=== CMake 配置（Ninja + CUDA sm_89）==="
$D/bin/guest $CM -S $SRC -B $SRC/build-cuda -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON 2>&1 | tail -18 | tee -a "$LOG"

log "=== 编译 llama-server + llama-cli（-j 48）==="
$D/bin/guest $CM --build $SRC/build-cuda --target llama-server llama-cli -j 48 2>&1 | tail -25 | tee -a "$LOG"
log "  build rc=$?"

log "=== 产物 ==="
ls -lh $SRC/build-cuda/bin/ 2>/dev/null | tee -a "$LOG"
$D/bin/guest $SRC/build-cuda/bin/llama-server --version 2>&1 | head -4 | tee -a "$LOG"
echo "LLAMA_BUILD_DONE" >> "$LOG"
