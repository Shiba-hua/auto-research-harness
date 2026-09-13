# 双卡 L40S 从零重建手册（L40S-2-gpus）

在**全新容器**上重建整套 harness 的完整流程与全部踩坑记录。宿主 glibc 2.27、
无 bubblewrap、内核 5.10（无 Landlock）。以下每条都是实测得到的，不是推测。

## 0. 机器事实

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA L40S，各 49140 MiB |
| 宿主 | glibc **2.27**，kernel 5.10，driver 550.54.14 → **CUDA 上限 12.9** |
| CPU / RAM | 144 核 / 502 GiB |
| `/dev/shm` | 120 GB（大，模型都放这里） |
| GitHub 直连 | **不通**（`http=000`）；需走 `ghfast.top` 前缀镜像 |

## 1. 为什么必须 PRoot

宿主的 glibc 2.27 拒收 `manylinux_2_28` 轮子，且 node-pty / sharp 需要 GLIBC≥2.28。
所以全部跑在 PRoot Ubuntu 22.04（glibc 2.35）里。

```bash
# PRoot 静态二进制（直连 GitHub 是 000，必须走镜像）
curl -sL -o proot "https://ghfast.top/https://github.com/proot-me/proot/releases/download/v5.4.1/proot"

# Ubuntu 22.04 base（清华镜像可用）
curl -sL -o ubuntu-base.tar.gz \
  "https://mirrors.tuna.tsinghua.edu.cn/ubuntu-cdimage/ubuntu-base/releases/22.04/release/ubuntu-base-22.04.5-base-amd64.tar.gz"

# Node 24（DSH 需要 >=24）
curl -sL -o node24.tar.xz "https://npmmirror.com/mirrors/node/v24.21.0/node-v24.21.0-linux-x64.tar.xz"
```

## 2. 最容易踩的五个坑（按杀伤力排序）

### 2.1 `/root/.dsh` 必须挂进 guest —— 否则用户配置被静默忽略

DSH 在 guest 内运行，读的是 **guest 的 `DSH_HOME`**。若 launcher 漏了
`-b /root/.dsh:/root/.dsh`，DSH 会用它自己 rootfs 里的空目录，**用户补丁层完全不生效且不报错**。

**判别方法（一秒钟确诊）**：

```bash
dsh --profile web --dump-config         > a.yml
dsh --profile web --dump-default-config > b.yml
diff a.yml b.yml      # 差异为 0 = 用户层没生效；正常应有几十行差异
```

### 2.2 用户补丁层的正确路径是 `$DSH_HOME/cordis.patch.yml`

不是 `profiles/web/` 下。DSH 源码注释原文：

> The home-level user patch layer (`$DSH_HOME/cordis.patch.yml`)

（`profiles/<name>/cordis.yml` 是 profile 的**空根配置**，即 `PROFILE_ROOT_FILENAME`，不是给用户改的。）

### 2.3 CUDA 头文件是跨目录软链

`/usr/local/cuda/include/cuda_runtime.h` 实际指向
`/opt/conda/envs/chatglm3/lib/python3.9/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h`。
只 bind `/usr/local/cuda` 会留下**悬空链接**，编译时报 `cuda_runtime.h: No such file`。
必须把那个 conda 环境也 bind 进去。

### 2.4 三个 CUDA 工具链都有毛病，只有第四个能用

| 来源 | 问题 |
|---|---|
| 宿主 `/usr/local/cuda-11.8` | ptxas 不支持 sm_89 FP8：`Feature 'cvt with .e4m3x2/.e5m2x2' requires .target sm_90` |
| pip `nvidia-cuda-nvcc-cu12` | **各版本都不含 nvcc 驱动**，只有 `ptxas` 与头文件 |
| pip `nvidia/cu13` | nvcc 13.2 可运行，但 `__ldg` 链接失败：`ptxas fatal: Unresolved extern function '__nv_ldg_f_impl'`（该 relocatable 工具链的头文件与 libdevice 同包却对不上） |
| **NVIDIA 官方 deb（可用）** | `cuda-nvcc-12-9_*.deb`（37 MB）+ `cuda-nvvm-12-9`（42 MB），与 pip 的 CUDA12 头/库拼装即可 |

```bash
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
curl -sL -O $BASE/cuda-nvcc-12-9_12.9.86-1_amd64.deb
curl -sL -O $BASE/cuda-nvvm-12-9_12.9.86-1_amd64.deb
dpkg-deb -x cuda-nvcc-12-9_*.deb extracted/     # → usr/local/cuda-12.9/bin/nvcc
# 再把 pip 的 cublas / cusparse / cudart / cccl 头文件与库并进同一个 CUDA_HOME
```

拼装后必须补齐这些 CMake target 依赖，否则 configure 阶段逐个报缺：

- `CUDA::cublas` ← pip `nvidia/cublas/lib` 与 `include/cublas_v2.h`
- `CUDA::cuda_driver` ← `libcuda.so`（指向 `/root/dsh-harness/nvidia-driver/libcuda.so.1`）
- 开发符号链接：pip 只给 `libX.so.N`，需自己 `ln -s` 出 `libX.so`

### 2.5 驱动库要 bind 成一目录，且别漏 `libcuda.so.1`

只绑 `libnvidia-ml` / `libnvidia-cfg` 会让 `torch.cuda.is_available()` 返回 **false**
（症状是 `ImportError: libcuda.so.1: cannot open shared object file`）。
正确做法：把所有驱动库解引用拷到 `/root/dsh-harness/nvidia-driver/`，
bind 到 guest `/opt/nvidia-driver`，并设 `LD_LIBRARY_PATH`。

## 3. vLLM 0.25.0 安装要点

```bash
# 必须用 cu129 索引；PyPI 默认是 cu130 -> ImportError: libcudart.so.13（驱动只有 550）
PIP_INDEX_URL=https://wheels.vllm.ai/0.25.0/cu129/ \
PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  pip install "vllm==0.25.0"
```

- **`torchcodec` 必须移除**：cu130 构建会抛 `OSError`，而 vLLM 的守卫只捕 `ImportError`，于是逃逸。
- guest 内须装 `gcc libc6-dev g++ make ninja-build`（Triton JIT 与 CMake 需要）。
- **`VLLM_USE_FLASHINFER_SAMPLER=0`**：FlashInfer 采样器要 nvcc。该变量须加进 launcher
  的转发白名单 —— launcher 用 `env -i`，外部 export 会被丢掉。
- **MTP 要开 CUDA graph 必须 `--max-num-seqs 16`**：默认 256 会按 `[1..512]` 逐档捕获图，
  多耗约 1.5 GiB 导致 OOM。

## 4. 杀 vLLM 的正确姿势

**vLLM v1 的 `EngineCore` 是独立进程，cmdline 里没有 `vllm serve`。**
只杀 APIServer 会让它变孤儿，继续占 38 GiB 显存。必须多模式覆盖：

```bash
for pat in "[v]llm serve" "[V]LLM::EngineCore" "[r]esource_tracker" "[v]llm025-venv/bin/python"; do
  pgrep -f "$pat" | xargs -r kill -9
done
```

> 所有 `pgrep` / `pkill` 逻辑**一律写进脚本文件**。写在命令行里会让模式匹配到调用者自己的
> cmdline（例如 `echo "killed spec_ab"` 里含 `spec_ab`），把自己杀掉。

## 5. 网络与下载

- HF 走 `HF_ENDPOINT=https://hf-mirror.com`，且**必须 `HF_HUB_DISABLE_XET=1`**
  （镜像站不支持 Xet，否则报 `Reqwest error: builder error`）。
  该变量要在 **guest 脚本内** export；launcher 的 `env -i` 会丢掉外部环境。
- `HF_HOME` 不能指向 overlay（仅 62 GB），要指向 `/dev/shm`。
- pip 用清华镜像；npm 用 `https://registry.npmmirror.com`。
- npm 装 DSH 要放开 postinstall，否则原生模块缺失：
  `npm install -g @deepseek-ai/dsh@0.1.5-rc.1 --allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,@google/genai,protobufjs`

## 6. DSH 沙盒必须用 danger-full-access

本机既无 bubblewrap 也无 Landlock（需内核 5.13+，这里是 5.10），
`dsh-sandbox` **无法兑现 `workspace-write`**，会对每一次 bash 调用硬失败：

```
sandbox mode "workspace-write" is requested but no sandbox backend is usable on this host;
refusing to run the command unconfined.
```

安全改由两层承担：审批层（`approval: ask`）与命令层（`guarded_bash`）。
DSH 的 fail-closed 行为本身是对的，问题在配置。

## 7. 值班员动态装卸 GPU

`deploy/guard-arbiter` 按每张卡空闲显存决定 MiniCPM 跑 CPU 还是卸载到某张卡：

```bash
GUARD_VRAM_THRESH=8000 ./deploy/guard-arbiter
```

状态写 `state/guard-mode` 做抖动抑制。实测双向切换各约 6 秒，切换后问答均正确。
llama.cpp 需 **CUDA 版**才能 `-ngl` 卸载。

## 8. 推测解码框架可用性（实测判定）

| 路线 | vLLM 0.25.0 | llama.cpp | 说明 |
|---|---|---|---|
| 原生 MTP (`qwen3_5_mtp`) | 可用 | `draft-mtp` | 同源、零额外下载，实测稳态 1.51x |
| DFlash2 | 需未合并 PR #52816 | **可用**（`draft-dflash`） | llama.cpp 实测稳态 1.68x |
| DFlash | 需未合并 PR #40898 | 可用 | vLLM 报错：mixed sliding/full attention |
| DSpark (RadixArk) | 不兼容 | `draft-dspark` | 绑定 DeepSeek-V4 的 `hc_*` 超连接 |
| DSpark (RedHatAI) | **可用** | — | 用 vLLM Speculators 训练，但实测无加速 |

> DSpark 要区分来源：RedHatAI 版架构名 `Qwen3DSparkModel` 直接命中 vLLM registry，开箱可用；
> RadixArk 版走的是另一套实现。同名不同源。

---

## 9. `env -i` 白名单 —— 一个会反复咬人的坑（已复发两次）

launcher 用 `/usr/bin/env -i` 构造纯净环境，**只传显式列出的变量**。
任何"外部设了就该生效"的变量都会被静默丢掉，而且症状离原因很远。

**已复发两次：**

| 变量 | 症状 | 为什么会漏 |
|---|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | vLLM 启动时找 nvcc 失败 | 我以为 export 了就传进去了 |
| `LOCAL_LLM_KEY` | **DSH 界面选模型时报 `MISSING_CREDENTIAL`** | 配置里写了 `apiKeyEnv: LOCAL_LLM_KEY`，但环境里没有 |

**规则**：launcher 里每加一个需要向下传递的变量，都要**同时改两处** ——
① `guest` 的 `env -i` 列表（或在调用处显式展开），
② 若有转发循环，也要加进循环名单。

**排查口诀**：guest 里 `echo $VAR` 看到 "(未设置)"，就是白名单漏了，不要怀疑下游。

### 9.1 界面报凭据错误时的完整检查链

```
界面选模型 → pi-ai 解析 apiKeyEnv → 从进程环境找该变量 → 找不到就 MISSING_CREDENTIAL
```

依次确认：

1. `GR_EXTRA_BIND="/root/.dsh:/root/.dsh" bin/guest /bin/bash -c 'echo $LOCAL_LLM_KEY'` —— 有没有值
2. `bin/dsh-run --profile web --dump-config | grep apiKeyEnv` —— 配置引用的变量名是否一致
3. `bin/dsh-run --profile web --dump-config | grep -E "qwen|minicpm"` —— **模型 id 是否与后端实测一致**

> 第 3 条同样咬过一次：配置里写着旧机的 `qwen3-32b`，而新研究员服务提供的是 `qwen3-8-27b`。
> **重建机器后必须重新核对模型 id**，不能照抄旧配置。
