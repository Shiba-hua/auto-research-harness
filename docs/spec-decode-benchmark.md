# 双卡机推测解码实测报告（L40S-2-gpus）

> 全部数据来自目标机实测，非文献推断。原始记录：`/dev/shm` 同级 `evidence/`（`bench.jsonl`、`acceptance.jsonl`）。
> 旧单卡机结论已作废；本报告是**在新两卡机上重做**的结果。

---

## 1. 实验平台

| 项 | 值 |
|---|---|
| 机器 | `L40S-2-gpus`（容器 `d8d02894b3ab`），**2× NVIDIA L40S，各 49140 MiB** |
| 宿主 | glibc 2.27，kernel 5.10，driver 550.54.14（CUDA ≤ 12.9），144 核 / 502 GiB RAM |
| userspace | PRoot Ubuntu 22.04（glibc 2.35），gcc/g++ 11.4 |
| vLLM | **0.25.0+cu129**（`wheels.vllm.ai/0.25.0/cu129/`），torch 2.11.0+cu129 |
| llama.cpp | 自建 **CUDA 版**（`build-cuda`），CUDA **12.9** 工具链 |
| 关键环境开关 | `VLLM_USE_FLASHINFER_SAMPLER=0`；`--max-num-seqs 16`（压 CUDA graph 捕获）；torchcodec 已移除 |

## 2. 主结果表

### 2.1 Qwen3.8-27B（dense, AWQ-INT4）· vLLM · **256k 全上下文**

| 配置 | short | mid（稳态） | long(3428 tok) | 聚合 | 加速比 |
|---|---|---|---|---|---|
| 基线 + CUDA graph | 41.47 | 42.90 | 20.08 | 39.33 | — |
| **原生 MTP + CUDA graph** | 39.57 | **64.59** | 23.53 | **45.81** | **稳态 1.51× / 聚合 1.16×** |

**MTP 接受率（同一工作量前后增量）**

| 指标 | 值 |
|---|---|
| 总体接受率 | **53.49%**（560 / 1047） |
| pos0 / pos1 / pos2 | **100% / 67.2% / 49.0%** |
| 草稿轮数 | 259 |

> 对照公平性：两条腿 `max-model-len` 均为 **262144**、均开 CUDA graph、同一模型同一张卡。**加速不是靠砍上下文换来的。**
> MTP 的 KV 池 317,656 token，略小于基线 364,519（草稿模型占显存）。

### 2.2 Qwen3.8-27B（GGUF UD-Q4_K_M）· llama.cpp · 32k

| 配置 | short | mid（稳态） | long | 聚合 | 加速比 |
|---|---|---|---|---|---|
| 基线 | 42.11 | 41.52 | 15.42 | 37.59 | — |
| **DFlash2（官方 GGUF 草稿）** | 58.96 | **69.60** | 10.69 | **48.12** | **稳态 1.68× / 聚合 1.28×** |

**DFlash2 接受率**（llama.cpp `draft acceptance` 日志）

| 轮次 | 接受率 | 接受/生成 | 平均接受长度 |
|---|---|---|---|
| 1 | 38.69% | 106/274 | 2.15 |
| 2 | 46.37% | 115/248 | 2.39 |
| 3 | 67.32% | 103/153 | 3.02 |
| 4 | 47.15% | 116/246 | 2.41 |
| **均值** | **≈49.9%** | — | **≈2.49** |

加载证据（证明真跑在 DFlash2 路径上，不是退化成普通草稿）：
```
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
  - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true
```
`block_size=8` 与 `mask_token_id=248070` 与 DFlash2 config 完全一致；`n_extract=5` 对应 5 个目标层。

### 2.3 Qwen3.6-35B-A3B（MoE, FP8）· vLLM · 256k

| 配置 | short | mid | long | 聚合 | 加速比 |
|---|---|---|---|---|---|
| 基线 | 101.25 | 102.67 | 86.01 | 98.31 | — |
| **DSpark** | 115.88 | 106.72 | 73.38 | **100.75** | **1.02×（无实质加速）** |

MoE 只激活 3B，本身已很快（~100 tok/s），推测解码边际收益极小。

## 3. 三条草稿路线的可用性判定（关键结论）

| 草稿 | vLLM 0.25.0 | llama.cpp | 结论 |
|---|---|---|---|
| **原生 MTP** (`qwen3_5_mtp`) | ✅ 可用 | `draft-mtp` | **首选**：同源、零额外下载、稳定 1.5× |
| **DFlash2** (z-lab) | ❌ 需未合并 PR #52816 | ✅ **可用** | **llama.cpp 是正确路线**，实测 1.68× |
| **DFlash** (z-lab, MoE) | ❌ 需未合并 PR #40898 | ✅ | vLLM 报错原文：`DFlash does not yet support mixed sliding/full attention via layer_types` |
| **DSpark** (RadixArk, 27B) | ❌ 绑定 DeepSeek-V4 超连接 | — | 缺 `hc_mult`/`hc_eps`/`hc_sinkhorn_iters` |
| **DSpark** (RedHatAI, MoE) | ✅ **可用** | — | 但实测无加速 |

> **DSpark 的关键区分**：RedHatAI 那个用 vLLM 自家 Speculators 库训练，架构 `Qwen3DSparkModel` 直接命中 registry，开箱可用；而 RadixArk 的 27B 版本走的是另一套实现（DeepSeek-V4 的 `hc_*` 超连接）。同名不同源，不能一概而论。

## 4. 架构事实（实测，含对早期错误判断的更正）

Qwen3.8-27B 是**混合线性注意力**架构：

```
layer_types = {'linear_attention': 48, 'full_attention': 16}   共 64 层
full_attention_interval = 4      head_dim = 256
num_key_value_heads = 4          mtp_num_hidden_layers = 1
max_position_embeddings = 262144
```

**只有 16 层有 KV cache** → `2×16×4×256×2B = 64 KiB/token`。

- 早期曾按 64 层算成 256 KiB/token，**错了 4 倍**，已更正。
- 运行时验证：`Available KV cache memory 16.72 GiB / 254,134 tokens` → **69.1 KiB/token**，与推算吻合。

## 5. 环境建设中定位到的真实障碍（供复现）

| 现象 | 根因 | 处理 |
|---|---|---|
| `/root/.dsh` 未挂进 guest | 用户补丁层被完全忽略；判据：`--dump-config` 与 `--dump-default-config` **零差异** | launcher 加 `-b /root/.dsh:/root/.dsh`（修后差异 99 行） |
| 直连 GitHub `http=000` | 网络封锁 | 走 `ghfast.top` 镜像 |
| CUDA 11.8 编 llama.cpp 失败 | ptxas 不支持 sm_89 FP8（`cvt .e4m3x2` 需 sm_90） | 换 CUDA 12.9 |
| pip 的 `nvidia-cuda-nvcc-cu12` 无 nvcc | 该包各版本**只含 ptxas**，不含 nvcc 驱动 | 从 NVIDIA 官方 deb 提取（`cuda-nvcc-12-9`，37 MB） |
| pip 的 CUDA 13 工具链 `__ldg` 编译失败 | `ptxas fatal: Unresolved extern function '__nv_ldg_f_impl'`，该 relocatable 工具链缺陷 | 弃用，改 CUDA 12.9 |
| CMake 找不到 `CUDA::cublas` / `CUDA::cuda_driver` | 合成 CUDA_HOME 缺库 | 从 pip 包补 cuBLAS/cuSPARSE 等 + `libcuda.so` 链接 |

## 6. 附：值班员动态装卸 GPU（已实现并验证）

`bin/guard-arbiter` 按每张卡空闲显存决定 MiniCPM 跑 CPU 还是卸载到某卡（阈值默认 8000 MiB），状态落在 `state/guard-mode` 做抖动抑制。实测双向切换：

```
switch cpu -> gpu1 (best GPU1 free=48642MiB, thresh=8000MiB)   ready in 2 probes
switch gpu1 -> cpu (best GPU1 free=45383MiB, thresh=60000MiB)  ready in 2 probes
switch cpu -> gpu1 (best GPU1 free=48642MiB, thresh=8000MiB)   ready in 2 probes
```
两种模式下问答均正确（`2+3→5`、`7+8→15`），每次切换约 6 秒。
