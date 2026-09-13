#!/usr/bin/env bash
# 打印每张卡的空闲显存（MiB）
nvidia-smi --query-gpu=index,memory.free,memory.used,memory.total --format=csv,noheader,nounits
