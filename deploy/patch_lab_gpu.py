#!/usr/bin/env python3
"""把 lab_server.py 的 gpu_now() 从「单卡假设」改成「多卡正确」。

原实现（单卡能用、双卡必崩）：
    used, free, util = [x.strip() for x in out.strip().split(",")[:3]]
双卡时 nvidia-smi 输出两行，split(",") 后第三段是 '89\\n7398'，int() 直接 ValueError。

新的语义（写清楚，避免以后再猜）：
  - gpus[]           每张卡的明细
  - used_mib/free_mib 是**求和**（向后兼容旧字段名）
  - util_pct          取最大（最忙那张）
  - idle_gpu          第一张 used < GPU_IDLE_MIB 的卡号，没有则 None
  - max_free_mib      最空闲那张卡的空闲量
仲裁据此判断：
  - _credential  要求**所有**卡都空闲（"释放显存是起训练的前置条件"，保守取全空）
  - researcher_up 要求**至少一张**卡的空闲量 >= RESEARCHER_NEED_MIB
"""
import io
import re

P = "/root/dsh-harness/mcp/lab_server.py"
s = io.open(P, encoding="utf-8").read()

OLD = '''def gpu_now():
    rc, out, _ = sh("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu "
                    "--format=csv,noheader,nounits", timeout=20)
    if rc != 0:
        return {"ok": False, "error": "nvidia-smi failed"}
    used, free, util = [x.strip() for x in out.strip().split(",")[:3]]
    return {"ok": True, "used_mib": int(used), "free_mib": int(free), "util_pct": int(util)}
'''

NEW = '''GPU_IDLE_MIB = 1000          # 低于此视为「该卡空闲」
RESEARCHER_NEED_MIB = 20000  # 起研究员至少需要这么多空闲显存


def gpu_now():
    """按卡解析 GPU 状态。多卡安全。

    返回 gpus[] 明细；used_mib/free_mib 是求和（兼容旧字段），
    util_pct 取最忙那张；另给 idle_gpu / max_free_mib 供仲裁使用。
    """
    rc, out, _ = sh("nvidia-smi --query-gpu=index,memory.used,memory.free,"
                    "memory.total,utilization.gpu --format=csv,noheader,nounits", timeout=20)
    if rc != 0:
        return {"ok": False, "error": "nvidia-smi failed", "gpus": []}
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "used_mib": int(parts[1]),
                "free_mib": int(parts[2]),
                "total_mib": int(parts[3]),
                "util_pct": int(parts[4]),
            })
        except ValueError:
            continue
    if not gpus:
        return {"ok": False, "error": "no GPU rows parsed",
                "raw": out.strip()[:200], "gpus": []}
    idle = [g for g in gpus if g["used_mib"] < GPU_IDLE_MIB]
    best = max(gpus, key=lambda g: g["free_mib"])
    return {
        "ok": True,
        "gpus": gpus,
        "count": len(gpus),
        # 聚合（向后兼容旧字段名）
        "used_mib": sum(g["used_mib"] for g in gpus),
        "free_mib": sum(g["free_mib"] for g in gpus),
        "total_mib": sum(g["total_mib"] for g in gpus),
        "util_pct": max(g["util_pct"] for g in gpus),
        # 仲裁用
        "all_idle": len(idle) == len(gpus),
        "idle_gpu": idle[0]["index"] if idle else None,
        "max_free_mib": best["free_mib"],
        "best_gpu": best["index"],
    }
'''

assert OLD in s, "gpu_now 原文未找到（可能已被改过）"
s = s.replace(OLD, NEW)

# _credential：要求所有卡空闲
OLD_CRED = '''    u = gpu_now()
    left = vllm_pids()
    issued = bool(u.get("ok")) and u["used_mib"] < 1000 and len(left) == 0
    return {"issued": issued, "gpu_used_mib": u.get("used_mib"),
            "gpu_free_mib": u.get("free_mib"), "leftover_pids": left,
            "rule": "gpu_used_mib < 1000 AND leftover_pids == 0",'''
NEW_CRED = '''    u = gpu_now()
    left = vllm_pids()
    issued = bool(u.get("ok")) and bool(u.get("all_idle")) and len(left) == 0
    return {"issued": issued, "gpu_used_mib": u.get("used_mib"),
            "gpu_free_mib": u.get("free_mib"), "gpus": u.get("gpus"),
            "leftover_pids": left,
            "rule": "ALL GPUs used_mib < %d AND leftover_pids == 0" % GPU_IDLE_MIB,'''
assert OLD_CRED in s, "_credential 原文未找到"
s = s.replace(OLD_CRED, NEW_CRED)

# researcher_up：至少一张卡够用
OLD_RU = '''    # GPU 被别的进程占着 → 知情拒绝（研究员与训练不能共存）
    if g["used_mib"] >= 1000:
        return {"action": "refused", "ok": False,
                "why": "研究员与训练不能共存：当前显存已用 %d MiB，起研究员会 OOM" % g["used_mib"],
                "gpu": g,'''
NEW_RU = '''    # 没有一张卡够用 → 知情拒绝（研究员与训练不能共存）
    if g.get("max_free_mib", 0) < RESEARCHER_NEED_MIB:
        return {"action": "refused", "ok": False,
                "why": "没有 GPU 有足够空闲显存起研究员：最空闲的是 GPU%s（%d MiB），需要 %d MiB"
                       % (g.get("best_gpu"), g.get("max_free_mib", 0), RESEARCHER_NEED_MIB),
                "gpu": g,'''
assert OLD_RU in s, "researcher_up 原文未找到"
s = s.replace(OLD_RU, NEW_RU)

io.open(P, "w", encoding="utf-8").write(s)
print("patched lab_server.py: gpu_now / _credential / researcher_up")
