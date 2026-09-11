#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 strategy_curves 构造 manifest（八阶段模式，速记法两阶段暂缺）。

诚实原则：声明全部 8 个阶段，缺的那个留空——让工具如实报
eight_stage_complete=false，而不是伪造完整。
"""
import json
import glob
import os
import sys
from pathlib import Path

LAB = Path("/root/siton-tmp/rlvr-l40s-lab")
OUT = Path("/root/dsh-harness/evidence/strategy-curves")
BUDGETS = [512, 1024, 1536, 2048, 3072, 4096]

# 八阶段的目录名后缀 -> strategy_id
STAGES = [
    ("original",         "base"),
    ("pure",             "pure_grpo"),
    ("cot-sft",          "cot_sft"),
    ("cot-sft-grpo",     "cot_sft_grpo"),
    ("cod-sft",          "cod_sft"),
    ("cod-sft-grpo",     "cod_sft_grpo"),
    ("shorthand-cod-sft",      "shorthand_cod_sft"),
    ("shorthand-cod-sft-grpo", "shorthand_cod_sft_grpo"),
]
LABELS = {
    "base": "original base", "pure_grpo": "pure GRPO",
    "cot_sft": "CoT SFT", "cot_sft_grpo": "CoT SFT + GRPO",
    "cod_sft": "CoD SFT", "cod_sft_grpo": "CoD SFT + GRPO",
    "shorthand_cod_sft": "shorthand CoD SFT",
    "shorthand_cod_sft_grpo": "shorthand CoD SFT + GRPO",
}
# 每阶段的真实 checkpoint（adapter 优先，无则原始 base）
CKPT = {
    # (base_model, adapter) —— GRPO 阶段的底座必须是 SFT merged_model
    "base": ("models/qwen25-0.5b-instruct", None),
    "pure_grpo": ("models/qwen25-0.5b-instruct", "results/{p}-student-pure-grpo-001/best_adapter"),
    "cot_sft": ("results/{p}-student-cot-sft-001/merged_model", None),
    "cot_sft_grpo": ("results/{p}-student-cot-sft-001/merged_model",
                     "results/{p}-student-cot-grpo-{n}/best_adapter"),
    "cod_sft": ("results/{p}-student-cod-sft-001/merged_model", None),
    "cod_sft_grpo": ("results/{p}-student-cod-sft-001/merged_model",
                     "results/{p}-student-cod-grpo-001/best_adapter"),
}

def resolve(project, sid):
    """解析该阶段的 checkpoint 绝对路径；不存在则返回 None（不编造）。"""
    n = "002" if (project == "gsm8k" and sid == "cot_sft_grpo") else "001"
    spec = CKPT.get(sid)
    if not spec:
        return None
    base, adapter = spec
    base_p = LAB / base.format(p=project)
    if not base_p.exists():
        return None
    if adapter:
        ap = LAB / adapter.format(p=project, n=n)
        if not ap.exists():
            return None
        return str(base_p), str(ap)
    return str(base_p), None


def collect(project, suffix):
    """收集该阶段六预算的真实 predictions 路径。"""
    evals = []
    for b in BUDGETS:
        tail = "002" if suffix.endswith("-grpo") else "001"
        d = LAB / "results" / ("%s-student-%s-audit-%d-%s" % (project, suffix, b, tail))
        pred = d / "predictions.jsonl"
        summ = d / "evaluation_summary.json"
        item = {"budget": b, "status": "complete" if pred.exists() else "missing"}
        if pred.exists():
            item["predictions"] = str(pred)
        if summ.exists():
            try:
                acc = json.loads(summ.read_text(encoding="utf-8")).get("accuracy")
                if acc is not None:
                    item["val_acc"] = acc
            except Exception:
                pass
        evals.append(item)
    return evals


def build(project, primary):
    strategies = []
    for suffix, sid in STAGES:
        ck = resolve(project, sid)
        evals = collect(project, suffix) if ck else [
            {"budget": b, "status": "missing"} for b in BUDGETS]
        s = {"strategy_id": sid, "label": LABELS[sid], "family": "student",
             "evaluations": evals}
        if sid in ("cot_sft", "cod_sft"):
            arm = "cot" if sid == "cot_sft" else "cod"
            s["sft_training_manifest"] = str(
                LAB / "results" / ("%s-student-%s-sft-001" % (project, arm)) / "manifest.json")
        if ck:
            # 有 adapter 时 checkpoint 必须是 adapter 路径（见 strategy_curves.py:146）
            s["checkpoint"] = ck[1] if ck[1] else ck[0]
            if ck[1]:
                s["model"] = ck[0]
        else:
            s["missing_reason"] = (
                "速记法规则式派生仅产出 29 条适用记录，不足以支撑 SFT 语料"
                if "shorthand" in sid else "checkpoint 不存在")
        strategies.append(s)
    return {"schema": "rlvr.strategy-curves-manifest/v1",
            "comparison_mode": "eight_stage",
            "project": project, "primary_budget": primary,
            "strategies": strategies}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for project, primary in (("arc", 1024), ("gsm8k", 1536)):
        m = build(project, primary)
        p = OUT / ("manifest-%s.json" % project)
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        have = sum(1 for s in m["strategies"]
                   if any(e.get("status") == "complete" for e in s["evaluations"]))
        out[project] = {"manifest": str(p), "stages_with_data": have,
                        "primary_budget": primary}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
