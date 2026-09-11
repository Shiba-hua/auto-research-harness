# -*- coding: utf-8 -*-
"""6/8 阶段诊断曲线（明确标注为非协议图）

为什么不用 rlvr_lab.strategy_curves：其 eight_stage 与 legacy_four_arm 两种模式
都要求速记法阶段（shorthand_cod_sft / shorthand_cod_sft_grpo）的真实 checkpoint，
而该数据在 64 源上实测不可实施（见 docs/shorthand_derivation_plan.md 与
evidence/auto-strategy-curves/BLOCKER.md）。故本图是诊断性质，不是协议图。
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/root/siton-tmp/rlvr-l40s-lab/results"
OUT = "/root/dsh-harness/evidence/strategy-curves/diagnostic"
os.makedirs(OUT, exist_ok=True)
BUDGETS = [512, 1024, 1536, 2048, 3072, 4096]
STAGES = [("base","original","-001"), ("pure_grpo","pure","-001"),
          ("cot_sft","cot-sft","-001"), ("cot_sft_grpo","cot-sft-grpo","-002"),
          ("cod_sft","cod-sft","-001"), ("cod_sft_grpo","cod-sft-grpo","-002")]
STYLE = {"base":("gray","o","-"), "pure_grpo":("purple","o","-"),
         "cot_sft":("tab:blue","^","--"), "cot_sft_grpo":("tab:blue","o","-"),
         "cod_sft":("tab:green","^","--"), "cod_sft_grpo":("tab:green","o","-")}

def acc(proj, suffix, b, tail):
    p = os.path.join(R, "%s-student-%s-audit-%d-%s" % (proj, suffix, b, tail.strip("-")), "evaluation_summary.json")
    if not os.path.exists(p): return None
    return json.load(open(p, encoding="utf-8")).get("accuracy")

for proj, main in (("arc",1024), ("gsm8k",1536)):
    fig, ax = plt.subplots(figsize=(10,6), dpi=160)
    for sid, suffix, tail in STAGES:
        tail = tail.strip("-")
        xs, ys = [], []
        for b in BUDGETS:
            a = acc(proj, suffix, b, tail)
            if a is not None:
                xs.append(b); ys.append(a)
        if not xs: continue
        c, m, ls = STYLE[sid]
        ax.plot(xs, ys, color=c, marker=m, linestyle=ls, label=sid)
    ax.axvline(main, color="k", alpha=.2, linestyle=":")
    ax.set_xlabel("max_new_tokens budget"); ax.set_ylabel("accuracy")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("%s: 6/8 stages (DIAGNOSTIC ONLY - not the protocol eight-stage figure)\n"
                 "shorthand_cod_sft / shorthand_cod_sft_grpo absent: source corpus yields too few teachable mechanisms"
                 % proj.upper(), fontsize=9)
    f = os.path.join(OUT, "diagnostic-%s.png" % proj)
    fig.tight_layout(); fig.savefig(f)
    print("wrote", f)
