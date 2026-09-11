#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 cot_sft_grpo + cod_sft_grpo 的六预算审计队列（修正版）。

修正点：GRPO adapter 是在 **SFT 合并权重**上训练的，因此
  model 必须是对应阶段的 merged_model，而不是原始 base。
依据：configs/students/{arc,gsm8k}-{cot,cod}-grpo.json 的 model_path 字段
      以及 results/<run>/config_from_source.json 的实际记录。
"""
import json
import hashlib
import copy
from pathlib import Path
from rlvr_lab.evaluate import local_artifact_identity

LAB = Path("/root/siton-tmp/rlvr-l40s-lab")
BUDGETS = [512, 1024, 1536, 2048, 3072, 4096]

# (project, stage) -> (adapter run 目录名, 该 adapter 训练时用的底座 merged_model)
PLAN = {
    ("arc", "cot_sft_grpo"): ("arc-student-cot-grpo-001", "arc-student-cot-sft-001"),
    ("arc", "cod_sft_grpo"): ("arc-student-cod-grpo-001", "arc-student-cod-sft-001"),
    ("gsm8k", "cot_sft_grpo"): ("gsm8k-student-cot-grpo-002", "gsm8k-student-cot-sft-001"),
    ("gsm8k", "cod_sft_grpo"): ("gsm8k-student-cod-grpo-001", "gsm8k-student-cod-sft-001"),
}
DATA = {"arc": LAB / "data/arc-disjoint/audit.jsonl", "gsm8k": LAB / "data/gsm8k/audit.jsonl"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tpl = json.loads((LAB / "configs/students/arc-pure-eval-queue-001.json").read_text(encoding="utf-8"))
    out = copy.deepcopy(tpl)
    out["output"] = str(LAB / "evidence/auto-research-sft-grpo-eval-queue-002")
    jobs = []
    for (proj, stage), (adapter_run, merged_run) in PLAN.items():
        model = LAB / "results" / merged_run / "merged_model"
        ad = LAB / "results" / adapter_run / "best_adapter"
        if not model.is_dir():
            raise SystemExit("merged_model 不存在: %s" % model)
        if not ad.is_dir():
            raise SystemExit("best_adapter 不存在: %s" % ad)
        ident = local_artifact_identity(model, ad)
        m_sha = ident["base"]["files_sha256"]
        a_sha = ident["adapter"]["files_sha256"]
        d_sha = sha(DATA[proj])
        for b in BUDGETS:
            jid = "%s-student-%s-audit-%d-002" % (proj, stage.replace("_", "-"), b)
            jobs.append({
                "id": jid, "project": proj, "stage": stage, "budget": b,
                "model": str(model), "adapter": str(ad),
                "model_files_sha256": m_sha, "adapter_files_sha256": a_sha,
                "data": str(DATA[proj]), "data_sha256": d_sha,
                "output": str(LAB / "results" / jid),
                "job_output": str(LAB / "evidence" / jid),
                "timeout_seconds": 7200,
            })
    out["jobs"] = jobs
    dst = LAB / "configs/students/auto-research-sft-grpo-eval-queue-002.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已生成", dst)
    print("job 数:", len(jobs))
    for j in jobs[:4]:
        print("  %s\n    model  = %s\n    adapter= %s" % (j["id"], j["model"], j["adapter"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
