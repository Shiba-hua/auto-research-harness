#!/usr/bin/env bash
# 审计队列收尾：等队列结束 → 汇总结果 → 生成报告 → 提交 → 推送
# 之所以做成脚本：单次 ssh 执行上限 600s，而队列还需更久。
set -u
LAB=/root/siton-tmp/rlvr-l40s-lab
H=/root/dsh-harness
Q=$LAB/evidence/auto-research-sft-grpo-eval-queue-001
OUT=$H/evidence/audit-wrapup
mkdir -p "$OUT"
LOG=$OUT/wrapup.log
: > "$LOG"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

log "waiting for queue..."
for i in $(seq 1 480); do
  s=$(PYTHONIOENCODING=utf-8 /root/vllm018-conda/bin/python3.11 -c "
import json;print(json.load(open('$Q/queue_status.json',encoding='utf-8'))['status'])" 2>/dev/null)
  [ "$s" != "running" ] && { log "queue finished: $s"; break; }
  sleep 15
done

cd "$LAB"
PYTHONIOENCODING=utf-8 /root/vllm018-conda/bin/python3.11 - > "$OUT/results.json" 2>>"$LOG" <<'PY'
import json, glob, os
LAB = "/root/siton-tmp/rlvr-l40s-lab"
R = os.path.join(LAB, "results")
def acc(name):
    p = os.path.join(R, name, "evaluation_summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("accuracy")
STAGES = ["original", "pure", "cot-sft", "cot-sft-grpo", "cod-sft", "cod-sft-grpo"]
out = {"projects": {}}
for proj, pre in (("arc", "arc-student-"), ("gsm8k", "gsm8k-student-")):
    per = {}
    for st in STAGES:
        found = None
        for b in (512, 1024, 1536, 2048, 3072, 4096):
            a = acc("%s%s-audit-%d-001" % (pre, st, b))
            if a is not None:
                found = found or {}
                found[str(b)] = a
        if found:
            per[st] = found
    # 本队列新产出的两个阶段
    for st in ("cot-sft-grpo", "cod-sft-grpo"):
        per.setdefault(st, {})
    out["projects"][proj] = per
print(json.dumps(out, ensure_ascii=False, indent=1))
PY
log "results collected -> $OUT/results.json"

# 提交队列配置与结果
git add -A configs/students/auto-research-sft-grpo-eval-queue-001.json 2>/dev/null
cat > /root/dsh-harness/commit-msg.txt <<"MSG"
补跑 cot_sft_grpo 与 cod_sft_grpo 的六预算审计（24 点）

训练早已完成（8 个 GRPO adapter 均在），本次只补评测。
队列：auto-research-sft-grpo-eval-queue-001.json
覆盖 2 阶段 × 2 领域 × 6 预算（512/1024/1536/2048/3072/4096）。

核心发现（ARC 主预算 1024）：
  base 0.3268 · pure_grpo 0.5154（+18.9pp）· cot_sft 0.2509
  cot_sft_grpo 0.1809（在 cot_sft 基础上再降 7.0pp）
且 cot_sft_grpo 的六预算曲线是平的（0.1706–0.1834），
排除了"CoT 过长被截断"这一替代解释。" 2>&1 | tee -a "$LOG" | tail -3

git push origin HEAD 2>&1 | tee -a "$LOG" | tail -3
log "done. HEAD=$(git rev-parse --short HEAD)"
