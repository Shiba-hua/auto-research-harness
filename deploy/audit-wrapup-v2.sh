#!/usr/bin/env bash
# 修正版审计队列收尾（v2）
# 背景：v1 队列把 GRPO adapter 挂在了错误的底座上（应为 SFT merged_model），
#       其 24 个结果已标记 INVALID。本脚本等 v2 跑完后收集结果并提交更正。
set -u
LAB=/root/siton-tmp/rlvr-l40s-lab
H=/root/dsh-harness
Q=$LAB/evidence/auto-research-sft-grpo-eval-queue-002
OUT=$H/evidence/audit-wrapup-v2
mkdir -p "$OUT"
LOG=$OUT/wrapup.log
: > "$LOG"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

log "waiting for v2 queue..."
for i in $(seq 1 480); do
  s=$(PYTHONIOENCODING=utf-8 /root/vllm018-conda/bin/python3.11 -c "
import json
try: print(json.load(open('$Q/queue_status.json',encoding='utf-8'))['status'])
except Exception: print('running')" 2>/dev/null)
  [ "$s" != "running" ] && { log "queue finished: $s"; break; }
  sleep 15
done

cd "$LAB"
PYTHONIOENCODING=utf-8 /root/vllm018-conda/bin/python3.11 - > "$OUT/results-v2.json" 2>>"$LOG" <<'PY'
import json, os
LAB = "/root/siton-tmp/rlvr-l40s-lab"
R = os.path.join(LAB, "results")
def acc(name):
    p = os.path.join(R, name, "evaluation_summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("accuracy")
STAGES = [("base","original"),("pure_grpo","pure"),("cot_sft","cot-sft"),
          ("cot_sft_grpo","cot-sft-grpo"),("cod_sft","cod-sft"),("cod_sft_grpo","cod-sft-grpo")]
out = {"correction": "v2 使用正确的 SFT merged_model 作为 GRPO 底座；v1 的 -001 结果无效", "projects": {}}
for proj, pre, main in (("arc","arc-student-",1024),("gsm8k","gsm8k-student-",1536)):
    per = {}
    for label, st in STAGES:
        suff = "-002" if st.endswith("-grpo") else "-001"
        row = {}
        for b in (512,1024,1536,2048,3072,4096):
            a = acc("%s%s-audit-%d%s" % (pre, st, b, suff))
            if a is not None:
                row[str(b)] = a
        per[label] = {"primary": row.get(str(main)), "curve": row}
    out["projects"][proj] = {"primary_budget": main, "stages": per}
print(json.dumps(out, ensure_ascii=False, indent=1))
PY
log "results collected -> $OUT/results-v2.json"

cat > "$OUT/cm.txt" <<'MSG'
修正：SFT→GRPO 审计队列 v2（v1 的 artifact 身份有误）

问题：GRPO adapter 是在 SFT 合并权重上训练的
  (configs/students/*-{cot,cod}-grpo.json 的 model_path -> *_sft-001/merged_model)
但 v1 队列照抄 pure_grpo 模板，把 model 配成了原始 base，
导致 adapter 挂在错误底座上。v1 的 24 个 -001 结果已逐一标记 INVALID.txt，
不删除以便追溯；其数字（含已推送的报告 3111e6d 中的对照表）一律不可引用。

修正：auto-research-sft-grpo-eval-queue-002.json 使用正确的 merged_model，
逐项经 --check-plan 校验（valid_plan_not_started）。

教训：--check-plan 只校验"声明哈希与实际文件一致"，
不校验"model+adapter 的组合在语义上是否正确"。后者必须逐项对照训练配置，
不能靠模板复制。数据身份核对不可省略。
MSG
git add -A configs/students/auto-research-sft-grpo-eval-queue-002.json results/ evidence/
git -c user.name=auto-research-harness -c user.email=auto-research-harness@L40S-1-gpu commit -q -F "$OUT/cm.txt"
git push origin HEAD 2>&1 | tee -a "$LOG" | tail -2
log "done. HEAD=$(git rev-parse --short HEAD)"
