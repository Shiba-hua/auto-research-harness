#!/usr/bin/env bash
# Step 2 -- bring up Qwen3-32B-AWQ on vLLM and measure VRAM / throughput / 40k context.
# Shutdown follows the proven rule: SIGTERM the GUEST vllm PID (never the PRoot wrapper).
set -uo pipefail

RUN=/root/dsh-harness/bin/vllm-run
MODEL=/models/Qwen3-32B-AWQ
NAME=researcher
PORT=18001
MAXLEN=40960
GPUUTIL=0.92
EV=/root/dsh-harness/evidence/researcher-bringup
mkdir -p "$EV"
LOG="$EV/vllm-32b.log"; SUM="$EV/summary.jsonl"
: > "$LOG"; : > "$SUM"

gpu_used()  { nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits | tr -d ' '; }
gpu_free()  { nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | tr -d ' '; }
vllm_pids() { pgrep -f "[v]llm serve" || true; }
emit() { printf '%s\n' "$1" | tee -a "$SUM"; }

emit "{\"phase\":\"start\",\"gpu_used_mib\":$(gpu_used),\"model\":\"$MODEL\",\"max_model_len\":$MAXLEN,\"gpu_util\":$GPUUTIL}"

# ---------- launch ----------
T0=$(date +%s)
setsid nohup "$RUN" vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPUUTIL" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG" 2>&1 &
WRAP=$!
emit "{\"phase\":\"launched\",\"wrapper_pid\":$WRAP}"

READY=no
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { READY=yes; break; }
  if ! pgrep -f "[v]llm serve" >/dev/null 2>&1; then break; fi
  sleep 5
done
T1=$(date +%s)
emit "{\"phase\":\"readiness\",\"ready\":\"$READY\",\"load_seconds\":$((T1-T0)),\"gpu_used_mib\":$(gpu_used)}"

if [ "$READY" != yes ]; then
  emit '{"phase":"LOAD_FAILED"}'
  echo "--- server log tail ---"; tail -25 "$LOG"
  P=$(vllm_pids); [ -n "$P" ] && kill -TERM $P 2>/dev/null
  exit 1
fi

# KV cache actually granted (vLLM logs it) -- the number that decides 40k viability
KVLINES=$(grep -iE "GPU KV cache size|Maximum concurrency" "$LOG" | tail -2 | tr '\n' ' ')
emit "$(python3 -c "import json,sys;print(json.dumps({'phase':'kv_cache','log':'$KVLINES'.replace('\"','')}))")"

# ---------- decode throughput (thinking OFF for a clean measurement) ----------
req() {  # $1=max_tokens $2=prompt
  local MT="$1" P="$2" S E R
  S=$(date +%s.%N)
  R=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'$NAME','max_tokens':$MT,'temperature':0.6,'top_p':0.95,'top_k':20,
 'chat_template_kwargs':{'enable_thinking':False},
 'messages':[{'role':'user','content':sys.argv[1]}]}))" "$P")")
  E=$(date +%s.%N)
  python3 - "$R" "$S" "$E" <<'PY'
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: print(json.dumps({"ok":False,"raw":sys.argv[1][:200]})); raise SystemExit
u=d.get("usage",{}); el=float(sys.argv[3])-float(sys.argv[2])
ct=u.get("completion_tokens",0); pt=u.get("prompt_tokens",0)
txt=(d.get("choices",[{}])[0].get("message",{}) or {}).get("content","") or ""
print(json.dumps({"ok":True,"prompt_tokens":pt,"completion_tokens":ct,
                  "elapsed_s":round(el,2),"decode_tps":round(ct/el,2) if el>0 else None,
                  "finish":d.get("choices",[{}])[0].get("finish_reason"),
                  "preview":txt[:60]}, ensure_ascii=False))
PY
}

echo "{\"phase\":\"decode_short\",\"r\":$(req 256 '用中文简要说明什么是 RLVR，200 字以内。')}"  | tee -a "$SUM"
echo "{\"phase\":\"decode_long\",\"r\":$(req 1024 '请写一段关于强化学习与可验证奖励的长文。')}" | tee -a "$SUM"

# ---------- prefill throughput: long prompt, 1 output token ----------
LONG=$(python3 -c "print('请阅读以下重复文本并回答它出现了几次。\n' + ('机器学习与强化学习实验记录。'*2200))")
echo "{\"phase\":\"prefill_long\",\"r\":$(req 1 "$LONG")}" | tee -a "$SUM"

# ---------- 40k context acceptance: send ~38k-token prompt ----------
HUGE=$(python3 -c "print('以下是实验日志，请只回答最后一个词。\n' + ('RLVR experiment log line. '*9000) + '\n最后一个词：确认')")
emit "{\"phase\":\"ctx40k_request\",\"prompt_chars\":${#HUGE}}"
echo "{\"phase\":\"ctx40k\",\"r\":$(req 8 "$HUGE")}" | tee -a "$SUM"

# ---------- tool calling ----------
TOOLREQ=$(python3 -c "
import json
print(json.dumps({'model':'$NAME','max_tokens':256,'temperature':0.6,
 'chat_template_kwargs':{'enable_thinking':False},
 'messages':[{'role':'user','content':'现在有哪些实验在跑？用工具查。'}],
 'tools':[{'type':'function','function':{'name':'lab_run_list','description':'列出服务器上正在运行和最近的实验',
   'parameters':{'type':'object','properties':{},'required':[]}}}]}))")
echo "{\"phase\":\"tool_call\",\"r\":$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$TOOLREQ" | python3 -c "
import json,sys
d=json.load(sys.stdin); m=d['choices'][0]['message']
tc=m.get('tool_calls') or []
print(json.dumps({'has_tool_calls':bool(tc),
 'name':tc[0]['function']['name'] if tc else None,
 'args':tc[0]['function']['arguments'] if tc else None,
 'content_preview':(m.get('content') or '')[:80]},ensure_ascii=False))")}" | tee -a "$SUM"

emit "{\"phase\":\"gpu_at_peak\",\"gpu_used_mib\":$(gpu_used),\"gpu_free_mib\":$(gpu_free)}"

# ---------- shutdown + VRAM release credential ----------
P=$(vllm_pids)
[ -n "$P" ] && kill -TERM $P 2>/dev/null
REL=no
for i in $(seq 1 90); do U=$(gpu_used); [ "$U" -lt 1000 ] && { REL=yes; T=$i; break; }; sleep 1; done
U=$(gpu_used); LEFT=$(vllm_pids | wc -l | tr -d ' ')
emit "{\"phase\":\"shutdown\",\"released\":$([ "$REL" = yes ] && echo true || echo false),\"release_seconds\":${T:-null},\"gpu_used_mib\":$U,\"leftover_pids\":$LEFT}"
if [ "$U" -lt 1000 ] && [ "$LEFT" -eq 0 ]; then
  emit '{"phase":"gpu_credential","issued":true}'
else
  emit '{"phase":"gpu_credential","issued":false}'
fi
