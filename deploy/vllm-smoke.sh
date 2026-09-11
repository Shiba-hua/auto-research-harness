#!/usr/bin/env bash
# V4/V5/V7/V8 -- researcher inference stack validation using the EXISTING lab model
# (Qwen3-0.6B, already on disk). No downloads. See docs/09-研究员推理栈方案.md §6.
#
# KEY LESSON (found 2026-09-11): killing the PRoot wrapper process does NOT stop vLLM --
# the guest process is orphaned to PID 1 and keeps holding VRAM. vLLM must be signalled
# by its OWN host-side PID, and VRAM release must be PROVEN, not assumed.
set -uo pipefail

RUN=/root/dsh-harness/bin/vllm-run
MODEL=/workspace/models/qwen3-0.6b
PORT=18001
EV=/root/dsh-harness/evidence/vllm-smoke
mkdir -p "$EV"
LOG="$EV/vllm-server.log"
SUM="$EV/summary.jsonl"
: > "$LOG"; : > "$SUM"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
vllm_pids() { pgrep -f "[v]llm serve" || true; }   # [v] avoids self-match

emit() { printf '%s\n' "$1" >> "$SUM"; printf '%s\n' "$1"; }

emit "{\"phase\":\"start\",\"gpu_used_mib\":$(gpu_used),\"gpu_free_mib\":$(gpu_free)}"

# ---------- start server (own process group, so cleanup is possible) ----------
setsid nohup "$RUN" vllm serve "$MODEL" \
  --served-model-name researcher \
  --host 127.0.0.1 --port "$PORT" \
  --dtype float16 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 4096 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG" 2>&1 &
WRAP=$!
emit "{\"phase\":\"server_started\",\"wrapper_pid\":$WRAP}"

READY=no
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { READY=yes; break; }
  sleep 2
done
emit "{\"phase\":\"readiness\",\"ready\":\"$READY\",\"wait_s\":$((i*2)),\"gpu_used_mib\":$(gpu_used),\"guest_pids\":\"$(vllm_pids|tr '\n' ' ')\"}"

if [[ "$READY" == "yes" ]]; then
  # ---------- V4: chat completion (UTF-8 safe) ----------
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"researcher","temperature":0,"max_tokens":64,
    "messages":[{"role":"user","content":"用一句中文回答：什么是 RLVR？"}]}' > "$EV/v4-chat.json"

  # ---------- V8: determinism, same prompt x5, temperature=0 ----------
  : > "$EV/v8-raw.jsonl"
  for i in 1 2 3 4 5; do
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
      "model":"researcher","temperature":0,"max_tokens":24,
      "messages":[{"role":"user","content":"只回答一个词：中国的首都"}]}' >> "$EV/v8-raw.jsonl"
    printf '\n' >> "$EV/v8-raw.jsonl"
  done

  PYTHONIOENCODING=utf-8 python3 - "$EV/v4-chat.json" "$EV/v8-raw.jsonl" <<'PY' >> "$SUM"
import json,sys,hashlib
def load(p):
    try: return json.loads(open(p,encoding='utf-8',errors='replace').read())
    except Exception as e: return {"_error":str(e)}
v4=load(sys.argv[1])
try:
    msg=v4["choices"][0]["message"]
    txt=msg.get("content") or ""
    print(json.dumps({"phase":"v4_chat","ok":bool(txt),"chars":len(txt),
                      "preview":txt[:80]}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"phase":"v4_chat","ok":False,"error":str(e),
                      "raw":json.dumps(v4,ensure_ascii=False)[:200]}, ensure_ascii=False))

hs=[]; outs=[]
for line in open(sys.argv[2],encoding='utf-8',errors='replace'):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    try:
        t=d["choices"][0]["message"]["content"] or ""
    except Exception:
        t=""
    outs.append(t.strip()); hs.append(hashlib.sha256(t.strip().encode()).hexdigest()[:12])
print(json.dumps({"phase":"v8_determinism","runs":len(hs),"unique":len(set(hs)),
                  "identical": len(set(hs))==1, "hashes":hs,
                  "samples":[o[:40] for o in outs]}, ensure_ascii=False))
PY

  emit "{\"phase\":\"v5_gpu_loaded\",\"gpu_used_mib\":$(gpu_used),\"gpu_free_mib\":$(gpu_free)}"
fi

# ---------- V7: correct shutdown = signal the GUEST vllm process, not the wrapper ----------
emit "{\"phase\":\"v7_shutdown_begin\",\"wrapper_pid\":$WRAP,\"gpu_used_mib\":$(gpu_used)}"
WRAP_STILL=$(kill -0 "$WRAP" 2>/dev/null && echo yes || echo no)
emit "{\"phase\":\"v7_wrapper_alive_before\",\"alive\":\"$WRAP_STILL\"}"

# naive approach first: kill the wrapper only, observe
kill -TERM "$WRAP" 2>/dev/null
sleep 5
emit "{\"phase\":\"v7_after_killing_wrapper_only\",\"gpu_used_mib\":$(gpu_used),\"orphan_pids\":\"$(vllm_pids|tr '\n' ' ')\"}"

# correct approach: signal vLLM itself
P=$(vllm_pids)
if [[ -n "$P" ]]; then kill -TERM $P 2>/dev/null; fi
REL=no
for i in $(seq 1 45); do
  U=$(gpu_used)
  if [[ "$U" -lt 1000 ]]; then REL=yes; T=$i; break; fi
  sleep 1
done
U=$(gpu_used)
LEFT=$(vllm_pids | wc -l | tr -d ' ')
emit "{\"phase\":\"v7_after_signalling_vllm\",\"released\":$([[ $REL == yes ]] && echo true || echo false),\"release_seconds\":${T:-null},\"gpu_used_mib\":$U,\"gpu_free_mib\":$(gpu_free),\"leftover_pids\":$LEFT}"

# credential: release is only PROVEN if used < threshold AND no leftover process
if [[ "$U" -lt 1000 && "$LEFT" -eq 0 ]]; then
  emit '{"phase":"v7_credential","issued":true,"rule":"gpu_used_mib<1000 AND leftover_pids==0"}'
else
  emit '{"phase":"v7_credential","issued":false,"rule":"gpu_used_mib<1000 AND leftover_pids==0"}'
fi
