"""采集推测解码接受率：跑固定工作负载，取计数器增量。"""
import json, re, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "18001"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-8-27b"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "unknown"
BASE = "http://127.0.0.1:%s" % PORT

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        t = r.read().decode()
    out = {}
    for m in re.finditer(r'^vllm:spec_decode_num_(accepted_tokens|draft_tokens)(_per_pos)?_total\{([^}]*)\}\s+([0-9.e+]+)$',
                         t, re.M):
        kind, posflag, labels, val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        pos = re.search(r'position="(\d+)"', labels)
        key = kind + ("_pos" + pos.group(1) if pos else "")
        out[key] = val
    return out

def gen(p, mt=256):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": mt, "temperature": 0}
    rq = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(rq, timeout=900) as r:
        d = json.load(r)
    return d, time.time() - t0

PROMPTS = [
    "请写一段约 300 字的科普短文，介绍什么是推测解码。",
    "用三句话解释 RLVR 与普通 RLHF 的区别。",
    "写一个 Python 函数判断回文数，并给出简短注释。",
    "列举三条提高 vLLM 推理吞吐的工程手段，各一句说明。",
]

before = metrics()
tot_ct = tot_dt = 0.0
for p in PROMPTS:
    d, dt = gen(p)
    tot_ct += d.get("usage", {}).get("completion_tokens", 0)
    tot_dt += dt

after = metrics()
res = {"label": LABEL, "port": PORT, "model": MODEL,
       "total_completion_tokens": tot_ct, "total_wall_s": round(tot_dt, 3),
       "aggregate_tok_s": round(tot_ct / tot_dt, 2) if tot_dt else 0}

draft = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
acc = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
res["draft_tokens"] = draft
res["accepted_tokens"] = acc
res["acceptance_rate"] = round(acc / draft, 4) if draft else None
# 每位置接受率：position i 的接受数 / 总 draft 轮数
rounds = None
per = {}
for k in sorted(after):
    if k.startswith("accepted_tokens_pos"):
        i = k.rsplit("pos", 1)[1]
        per["pos" + i] = after[k] - before.get(k, 0)
res["per_position_accepted"] = per
if per:
    # position0 接受数 ≈ draft 轮数
    r0 = per.get("pos0", 0)
    res["draft_rounds"] = r0
    res["per_position_rate"] = {k: round(v / r0, 4) for k, v in per.items()} if r0 else {}

print(json.dumps(res, ensure_ascii=False, indent=2))
with open("/root/dsh-harness/evidence/acceptance.jsonl", "a") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
