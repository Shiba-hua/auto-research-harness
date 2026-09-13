import json, sys, time, urllib.request

URL = "http://127.0.0.1:18003/v1/chat/completions"
MODEL = "qwen3-8-27b-gguf"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "unknown"
LONG = ("以下是一段用于填充上下文的无关文本，请忽略其内容。" * 240)

CASES = [
    ("short", "请写一段约 300 字的科普短文，介绍什么是推测解码。", 256),
    ("mid",   "用三句话解释 RLVR（可验证奖励强化学习）与普通 RLHF 的区别。", 256),
    ("long",  LONG + "\n\n请忽略上面所有内容，只回答：1+1 等于几？", 128),
]

def gen(p, mt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": mt, "temperature": 0}
    rq = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(rq, timeout=900) as r:
        d = json.load(r)
    return d, time.time() - t0

out = {"label": LABEL, "max_model_len": 262144, "runs": []}
tc = td = 0.0
for name, p, mt in CASES:
    try:
        d, dt = gen(p, mt)
        u = d.get("usage", {})
        ct = u.get("completion_tokens", 0); pt = u.get("prompt_tokens", 0)
        tc += ct; td += dt
        out["runs"].append({"case": name, "prompt_tokens": pt, "completion_tokens": ct,
                            "wall_s": round(dt, 3), "tok_s": round(ct/dt, 2) if dt else 0})
        print("  %-6s prompt=%-6d out=%-4d %.2fs -> %.2f tok/s" % (name, pt, ct, dt, ct/dt if dt else 0))
    except Exception as e:
        out["runs"].append({"case": name, "error": str(e)[:110]})
        print("  %-6s FAIL %s" % (name, str(e)[:110]))
if td:
    out["aggregate_tok_s"] = round(tc/td, 2)
    print("AGGREGATE %s: %.2f tok/s (%d tok / %.2fs)" % (LABEL, out["aggregate_tok_s"], tc, td))
with open("/root/dsh-harness/evidence/bench.jsonl", "a") as f:
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
