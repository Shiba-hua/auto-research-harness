#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 工具选择评测（fixture eval，pass^k）

问：值班员 MiniCPM5-2B 能否为契约表里的每种说法选中正确的工具？
用真实的 15 个 MCP 工具 schema，走 llama-server 的 OpenAI tools 接口。
每个用例跑 K 次（k=3），全部命中才算 pass（pass^k 而非 pass@k）。
"""
import json
import subprocess
import sys
import time
import urllib.request

PYTHON = "/root/vllm018-conda/bin/python3.11"
SRV = "/root/dsh-harness/mcp/lab_server.py"
GUARD = "http://127.0.0.1:18099/v1/chat/completions"
K = 3


def mcp_tools():
    inp = "".join(json.dumps(m) + "\n" for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    p = subprocess.run([PYTHON, SRV], input=inp.encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, timeout=60)
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("id") == 2:
            return d["result"]["tools"]
    return []


def to_openai(tools):
    return [{"type": "function", "function": {
        "name": t["name"],
        "description": t.get("description", ""),
        "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}}
        for t in tools]


def ask(utterance, tools):
    body = {"model": "minicpm5-2b", "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(GUARD, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    tc = msg.get("tool_calls") or []
    if not tc:
        return None, None, (msg.get("content") or "")[:60]
    fn = tc[0].get("function") or {}
    return fn.get("name"), fn.get("arguments"), None


# 契约表 → 期望工具（accept 是允许的集合，允许多个合理选择）
CASES = [
    ("现在什么在跑？", {"lab_run_list"}),
    ("显卡闲着吗", {"lab_gpu"}),
    ("队列里有什么", {"req_list"}),
    ("帮我记一下：验证速记法规则式教程", {"req_add"}),
    ("gsm8k-grpo-formal-001 跑到哪了", {"lab_run_status", "lab_run_list"}),
    ("画一下 gsm8k-grpo-formal-001 的图", {"plot_metrics"}),
    ("把 kl 和 loss 画成一张图", {"plot_series"}),
    ("现在有哪些实验", {"lab_run_list"}),
    ("八阶段审计点还缺多少", {"lab_gap"}),
    ("研究员现在是加载着的吗", {"researcher_status", "lab_gpu"}),
    ("停掉研究员，我要跑训练", {"lab_abort"}),
    ("列出 /etc 下有什么文件", {"guarded_bash"}),
]


def main():
    tools = to_openai(mcp_tools())
    print("工具数:", len(tools))
    print("=" * 74)
    print("%-4s %-6s %-22s %s" % ("结果", "命中", "期望", "说法"))
    print("=" * 74)
    npass = 0
    detail = []
    for utt, expect in CASES:
        picks = []
        for _ in range(K):
            try:
                name, args, content = ask(utt, tools)
            except Exception as e:                              # noqa: BLE001
                name, args, content = None, None, "ERR %r" % (e,)
            picks.append(name)
            time.sleep(0.5)
        hit = all(p in expect for p in picks)
        if hit:
            npass += 1
        detail.append({"utterance": utt, "expect": sorted(expect), "picks": picks, "pass": hit})
        print("%-4s %-6s %-22s %s" % ("✅" if hit else "❌",
                                      "%d/%d" % (sum(1 for p in picks if p in expect), K),
                                      ",".join(sorted(expect))[:22], utt[:30]))
    print("=" * 74)
    print("pass^%d: %d/%d 用例全部命中" % (K, npass, len(CASES)))
    out = {"k": K, "cases": len(CASES), "passed": npass,
           "pass_pow_k_rate": round(npass / len(CASES), 3),
           "tools": len(tools), "detail": detail}
    with open("/root/dsh-harness/evidence/tool-eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已写入 evidence/tool-eval.json")


if __name__ == "__main__":
    main()
