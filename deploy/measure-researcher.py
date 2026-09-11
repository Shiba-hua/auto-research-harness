#!/usr/bin/env python3
# Measurement client -- runs INSIDE the PRoot guest (Python 3.11, clean UTF-8),
# so Chinese prompts never pass through a shell or an old host Python.
# Talks to the vLLM server on 127.0.0.1:18001. Emits one JSON object per line.
import json, time, urllib.request, sys

BASE = "http://127.0.0.1:18001/v1/chat/completions"
NAME = "researcher"


def post(payload, timeout=900):
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def chat(text, max_tokens, thinking=False, tools=None):
    p = {"model": NAME, "max_tokens": max_tokens,
         "temperature": 0.6, "top_p": 0.95, "top_k": 20,
         "chat_template_kwargs": {"enable_thinking": thinking},
         "messages": [{"role": "user", "content": text}]}
    if tools:
        p["tools"] = tools
    return post(p)


def out(phase, obj):
    obj["phase"] = phase
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def summarize(d, el):
    u = d.get("usage", {}) or {}
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {
        "ok": True,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "elapsed_s": round(el, 2),
        "decode_tps": round(u.get("completion_tokens", 0) / el, 2) if el > 0 else None,
        "finish": ch.get("finish_reason"),
        "preview": (msg.get("content") or "")[:70],
    }


# ---------- 1. decode, short ----------
try:
    d, el = chat("用中文简要说明什么是 RLVR，200 字以内。", 256)
    out("decode_short", summarize(d, el))
except Exception as e:
    out("decode_short", {"ok": False, "error": repr(e)})

# ---------- 2. decode, long ----------
try:
    d, el = chat("请写一段关于强化学习与可验证奖励（RLVR）的长文，尽量详细。", 1024)
    out("decode_long", summarize(d, el))
except Exception as e:
    out("decode_long", {"ok": False, "error": repr(e)})

# ---------- 3. prefill: big prompt, single output token ----------
try:
    filler = "RLVR experiment log line. " * 6000          # ~150k chars
    d, el = chat("下面是一段重复文本，请只回答它一共出现了多少个单词。\n" + filler, 1)
    s = summarize(d, el)
    s["prefill_tps"] = round(s["prompt_tokens"] / el, 2) if el > 0 else None
    out("prefill_big", s)
except Exception as e:
    out("prefill_big", {"ok": False, "error": repr(e)})

# ---------- 4. 40k context acceptance ----------
try:
    filler = "RLVR experiment log line. " * 6200
    prompt = "以下是实验日志。请只回答最后一行写的是什么。\n" + filler + "\n最后一行：确认"
    d, el = chat(prompt, 16)
    s = summarize(d, el)
    s["note"] = "目标：prompt_tokens 接近 38000-40000 而不报错"
    out("ctx_40k", s)
except Exception as e:
    out("ctx_40k", {"ok": False, "error": repr(e)})

# ---------- 5. tool calling ----------
try:
    tools = [{"type": "function", "function": {
        "name": "lab_run_list",
        "description": "列出服务器上正在运行和最近的实验",
        "parameters": {"type": "object", "properties": {}, "required": []}}}]
    d, el = chat("现在有哪些实验在跑？用工具查。", 256, tools=tools)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    tc = msg.get("tool_calls") or []
    out("tool_call", {
        "ok": True,
        "has_tool_calls": bool(tc),
        "name": tc[0]["function"]["name"] if tc else None,
        "args": tc[0]["function"]["arguments"] if tc else None,
        "content_preview": (msg.get("content") or "")[:80],
        "elapsed_s": round(el, 2),
    })
except Exception as e:
    out("tool_call", {"ok": False, "error": repr(e)})

# ---------- 6. thinking mode actually thinks ----------
try:
    d, el = chat("一个班有 30 人，60% 是女生，女生里一半戴眼镜。戴眼镜的女生有多少人？", 700, thinking=True)
    out("thinking_mode", summarize(d, el))
except Exception as e:
    out("thinking_mode", {"ok": False, "error": repr(e)})

print(json.dumps({"phase": "client_done"}), flush=True)
