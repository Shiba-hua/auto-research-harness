#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto-research-harness -- MCP tool server for the L40S research lab.

Pure stdlib (no pip deps). Speaks MCP over stdio: newline-delimited JSON-RPC 2.0.
Registered into DSH via @deepseek-ai/dsh-mcp-client as serverName "lab",
so every tool surfaces as  mcp__lab__<name>.

Design rules (see docs/08-验收与交付计划.md):
  * Deterministic first. Destructive ops never consult a model.
  * Every mutation is announced with its cost and is reversible or gated.
  * Reads are confined to LAB and HARNESS; writes only to HARNESS (+ explicit lab report paths).
"""
import json
import os
import re
import subprocess
import sys
import time
import hashlib

LAB = "/root/siton-tmp/rlvr-l40s-lab"
HARNESS = "/root/dsh-harness"
REQ = os.path.join(HARNESS, "requests")
EVID = os.path.join(HARNESS, "evidence")
VLLM = "/root/dsh-harness/bin/vllm-run"
SERVER = "lab"
VERSION = "0.1.0"

# ----------------------------------------------------------------------------- utils


def jdump(o):
    return json.dumps(o, ensure_ascii=False)


def sh(cmd, timeout=60, cwd=None):
    """Run a command; return (rc, stdout, stderr) with text decoded defensively."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
        return (p.returncode,
                p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace"))
    except subprocess.TimeoutExpired:
        return (124, "", "timeout after %ss" % timeout)
    except Exception as e:                                    # noqa: BLE001
        return (127, "", repr(e))


def gpu_now():
    rc, out, _ = sh("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu "
                    "--format=csv,noheader,nounits", timeout=20)
    if rc != 0:
        return {"ok": False, "error": "nvidia-smi failed"}
    used, free, util = [x.strip() for x in out.strip().split(",")[:3]]
    return {"ok": True, "used_mib": int(used), "free_mib": int(free), "util_pct": int(util)}


def vllm_pids():
    rc, out, _ = sh("pgrep -f '[v]llm serve'", timeout=15)
    return [int(x) for x in out.split() if x.strip().isdigit()] if rc == 0 else []


def safe_join(root, rel):
    """Resolve rel under root and refuse escapes."""
    p = os.path.realpath(os.path.join(root, rel.lstrip("/")))
    if not (p == root or p.startswith(root + os.sep)):
        raise ValueError("path escapes %s: %s" % (root, rel))
    return p


def run_dirs():
    d = os.path.join(LAB, "results")
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isdir(p):
            out.append((os.path.getmtime(p), name, p))
    out.sort(reverse=True)
    return out


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:                                          # noqa: BLE001
        return default


def tail_file(p, n=20):
    rc, out, _ = sh(["tail", "-n", str(n), p], timeout=20)
    return out if rc == 0 else "(cannot read %s)" % p


# ----------------------------------------------------------------------------- tools

def t_lab_run_list(a):
    """List lab runs, newest first."""
    limit = int(a.get("limit", 20))
    rows = []
    for mt, name, p in run_dirs()[:limit]:
        st = read_json(os.path.join(p, "status.json"), {}) or {}
        rows.append({
            "run_id": name,
            "state": st.get("state") or st.get("status") or ("unknown" if not st else "?"),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)),
            "has_status": bool(st),
        })
    return {"ok": True, "count": len(rows), "results_root": os.path.join(LAB, "results"),
            "runs": rows,
            "note": "只读扫描 results/ 目录，未修改任何内容"}


def t_lab_run_status(a):
    rid = str(a.get("run_id", "")).strip()
    if not rid:
        return {"ok": False, "error": "run_id 必填"}
    res = safe_join(os.path.join(LAB, "results"), rid)
    ev = safe_join(os.path.join(LAB, "evidence"), rid)
    out = {"ok": True, "run_id": rid, "results_dir": res, "evidence_dir": ev}
    for key, path in (("status", os.path.join(res, "status.json")),
                      ("job", os.path.join(ev, "job", "job.json")),
                      ("runtime", os.path.join(res, "runtime.json")),
                      ("gpu_release", os.path.join(ev, "job", "gpu-release.json"))):
        if os.path.isfile(path):
            out[key] = read_json(path)
    for key, path in (("stdout_tail", os.path.join(ev, "job", "stdout.log")),
                      ("supervisor_tail", os.path.join(ev, "job", "supervisor.log"))):
        if os.path.isfile(path):
            out[key] = tail_file(path, 15)
    if not any(k in out for k in ("status", "job", "stdout_tail")):
        out["ok"] = False
        out["error"] = "找不到该 run 的任何状态文件（确认 run_id 是否正确）"
    return out


def t_lab_tail(a):
    rid = str(a.get("run_id", "")).strip()
    n = int(a.get("lines", 20))
    which = str(a.get("file", "stdout.log"))
    if not rid:
        return {"ok": False, "error": "run_id 必填"}
    p = safe_join(os.path.join(LAB, "evidence", rid, "job"), which)
    if not os.path.isfile(p):
        return {"ok": False, "error": "文件不存在: %s" % p}
    return {"ok": True, "file": p, "lines": n, "tail": tail_file(p, n)}


def t_lab_gpu(a):
    g = gpu_now()
    rc, out, _ = sh("nvidia-smi --query-compute-apps=pid,used_memory,process_name "
                    "--format=csv,noheader", timeout=20)
    g["compute_apps"] = out.strip() or "(none)"
    g["vllm_pids"] = vllm_pids()
    g["researcher_loaded"] = bool(g["vllm_pids"])
    return g


# ---- GPU arbitration: the ONE destructive op that must never consult a model ----

def _credential():
    """Release is proven only when VRAM is low AND no orphan remains."""
    u = gpu_now()
    left = vllm_pids()
    issued = bool(u.get("ok")) and u["used_mib"] < 1000 and len(left) == 0
    return {"issued": issued, "gpu_used_mib": u.get("used_mib"),
            "gpu_free_mib": u.get("free_mib"), "leftover_pids": left,
            "rule": "gpu_used_mib < 1000 AND leftover_pids == 0",
            "note": ("必须 SIGTERM guest 内的 vllm 宿主 PID；"
                     "杀 PRoot 包装进程只会把它孤儿化并继续占显存")}


def t_lab_abort(a):
    """Stop the researcher (vLLM) -- deterministic, credential-gated."""
    confirm = bool(a.get("confirm"))
    pids = vllm_pids()
    cost = ("将中止研究员推理服务（vLLM, %d 个进程, 当前占用 %s MiB 显存）。"
            "代价：正在进行的推理作废；训练与研究员不能共存，所以释放显存是起训练的前置条件。"
            % (len(pids), gpu_now().get("used_mib")))
    if not confirm:
        return {"ok": True, "action": "needs_confirmation", "will_do": cost,
                "pids": pids,
                "resume_hint": "确认后带 confirm=true 再次调用"}
    if not pids:
        return {"ok": True, "action": "nothing_to_do", "credential": _credential()}
    rc, _, err = sh(["kill", "-TERM"] + [str(p) for p in pids], timeout=20)
    t0 = time.time()
    while time.time() - t0 < 90:
        if gpu_now().get("used_mib", 99999) < 1000 and not vllm_pids():
            break
        time.sleep(2)
    cred = _credential()
    cred["elapsed_s"] = round(time.time() - t0, 1)
    cred["kill_rc"] = rc
    if err.strip():
        cred["kill_stderr"] = err.strip()
    return {"ok": True, "action": "aborted", "credential": cred,
            "message": ("显存已释放 ✅" if cred["issued"]
                        else "⚠️ 显存未确认释放，禁止启动训练")}


# ---- read-only lab queries ----

def t_lab_gap(a):
    """Report the 8-stage x 2-domain x 6-budget gap from the curve protocol."""
    stages = ["base", "pure_grpo", "cot_sft", "cot_sft_grpo",
              "cod_sft", "cod_sft_grpo", "shorthand_cod_sft", "shorthand_cod_sft_grpo"]
    budgets = [512, 1024, 1536, 2048, 3072, 4096]
    found = {}
    root = os.path.join(LAB, "results")
    for _, name, p in run_dirs():
        for s in stages:
            if re.search(r"(^|[-_])%s($|[-_])" % re.escape(s.replace("_", "[-_]")), name):
                found.setdefault(s, []).append(name)
    return {"ok": True,
            "grid": {"stages": stages, "budgets": budgets,
                     "domains": ["math(gsm8k, 主预算1536)", "science(arc, 主预算1024)"],
                     "total_points": 96},
            "observed_stage_dirs": {k: sorted(set(v))[:6] for k, v in found.items()},
            "note": ("权威口径见 docs/curve_protocol.md；本工具只做目录名匹配的粗略盘点，"
                     "精确缺口以 student_audit_phase2.md / arc_student_pure_audit_progress.md 为准。"
                     "已知：48/96 已完成，主要比较 6/14")}


# ---- event-sourced request queue ----

def _q_paths():
    os.makedirs(REQ, exist_ok=True)
    return (os.path.join(REQ, "events.jsonl"), os.path.join(REQ, "current.json"))


def _q_load():
    ev, cur = _q_paths()
    items = {}
    if os.path.isfile(ev):
        with open(ev, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:                              # noqa: BLE001
                    continue
                i = e.get("id")
                if not i:
                    continue
                if e.get("op") == "created":
                    items[i] = {"id": i, "status": "queued", "text": e.get("text", ""),
                                "tier": e.get("tier", "researcher"),
                                "seq": e.get("seq", 0), "history": []}
                elif i in items:
                    if e.get("op") == "cancelled":
                        items[i]["status"] = "cancelled"
                    elif e.get("op") == "claimed":
                        items[i]["status"] = "claimed"
                    elif e.get("op") == "completed":
                        items[i]["status"] = "done"
                    elif e.get("op") == "edited":
                        items[i]["text"] = e.get("text", items[i]["text"])
                    elif e.get("op") == "retargeted":
                        items[i]["tier"] = e.get("tier", items[i]["tier"])
                    elif e.get("op") == "reprioritised":
                        items[i]["seq"] = e.get("seq", items[i]["seq"])
                    items[i]["history"].append(e.get("op"))
    ordered = sorted(items.values(), key=lambda x: (x["seq"], x["id"]))
    return ordered


def _q_append(evt):
    ev, cur = _q_paths()
    with open(ev, "a", encoding="utf-8") as f:
        f.write(jdump(evt) + "\n")
    tmp = cur + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_q_load(), f, ensure_ascii=False, indent=1)
    os.replace(tmp, cur)          # atomic, same directory


def t_req_list(a):
    items = _q_load()
    return {"ok": True, "count": len(items),
            "queue": [{k: it[k] for k in ("id", "status", "tier", "text")} for it in items]}


def t_req_add(a):
    text = str(a.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "text 必填"}
    items = _q_load()
    seq = int(a.get("priority", len(items) + 1))
    rid = "req-%d" % (int(time.time() * 1000) % 10 ** 9)
    _q_append({"op": "created", "id": rid, "text": text, "seq": seq,
               "tier": a.get("tier", "researcher"), "at": time.time()})
    return {"ok": True, "id": rid, "status": "queued", "text": text}


def _q_must_be_queued(rid):
    for it in _q_load():
        if it["id"] == rid:
            return it
    return None


def t_req_edit(a):
    rid = str(a.get("id", ""))
    it = _q_must_be_queued(rid)
    if not it:
        return {"ok": False, "error": "找不到 %s" % rid}
    if it["status"] != "queued":
        return {"ok": False, "error": "只能编辑 queued 状态的请求（当前 %s）" % it["status"]}
    if a.get("text"):
        _q_append({"op": "edited", "id": rid, "text": str(a["text"]), "at": time.time()})
    if a.get("tier"):
        _q_append({"op": "retargeted", "id": rid, "tier": str(a["tier"]), "at": time.time()})
    if a.get("priority") is not None:
        _q_append({"op": "reprioritised", "id": rid, "seq": int(a["priority"]), "at": time.time()})
    return {"ok": True, "id": rid, "updated": True}


def t_req_cancel(a):
    rid = str(a.get("id", ""))
    it = _q_must_be_queued(rid)
    if not it:
        return {"ok": False, "error": "找不到 %s" % rid}
    if it["status"] != "queued":
        return {"ok": False,
                "error": "已认领/进行中的请求不可删除，只能追加修正或叫停（当前 %s）" % it["status"]}
    _q_append({"op": "cancelled", "id": rid, "at": time.time()})
    return {"ok": True, "id": rid, "status": "cancelled"}


# ---- on-demand researcher (GPU arbitration) --------------------------------
#
# 为什么不能直接由这里启动 vLLM：本服务器跑在 dsh-run 的 PRoot guest 内，
# 再调 vllm-run 就是嵌套 PRoot —— 实测失败（内层 proot 二进制不可见）。
# 所以用请求文件：本进程写 state/researcher.request，宿主上的看门狗负责拉起。
# 两边都能看到 /root/dsh-harness/state/（guest 有该路径的 bind）。

import urllib.request

REQ_FILE = os.path.join(HARNESS, "state", "researcher.request")
STATUS_FILE = os.path.join(HARNESS, "state", "researcher.status")
RESEARCHER_URL = "http://127.0.0.1:18001/v1/models"


def _http_code(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:                                          # noqa: BLE001
        return 0


def _researcher_ready():
    return _http_code(RESEARCHER_URL) not in (0,)


def t_researcher_status(a):
    ready = _researcher_ready()
    st = read_json(STATUS_FILE, {}) or {}
    return {"ok": True, "ready": ready, "port": 18001,
            "gpu": gpu_now(), "vllm_pids": vllm_pids(),
            "last_request": st,
            "note": "ready=true 表示研究员已在服务，可以派活"}


def t_researcher_up(a):
    """按需拉起研究员（vLLM）。GPU 不空闲则知情拒绝，不擅自抢占。"""
    g = gpu_now()
    if not g.get("ok"):
        return {"ok": False, "error": "无法读取 GPU 状态"}
    if _researcher_ready():
        return {"ok": True, "action": "already_up", "gpu": g}

    # GPU 被别的进程占着 → 知情拒绝（研究员与训练不能共存）
    if g["used_mib"] >= 1000:
        return {"action": "refused", "ok": False,
                "why": "研究员与训练不能共存：当前显存已用 %d MiB，起研究员会 OOM" % g["used_mib"],
                "gpu": g,
                "options": ["A. 排队等待，稍后再试",
                            "B. 中止占用者（lab_abort，会作废正在跑的实验）"]}

    os.makedirs(os.path.dirname(REQ_FILE), exist_ok=True)
    with open(REQ_FILE, "w") as f:
        f.write(jdump({"requested_at": time.time(), "by": "researcher_up"}))
    deadline = time.time() + int(a.get("wait_seconds", 240))
    while time.time() < deadline:
        if _researcher_ready():
            break
        time.sleep(5)
    ready = _researcher_ready()
    return {"ok": ready, "action": "started" if ready else "timeout",
            "ready": ready, "waited_s": int(time.time() - (deadline - int(a.get("wait_seconds", 240)))),
            "gpu": gpu_now(), "vllm_pids": vllm_pids(),
            "message": ("研究员已就绪（Qwen3-32B-AWQ, :18001）" if ready
                        else "已发出启动请求但超时未就绪；用 researcher_status 复查")}


# ---- guarded bash (the bash guardrail) ----

WHITELIST = {
    "ls", "cat", "head", "tail", "grep", "rg", "wc", "stat", "file", "du", "df",
    "find", "ps", "pgrep", "echo", "date", "which", "env", "id", "uname",
    "sort", "uniq", "cut", "jq", "nvidia-smi", "pwd", "hostname", "whoami",
}
GIT_SAFE = {"status", "log", "diff", "show", "branch", "tag", "remote",
            "ls-files", "rev-parse", "describe", "shortlog", "blame"}
DENY_PAT = [
    (r"\.venv", "禁止触碰 lab 的 .venv"),
    (r"\brm\s+-rf\b", "禁止 rm -rf"),
    (r"(curl|wget)[^|]*\|\s*(ba)?sh", "禁止 curl|sh 式远程执行"),
    (r"git\s+push\s+.*--force", "禁止 force push"),
    (r"git\s+reset\s+--hard", "禁止 git reset --hard"),
    (r"git\s+checkout\s+codex/experiments", "禁止切回 codex 基线分支（会污染基线）"),
    (r"\bdd\s+of=/dev/", "禁止裸写块设备"),
    (r"\bmkfs\b", "禁止格式化"),
    (r"chmod\s+777", "禁止 chmod 777"),
    (r">\s*/(etc|usr|boot|sys|proc)/", "禁止写系统目录"),
    (r"\bmodels/|\bdata/", "禁止写 lab 的 models/ 与 data/（在 .gitignore 保护外，git 救不回）"),
    (r":\(\)\s*\{", "禁止 fork bomb"),
]
META_CHARS = ["|", ">", "<", "`", "$(", ";", "&&", "||", "&"]


def classify_command(cmd):
    c = (cmd or "").strip()
    if not c:
        return {"verdict": "reject", "reason": "空命令"}
    for pat, why in DENY_PAT:
        if re.search(pat, c):
            return {"verdict": "reject", "reason": why, "matched": pat}
    if any(m in c for m in META_CHARS):
        return {"verdict": "ask",
                "reason": "含 shell 元字符（管道/重定向/命令连接/替换），不做聪明的解析——一律转审批",
                "rule": "META_CHARS"}
    parts = c.split()
    if not parts:
        return {"verdict": "reject", "reason": "空命令"}
    head = os.path.basename(parts[0])
    if head == "git":
        sub = parts[1] if len(parts) > 1 else ""
        if sub in GIT_SAFE:
            return {"verdict": "allow", "reason": "git 只读子命令", "rule": "GIT_SAFE:%s" % sub}
        return {"verdict": "ask", "reason": "git 写操作需审批", "rule": "GIT_WRITE:%s" % sub}
    if head in WHITELIST:
        return {"verdict": "allow", "reason": "白名单只读命令", "rule": "WHITELIST:%s" % head}
    return {"verdict": "ask", "reason": "不在白名单，需审批", "rule": "DEFAULT_ASK"}


def t_guarded_bash(a):
    cmd = str(a.get("command", ""))
    v = classify_command(cmd)
    if v["verdict"] != "allow":
        return {"ok": True, "executed": False, "command": cmd, **v,
                "note": "未执行。allow=规则放行；ask=转审批；reject=规则拒绝"}
    rc, out, err = sh(cmd, timeout=int(a.get("timeout", 60)),
                      cwd=HARNESS)
    return {"ok": True, "executed": True, "verdict": "allow", "rule": v.get("rule"),
            "rc": rc, "stdout": out[-8000:], "stderr": err[-2000:]}


# ---- plotting -------------------------------------------------------------
#
# 本服务器运行在 dsh-run 的 PRoot guest 内，该 guest 里：
#   /opt/rlvr-vllm-venv/bin/python   有 matplotlib 3.10.3
#   <LAB>/src                        有 rlvr_lab
# 所以直接用子进程调用 lab 已验证的绘图代码，不执行任何模型生成的代码。

GUEST_PY = "/opt/rlvr-vllm-venv/bin/python"
LAB_SRC = os.path.join(LAB, "src")
PLOT_OUT = os.path.join(HARNESS, "evidence", "plots")


def _run_guest_py(code, timeout=300):
    env = dict(os.environ)
    env["PYTHONPATH"] = LAB_SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = "/tmp/mplcfg"
    try:
        p = subprocess.run([GUEST_PY, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout, env=env)
        return (p.returncode, p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace"))
    except Exception as e:                                     # noqa: BLE001
        return (127, "", repr(e))


def _img_block(path):
    with open(path, "rb") as f:
        import base64
        return {"type": "image", "data": base64.b64encode(f.read()).decode("ascii"),
                "mimeType": "image/png"}


def t_plot_metrics(a):
    """渲染某个 run 的标准诊断图。调用 lab 已验证的 rlvr_lab.plot.render()。"""
    rid = str(a.get("run_id", "")).strip()
    if not rid:
        return {"ok": False, "error": "run_id 必填"}
    run = safe_join(os.path.join(LAB, "results"), rid)
    if not os.path.isdir(run):
        return {"ok": False, "error": "找不到 run 目录: %s" % run}
    if not os.path.isfile(os.path.join(run, "metrics.jsonl")):
        return {"ok": False,
                "error": ("该目录没有 metrics.jsonl，不是训练 run（可能是概览/聚合目录）。"
                          "可用 lab_run_list 找有遥测的 run。")}
    out = safe_join(PLOT_OUT, rid)
    os.makedirs(out, exist_ok=True)
    code = (
        "import json,sys\n"
        "from rlvr_lab import plot as P\n"
        "r = P.render(%r, %r)\n"
        "print(json.dumps({'charts':[c if isinstance(c,str) else c.get('name') for c in r['charts']],"
        "'warnings':r.get('warnings',[])}))\n" % (run, out)
    )
    rc, so, se = _run_guest_py(code, timeout=int(a.get("timeout", 300)))
    if rc != 0:
        return {"ok": False, "error": "渲染失败", "stderr": se[-800:]}
    try:
        info = json.loads(so.strip().splitlines()[-1])
    except Exception:                                          # noqa: BLE001
        return {"ok": False, "error": "无法解析渲染结果", "stdout": so[-400:]}
    kinds = a.get("kinds") or info.get("charts") or []
    if kinds == ["all"]:
        kinds = info.get("charts") or []
    blocks = []
    shown = []
    for k in kinds:
        for ext in (".png",):
            p = os.path.join(out, k + ext)
            if os.path.isfile(p):
                blocks.append(_img_block(p))
                shown.append(k + ext)
    summary = {"ok": True, "run_id": rid,
               "charts_available": info.get("charts"),
               "images_shown": shown,
               "output_dir": out,
               "warnings": info.get("warnings")}
    return {"_content": [{"type": "text", "text": jdump(summary)}] + blocks}


def t_plot_series(a):
    """任意 X-Y 折线：从 run 的 JSONL 里取列，出图。补 lab 硬编码面板没有的视角。"""
    rid = str(a.get("source", a.get("run_id", ""))).strip()
    xk = str(a.get("x", "step")).strip()
    yk = a.get("y") or []
    if isinstance(yk, str):
        yk = [yk]
    if not rid or not yk:
        return {"ok": False, "error": "需要 source(=run_id) 与 y(至少一个列名)"}
    run = safe_join(os.path.join(LAB, "results"), rid)
    fn = a.get("file", "metrics.jsonl")
    src = safe_join(run, fn)
    if not os.path.isfile(src):
        return {"ok": False, "error": "找不到数据文件: %s" % src}
    out = safe_join(PLOT_OUT, rid)
    os.makedirs(out, exist_ok=True)
    name = str(a.get("output", "series"))[:60].replace("/", "_")
    png = os.path.join(out, name + ".png")
    params = jdump({"src": src, "x": xk, "y": yk, "png": png, "title": rid})
    helper = os.path.join(HARNESS, "mcp", "plot_series.py")
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = "/tmp/mplcfg"
    try:
        p = subprocess.run([GUEST_PY, helper, params], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=int(a.get("timeout", 180)), env=env)
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "error": "绘图进程异常: %r" % (e,)}
    if p.returncode != 0:
        return {"ok": False, "error": "绘图失败",
                "stderr": p.stderr.decode("utf-8", "replace")[-600:]}
    try:
        info = json.loads(p.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except Exception:                                          # noqa: BLE001
        info = {}
    if not info.get("plotted"):
        return {"ok": False, "error": "指定的列在数据里没有数值", "detail": info}
    return {"_content": [{"type": "text", "text": jdump(
        {"ok": True, "run_id": rid, "x": xk, "y": info.get("plotted"),
         "rows": info.get("rows"), "png": png})}, _img_block(png)]}


# ----------------------------------------------------------------------------- registry

TOOLS = [
    ("lab_run_list", "列出服务器上的实验 run（只读）", {}, t_lab_run_list),
    ("lab_run_status", "查询某个 run 的状态、job 信息与日志尾部（只读）",
     {"run_id": {"type": "string"}, "properties": {}}, t_lab_run_status),
    ("lab_tail", "读取某个 run 的日志尾部（只读）",
     {"run_id": {"type": "string"}, "lines": {"type": "integer"},
      "file": {"type": "string"}}, t_lab_tail),
    ("lab_gpu", "查询 GPU 显存、利用率、占用进程，以及研究员是否已装载（只读）", {}, t_lab_gpu),
    ("lab_gap", "盘点八阶段曲线协议的缺口（只读）", {}, t_lab_gap),
    ("lab_abort", "中止研究员推理服务并释放显存；返回显存释放凭证（确定性规则，不经模型）",
     {"confirm": {"type": "boolean"}}, t_lab_abort),
    ("researcher_up", "按需拉起研究员（Qwen3-32B-AWQ, :18001）。GPU 不空闲则知情拒绝，不擅自抢占",
     {"wait_seconds": {"type": "integer"}}, t_researcher_up),
    ("researcher_status", "查询研究员是否已就绪（只读）", {}, t_researcher_status),
    ("guarded_bash", "受护栏保护的通用命令执行：白名单放行 / 黑名单拒绝 / 含 shell 元字符一律转审批",
     {"command": {"type": "string"}, "timeout": {"type": "integer"}}, t_guarded_bash),
    ("req_list", "查看待办队列里已登记了什么（只读，不执行任务）", {}, t_req_list),
    ("req_add", "把一条待办【登记】进请求队列——只记录，不执行、也不去访问它。"
                "用户说「记一下 / 记下来 / 帮我记 / 登记 / 加到队列 / 待会做 / 排个队」时用它，"
                "后面的内容是待办文字，不要试图去完成它。",
     {"text": {"type": "string"}, "tier": {"type": "string"},
      "priority": {"type": "integer"}}, t_req_add),
    ("req_edit", "修改队列中 queued 状态的请求",
     {"id": {"type": "string"}, "text": {"type": "string"},
      "tier": {"type": "string"}, "priority": {"type": "integer"}}, t_req_edit),
    ("req_cancel", "取消队列中 queued 状态的请求",
     {"id": {"type": "string"}}, t_req_cancel),
    ("plot_metrics", "绘制某个 run 的标准诊断图（调用 lab 已验证的绘图代码）",
     {"run_id": {"type": "string"}, "kinds": {"type": "array", "items": {"type": "string"}}},
     t_plot_metrics),
    ("plot_series", "绘制任意 X-Y 折线（薄封装，补 lab 硬编码面板没有的视角）",
     {"source": {"type": "string"}, "x": {"type": "string"},
      "y": {"type": "array", "items": {"type": "string"}}}, t_plot_series),
]
BY_NAME = {t[0]: t for t in TOOLS}


def schema_for(name, props):
    p = {k: v for k, v in props.items() if k != "properties"}
    req = [k for k in props if k != "properties" and k in props]
    return {"type": "object", "properties": p, "required": []}


def tools_list():
    out = []
    for name, desc, props, _fn in TOOLS:
        out.append({"name": name, "description": desc,
                    "inputSchema": {"type": "object", "properties": props}})
    return out


# ----------------------------------------------------------------------------- MCP loop

def handle(msg):
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "auto-research-harness-lab", "version": VERSION}}}
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools_list()}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        ent = BY_NAME.get(name)
        if not ent:
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": jdump({"ok": False,
                                                            "error": "unknown tool %s" % name})}],
                "isError": True}}
        try:
            res = ent[3](args)
            is_err = isinstance(res, dict) and res.get("ok") is False
        except Exception as e:                                 # noqa: BLE001
            res = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            is_err = True
        # 富内容：工具可返回 {"_content":[...]} 携带图片等 MCP 内容块
        if isinstance(res, dict) and "_content" in res:
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": res["_content"], "isError": bool(is_err)}}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": jdump(res)}], "isError": bool(is_err)}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": "method not found: %s" % method}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:                                      # noqa: BLE001
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(jdump(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
