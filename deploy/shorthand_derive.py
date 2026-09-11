#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速记法规则式派生（P0 #1）

依据 docs/shorthand_rule_transform_feasibility.md + docs/shorthand_derivation_plan.md。

设计底线（不可越）：
  1. 只做**可验证的文字变换**，不调用计算器、不改原数值与最终答案
  2. 每条派生都必须携带**逐字一致的源 span**
  3. 派生文本是【派生内容】，与原文在学生可见处可区分
  4. 结构核验通过 != 语义等价；固定样本仍需人工/agent 审读
  5. 无法消歧 → 拒绝，并记录拒绝原因（不猜）

不做：整篇精确逆变换、另造新例子、改 loader、借用 v4 身份。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REVIEW_DIR = Path("/root/siton-tmp/rlvr-l40s-lab/evidence/shorthand-source-mechanism-review-001")
OUT_DIR = Path("/root/dsh-harness/evidence/shorthand-derivation")
DERIVED_MARK = "【派生说明】"

# ── R1 变量/别名绑定 ────────────────────────────────────────────────
# 触发：明确的 "let K be …" / "if Katherine is … K years old" 式定义，
#       且后续真的引用了该符号。
R1_DEFINE = re.compile(
    r"(?:let|suppose|assume|if)\s+(?:the\s+)?[A-Za-z][A-Za-z ]{0,24}?\s+"
    r"(?:be|is)\s+(?:,?\s*say\s*,?\s*)?([A-Z])\b[^.;]{0,60}", re.I)
R1_REF = re.compile(r"\b([A-Z])\b(?!\w)")


def rule_r1(text):
    """变量绑定：给出符号定义 + 至少一处后续引用，才能展开。"""
    for m in R1_DEFINE.finditer(text):
        sym = m.group(1)
        span = m.group(0).strip()
        after = text[m.end():]
        refs = [r for r in R1_REF.finditer(after) if r.group(1) == sym]
        if not refs:
            yield {"status": "rejected", "rule": "R1", "symbol": sym,
                   "source_span": span, "span_start": m.start(), "span_end": m.end(),
                   "reason": "定义了符号但后文无真实引用，不足以展开"}
            continue
        # 同一字母若被再次定义 → 无法消歧，拒绝（规则明确要求）
        if len(R1_DEFINE.findall(text)) > 1 and re.search(
                r"\b(?:let|if)\s+[^.;]{0,24}?\b%s\b\s+(?:be|is)\b" % sym, text[m.end():], re.I):
            yield {"status": "rejected", "rule": "R1", "symbol": sym,
                   "source_span": span, "span_start": m.start(), "span_end": m.end(),
                   "reason": "同一字母被重复定义，无法消歧"}
            continue
        first = refs[0]
        ref_span = after[first.start():first.end()]
        yield {"status": "applicable", "rule": "R1", "symbol": sym,
               "source_span": span, "span_start": m.start(), "span_end": m.end(),
               "ref_span": ref_span,
               "derived": "%s 源文用符号 %s 指代上文的量（原文：「%s」）。"
                          "后续出现的 %s 指同一实体，不是新变量。" % (DERIVED_MARK, sym, span, sym),
               "reason": "绑定定义与后续引用都在源内，可安全说明"}


# ── R2 等式中的数量与单位省略 ────────────────────────────────────────
# 触发：紧凑数式 + 操作数含义可从绑定上下文确定。
R2_EQ = re.compile(r"([A-Za-z\s]{2,24}?)\s*([=])\s*([^.;\n]{1,40})")
R2_NUM = re.compile(r"\d+(?:\.\d+)?")


def rule_r2(text, bindings):
    """数量/单位省略：只**说明**操作数指向什么，不重算、不猜单位。"""
    for m in R2_EQ.finditer(text):
        lhs, rhs = m.group(1).strip(), m.group(3).strip()
        span = m.group(0).strip()
        if not R2_NUM.search(rhs):
            continue
        if not any(s in lhs or s in rhs for s in bindings):
            yield {"status": "rejected", "rule": "R2",
                   "source_span": span, "span_start": m.start(), "span_end": m.end(),
                   "reason": "操作数无法从绑定上下文确定含义，拒绝展开"}
            continue
        yield {"status": "applicable", "rule": "R2",
               "source_span": span, "span_start": m.start(), "span_end": m.end(),
               "derived": "%s 原式「%s」中的数量沿用上文已绑定的量；"
                          "本说明只指出它指向什么，不重算也不为裸数字补单位。"
                          % (DERIVED_MARK, span),
               "reason": "操作数含义可从绑定上下文确定"}


# ── R3 隐式乘法/表达式缩写 ──────────────────────────────────────────
# 触发：已定义量后的 `30w` 型写法。
R3_IMPLICIT = re.compile(r"\b(\d+)\s*([A-Za-z])\b(?!\w)")


def rule_r3(text, bindings):
    """隐式乘法：仅当该字母是已绑定符号时可说明。"""
    for m in R3_IMPLICIT.finditer(text):
        n, sym = m.group(1), m.group(2)
        span = m.group(0)
        if sym not in bindings:
            continue  # 不是隐式乘法（可能是单位等），静默跳过而非拒绝
        yield {"status": "applicable", "rule": "R3", "symbol": sym,
               "source_span": span, "span_start": m.start(), "span_end": m.end(),
               "derived": "%s 原式「%s」是省略乘号的写法，等于 %s × %s，"
                          "其中 %s 就是上文绑定的量。" % (DERIVED_MARK, span, n, sym, sym),
               "reason": "字母是源内已绑定符号，含源内数量绑定"}


# ── R4 重复关系短语 ────────────────────────────────────────────────
# 触发：同一个关系词在源内出现 ≥2 次（回指），且中间有已绑定实体。
R4_REL = re.compile(r"\b(more than|less than|older than|younger than|"
                    r"times as many|twice as many|half as many)\b", re.I)


def rule_r4(text):
    """重复关系短语：只有确实出现回指时才恢复关系方向。"""
    hits = list(R4_REL.finditer(text))
    if len(hits) < 2:
        return
    counts = {}
    for h in hits:
        counts.setdefault(h.group(0).lower(), []).append(h)
    for rel, ms in counts.items():
        if len(ms) < 2:
            continue
        # 关系方向必须一致；若源中同时存在否定/反例，保守拒绝
        seg = text[ms[0].start():ms[-1].end()]
        if re.search(r"\b(?:not|but|however|instead|wrong|incorrect|actually)\b", seg, re.I):
            yield {"status": "rejected", "rule": "R4", "relation": rel,
                   "source_span": text[ms[0].start():ms[1].end()],
                   "span_start": ms[0].start(), "span_end": ms[1].end(),
                   "reason": "回指段落含否定/自我纠正，关系方向不得统一改写"}
            continue
        yield {"status": "applicable", "rule": "R4", "relation": rel,
               "source_span": seg[:120], "span_start": ms[0].start(), "span_end": ms[-1].end(),
               "derived": "%s 关系词「%s」在源内出现 %d 次，后续是**回指**同一关系，"
                          "方向不变。" % (DERIVED_MARK, rel, len(ms)),
               "reason": "关系框架已给出且后文确实回指"}


# ── R5 选项字母别名 ────────────────────────────────────────────────
R5_OPTION = re.compile(r"\b(?:option|choice)\s+([A-E])\b", re.I)


def rule_r5(text):
    for m in R5_OPTION.finditer(text):
        letter = m.group(1).upper()
        span = m.group(0)
        # 必须存在该字母到完整文本的映射才可展开；此处只能识别出引用，
        # 映射不存在时拒绝（不凭空猜测选项内容）。
        yield {"status": "rejected", "rule": "R5", "letter": letter,
               "source_span": span, "span_start": m.start(), "span_end": m.end(),
               "reason": "识别到选项字母引用，但源内未给出该字母到完整选项文本的映射，"
                         "不凭外部知识补全"}


# ── 主流程 ─────────────────────────────────────────────────────────

def derive_one(row):
    text = row.get("draft_response") or ""
    recs = []
    b = rule_r1(text)
    r1 = list(b)
    recs.extend(r1)
    bindings = {r["symbol"] for r in r1 if r["status"] == "applicable"}
    recs.extend(rule_r2(text, bindings))
    recs.extend(rule_r3(text, bindings))
    recs.extend(rule_r4(text))
    recs.extend(rule_r5(text))
    out = []
    for r in recs:
        r = dict(r)
        r["review_id"] = row.get("review_id")
        r["source_id"] = row.get("source_id")
        r["project"] = row.get("project")
        r["draft_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        # 结构核验：源 span 必须逐字出现在原文里
        sp = r.get("source_span") or ""
        r["span_verbatim_in_source"] = bool(sp) and (sp in text)
        # 派生文本与原文必须可区分
        r["derived_marked"] = str(r.get("derived", "")).startswith(DERIVED_MARK)
        out.append(r)
    return out


def main():
    rows = [json.loads(l) for l in
            (REVIEW_DIR / "source_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_recs = []
    for row in rows:
        all_recs.extend(derive_one(row))
    (OUT_DIR / "derivations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_recs), encoding="utf-8")

    app = [r for r in all_recs if r["status"] == "applicable"]
    rej = [r for r in all_recs if r["status"] == "rejected"]
    bad_span = [r for r in all_recs if not r.get("span_verbatim_in_source")]
    unmarked = [r for r in app if not r.get("derived_marked")]
    by_rule = {}
    for r in all_recs:
        by_rule.setdefault(r["rule"], {"applicable": 0, "rejected": 0})
        by_rule[r["rule"]][r["status"]] += 1
    summary = {
        "schema": "rlvr.shorthand-derivation/v1",
        "sources": len(rows),
        "records": len(all_recs),
        "applicable": len(app),
        "rejected": len(rej),
        "by_rule": by_rule,
        "checks": {
            "all_spans_verbatim_in_source": len(bad_span) == 0,
            "all_applicable_marked_as_derived": len(unmarked) == 0,
        },
        "non_goals": ["未调用计算器", "未改动原数值与最终答案",
                      "未另造新例子", "未改 loader", "未借用 v4 身份"],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
