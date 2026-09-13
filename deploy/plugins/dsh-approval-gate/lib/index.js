/**
 * dsh-approval-gate —— 安全员
 *
 * 挂在 DSH 的 `approval/request` waterfall 上。核心原则：
 *
 *   ★ 模型永远不能放松判定，只能收紧。
 *     即使模型被忽悠，最坏结果也只是「多问人类一次」，而不是「放行一次危险写入」。
 *
 * 判定顺序（`08-验收与交付计划.md` §2.5.2 / §2.5.3）：
 *
 *   1. 硬拒绝表（DENY）      —— 确定性，毫秒级，直接 rejected
 *   2. 白名单（ALLOW）       —— 确定性，直接 allowed-once
 *   3. 学习到的中性签名      —— 同一签名被人类放行 3 次后自动放行
 *   4. 模型语义复核（仅灰区）—— MiniCPM5-2B 二分类；suspicious 升格为「问人类」，
 *                              normal 不改变确定性层的判定（即不放松）
 *   5. 兜底                  —— next()，交给人类 UI answerer
 *
 * 失败一律 fail-closed：任何内部异常都返回 next()（问人类），绝不静默放行。
 *
 * 审计：每次裁决追加一行到 evidence/approval-gate.jsonl（含工具名、理由、判定、依据）。
 */

import fs from 'node:fs';
import path from 'node:path';

const WORKSPACE = '/root/dsh-harness';
const EVIDENCE = path.join(WORKSPACE, 'evidence');
const AUDIT = path.join(EVIDENCE, 'approval-gate.jsonl');
const STATE_DIR = '/root/.dsh/auto-approve';
const ALLOWLIST = path.join(STATE_DIR, 'allowlist.json');

/** 判定词汇：allowed-once 是唯一的授权值（DSH 原生约定）。 */
const ALLOW = 'allowed-once';
const REJECT = 'rejected';
const DEFER = Symbol('defer'); // 交给人类

/**
 * 硬拒绝表：这些命中就 rejected，不受任何后续层影响。
 * 对应规格 §2.5.3 的「拒绝」行。
 */
const DENY_RULES = [
  { re: /(^|[/\s"'])\.venv([/\s"']|$)/, why: '禁止写入 lab 的 .venv（editable .pth 会让改动静默失效）' },
  { re: /(^|[/\s"'])models?\//, why: '禁止写入 models 目录' },
  { re: /(^|[/\s"'])data\//, why: '禁止写入 data 目录' },
  { re: /\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(\/|~|\$HOME)(\s|$)/, why: '危险的递归删除（根/家目录）' },
  { re: /\bmkfs(\.\w+)?\b/, why: '磁盘格式化' },
  { re: /\bdd\b[^\n]*\bof=\/dev\//, why: '裸设备写入' },
  { re: /\bgit\s+push\b[^\n]*--force/, why: 'force push 会覆盖远端历史' },
  { re: /\bgit\s+worktree\b/, why: '禁用 git worktree（lab venv 的 editable .pth 不生效）' },
  { re: /\bchmod\s+-R\s+777\s+\//, why: '危险的全局权限修改' },
];

/**
 * 白名单：harness 自身文件与运行时目录，直接放行。
 * 对应规格 §2.5.3 的「写 workspace 内的 harness 自身文件 → 通过」。
 */
const ALLOW_RULES = [
  // 注意：不能锚 ^ —— 审批理由通常是「动词 + 路径」，路径不在开头。
  { re: /(^|[\s"'])@?\/root\/dsh-harness\//, why: 'harness 自身文件（队列/配置/日志，事件溯源可重建）' },
  { re: /\/dev\/shm\/(models|hf|pipcache)\//, why: 'tmpfs 临时区' },
];

/**
 * 需要人类批准：命中即跳过自动放行，直接 next()。
 * 对应规格 §2.5.3 的「需要人类批准」行。
 */
const ASK_RULES = [
  { re: /\/(tests|configs)\/[^\s"']*\.(json|py|sh|ya?ml)/, why: '测试或配置阈值改动（防「改测试让它过」）' },
  { re: /\bgit\s+(checkout|restore)\s+--?\s/, why: '可能丢弃未提交改动' },
  { re: /\bgit\s+reset\s+--hard/, why: '可能丢弃未提交改动' },
  { re: /\bgit\s+clean\b/, why: '删除未跟踪文件（git 救不回）' },
  { re: /\b(shred|truncate)\b/, why: '不可逆的内容销毁' },
  { re: /\b(curl|wget)\b[^\n]*\|\s*(ba)?sh/, why: '下载即执行（供应链风险）' },
];

/** 从审批事件里取出可判定的文本：工具名 + 理由 + 调用参数（若理由里带了）。 */
function subject(req) {
  return [req?.toolName ?? '', req?.reason ?? ''].join(' ').trim();
}

function firstMatch(rules, text) {
  for (const r of rules) if (r.re.test(text)) return r;
  return null;
}

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return fallback; }
}

function writeJson(file, value) {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(value, null, 2));
  } catch { /* 状态持久化失败不应影响裁决 */ }
}

/** 学习到的中性签名：{ [sig]: 人类已放行次数 }。 */
function loadAllowlist() {
  const d = readJson(ALLOWLIST, { neutral: {} });
  if (!d.neutral || typeof d.neutral !== 'object') d.neutral = {};
  return d;
}

/** 签名：同一类请求的稳定指纹（工具名 + 首个路径/命令片段）。 */
function signature(req) {
  const t = subject(req);
  const m = t.match(/(\/[\w./-]{3,}|\b[a-z][\w-]{2,}\b)/i);
  return `${req?.toolName ?? 'unknown'}::${m ? m[1] : t.slice(0, 48)}`;
}

function audit(entry) {
  try {
    fs.mkdirSync(EVIDENCE, { recursive: true });
    fs.appendFileSync(AUDIT, JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n');
  } catch { /* 审计写失败不应阻塞审批，但也不放行 */ }
}

/** 灰区模型复核：只做 normal|suspicious 二分类。任何异常都视为「未裁决」。 */
async function modelReview(req) {
  const base = process.env.GUARD_BASE_URL || 'http://127.0.0.1:18099/v1';
  const model = process.env.GUARD_MODEL || 'minicpm5-2b';
  const prompt =
    '你是审批安全员。只判断这次操作是否可疑，输出严格的 JSON：{"verdict":"normal|suspicious","reason":"..."}。\n' +
    `工具：${req?.toolName ?? '未知'}\n说明：${(req?.reason ?? '无').slice(0, 1500)}\n` +
    '可疑 = 可能不可逆、可能越界、可能改动测试或配置、可能触碰凭据。谨慎优先。';
  const body = JSON.stringify({
    model,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 96,
    temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
  });
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 12000);
  try {
    const r = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: ctl.signal,
    });
    const d = await r.json();
    const txt = d?.choices?.[0]?.message?.content ?? '';
    const m = txt.match(/\{[\s\S]*\}/);
    if (!m) return { verdict: 'unavailable' };
    const j = JSON.parse(m[0]);
    const v = String(j.verdict ?? '').toLowerCase();
    if (v !== 'normal' && v !== 'suspicious') return { verdict: 'unavailable' };
    return { verdict: v, reason: String(j.reason ?? '').slice(0, 200) };
  } catch {
    return { verdict: 'unavailable' };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 核心裁决。返回 { outcome | DEFER, layer, why }。
 * 注意：模型层只能把 DEFER 维持或升格，不能把 DENY 翻成 ALLOW。
 */
async function decide(req, cfg) {
  const text = subject(req);

  const deny = firstMatch(DENY_RULES, text);
  if (deny) return { outcome: REJECT, layer: 'deny', why: deny.why };

  const ask = firstMatch(ASK_RULES, text);
  if (ask) return { outcome: DEFER, layer: 'ask', why: ask.why };

  const allow = firstMatch(ALLOW_RULES, text);
  if (allow) {
    // 白名单也要过一遍模型（模型只能收紧）：可疑则升格为问人类
    if (cfg.modelReview) {
      const mv = await modelReview(req);
      if (mv.verdict === 'suspicious') {
        return { outcome: DEFER, layer: 'model-tighten', why: `模型判可疑：${mv.reason || '未给理由'}` };
      }
    }
    return { outcome: ALLOW, layer: 'allow', why: allow.why };
  }

  // 学习到的中性签名
  const sig = signature(req);
  const wl = loadAllowlist();
  if ((wl.neutral[sig] ?? 0) >= cfg.neutralThreshold) {
    return { outcome: ALLOW, layer: 'learned', why: `同一签名已被人类放行 ${wl.neutral[sig]} 次` };
  }

  // 灰区：模型复核
  if (cfg.modelReview) {
    const mv = await modelReview(req);
    if (mv.verdict === 'suspicious') {
      return { outcome: DEFER, layer: 'model-tighten', why: `模型判可疑：${mv.reason || '未给理由'}` };
    }
    if (mv.verdict === 'normal') {
      // 模型说 normal —— 不改变确定性层判定，即仍然问人类
      return { outcome: DEFER, layer: 'gray-model-normal', why: '灰区，模型判正常，仍需人类确认' };
    }
  }
  return { outcome: DEFER, layer: 'gray', why: '无明显策略命中' };
}

export default function apply(ctx, rawCfg = {}) {
  const cfg = {
    modelReview: rawCfg.modelReview !== false,
    neutralThreshold: Number(rawCfg.neutralThreshold ?? 3),
  };

  // 记下人类最终放行的中性签名，用于学习
  const pending = new Map();

  ctx.on('approval/request', async (req, next) => {
    let d;
    try {
      d = await decide(req, cfg);
    } catch (err) {
      // fail-closed：内部异常一律交人类
      audit({ tool: req?.toolName, reason: req?.reason, outcome: 'defer-on-error', err: String(err) });
      return next();
    }

    if (d.outcome === DEFER) {
      const sig = signature(req);
      pending.set(sig, (pending.get(sig) ?? 0) + 1);
      const outcome = await next();
      // 人类放行了 → 累积该签名
      if (outcome === ALLOW) {
        const wl = loadAllowlist();
        wl.neutral[sig] = (wl.neutral[sig] ?? 0) + 1;
        writeJson(ALLOWLIST, wl);
      }
      audit({ tool: req?.toolName, reason: req?.reason, layer: d.layer, why: d.why,
              outcome: `deferred->${outcome}`, sig });
      return outcome;
    }

    audit({ tool: req?.toolName, reason: req?.reason, layer: d.layer, why: d.why, outcome: d.outcome });
    return d.outcome;
  }, true /* prepend：安全员先判，人类 answerer 仍是 terminal */);

  // 挂载证明：写审计文件（比日志可靠 —— DSH 的 logger 未必有 info 级）。
  audit({
    event: 'mounted',
    version: 'v3',
    pipeline: 'DENY->allowlist->learned->model-tighten->human',
    modelReview: cfg.modelReview,
    neutralThreshold: cfg.neutralThreshold,
    audit: AUDIT,
    allowlist: ALLOWLIST,
  });

  const banner =
    `[dsh-approval-gate] v3 mounted: DENY->allowlist->learned(${cfg.neutralThreshold})` +
    `->modelTighten->human; modelReview=${cfg.modelReview}; audit=${AUDIT}`;
  const log = ctx.logger ?? {};
  for (const level of ['info', 'warn', 'debug']) {
    if (typeof log[level] === 'function') { log[level](banner); break; }
  }
}

export const name = 'dsh-approval-gate';
export const inject = [];
