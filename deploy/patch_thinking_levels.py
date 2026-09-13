#!/usr/bin/env python3
"""给值班员与研究员都装上「可动态切换的思考档位」。

实测得到的事实（不是推测）：
  - MiniCPM5-2B 模板只有 `enable_thinking` 布尔；但 llama.cpp **认顶层
    `thinking_budget_tokens`**：设 8 时 reasoning 从 389 字符降到 11。所以
    值班员能有真档位，不是只有开关。
  - Qwen3.8-27B 模板有 `enable_thinking` + `reasoning_effort`，
    合法值 xhigh / medium / low（默认 xhigh）。

DSH 机制：
  - provider 层 `thinkingBudgets: {minimal,low,medium,high}` -> 数字
  - provider 层 `compat.chatTemplateKwargs.<kwarg>.$var` ->
      thinking.enabled(布尔) / thinking.effort(该档的 wire 值) / thinking.budget
  - provider 层 `compat.supportsThinkingTokenBudget` + `thinkingTokenBudgetField`
  - model 层 `reasoningEfforts: {<档>: <wire 值>}`（只有 off 可留空）
"""
import io

P = "/root/.dsh/cordis.patch.yml"
s = io.open(P, encoding="utf-8").read()

GUARD_OLD = '''        compat:
          thinkingFormat: chat-template
          chatTemplateKwargs:
            enable_thinking:
              $var: thinking.enabled
              # llama.cpp 默认开思考，所以"关"必须显式发 false，不能靠省略。
              omitWhenOff: false
        defaultContextWindow: 131072'''

GUARD_NEW = '''        compat:
          thinkingFormat: chat-template
          chatTemplateKwargs:
            enable_thinking:
              $var: thinking.enabled
              # llama.cpp 默认开思考，所以"关"必须显式发 false，不能靠省略。
              omitWhenOff: false
          # 值班员的模板只有 enable_thinking 布尔，但 llama.cpp 支持按请求传
          # 思考 token 预算 —— 实测 thinking_budget_tokens=8 能把 reasoning
          # 从 389 字符压到 11，所以档位是真的，不是只有开关。
          supportsThinkingTokenBudget: true
          thinkingTokenBudgetField: thinking_budget_tokens
        # 各档的思考预算（token）。-1 = 不限制（llama.cpp 语义）。
        thinkingBudgets:
          minimal: 32
          low: 128
          medium: 512
          high: -1
        defaultContextWindow: 131072'''

assert GUARD_OLD in s, "guard compat 段未找到"
s = s.replace(GUARD_OLD, GUARD_NEW)

GUARD_MODEL_OLD = '''            # 可调档位 —— 声明后 UI 会给出选择器。
            # MiniCPM5-2B 的模板只有 enable_thinking 这一个布尔开关，
            # 没有真正的"思考预算"，因此只提供 off / medium 两档：
            # 任何非 off 档都等价于「开」。非 off 档的线值此处不被使用
            # （只有 thinking.enabled 被读），但校验要求它非空。
            reasoningEfforts:
              off:
              medium: medium'''

GUARD_MODEL_NEW = '''            # 可调档位 —— 声明后 UI 会给出选择器。
            # off  -> enable_thinking=false（模板会写 <think></think> 空思考）
            # 其余 -> enable_thinking=true，并用 thinkingBudgets 给预算上限
            # 实测：值班员关思考确实更笨，所以默认建议 medium 以上。
            reasoningEfforts:
              off:
              minimal: minimal
              low: low
              medium: medium
              high: high'''

assert GUARD_MODEL_OLD in s, "guard reasoningEfforts 段未找到"
s = s.replace(GUARD_MODEL_OLD, GUARD_MODEL_NEW)

RES_OLD = '''        apiKeyEnv: LOCAL_LLM_KEY
        defaultContextWindow: 262144
        defaultMaxTokens: 8192
        models:
          - id: qwen3-8-27b
            name: Qwen3.8-27B-AWQ（研究员，GPU 按需 :18001）
            contextWindow: 262144
            reasoningEfforts: false'''

RES_NEW = '''        apiKeyEnv: LOCAL_LLM_KEY
        # 研究员的模板支持 enable_thinking + reasoning_effort
        # （合法值 xhigh / medium / low，模型默认 xhigh）。
        compat:
          thinkingFormat: chat-template
          chatTemplateKwargs:
            enable_thinking:
              $var: thinking.enabled
              omitWhenOff: false
            reasoning_effort:
              $var: thinking.effort
        defaultContextWindow: 262144
        defaultMaxTokens: 8192
        models:
          - id: qwen3-8-27b
            name: Qwen3.8-27B-AWQ（研究员，GPU 按需 :18001）
            contextWindow: 262144
            # 模板的 reasoning_effort 只认这三个值（外加 off=关思考）。
            # 注：这是「思考努力档」，不是 token 预算 —— Qwen 模板没有预算变量，
            # 所以不要在 provider 层配 thinkingBudgets（会被校验拒绝或静默无效）。
            reasoningEfforts:
              off:
              low: low
              medium: medium
              xhigh: xhigh'''

assert RES_OLD in s, "researcher 段未找到"
s = s.replace(RES_OLD, RES_NEW)

io.open(P, "w", encoding="utf-8").write(s)
print("patched: guard(5 levels + budget) / researcher(4 levels + effort)")
