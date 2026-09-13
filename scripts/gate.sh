#!/bin/bash
# gate.sh — 「容器 Up ≠ 服务可信」。一次判定当前引擎是否可交付。
#
# 用法：
#   gate.sh            快档（~2 分钟）：/health + 算术 + 结构化 + 工具 + corruption + code-gate
#   gate.sh --full     全档（~12 分钟）：快档 + 终止性 + needle 梯度(30K/229K/470K)
#   gate.sh --json     机器可读汇总（最后一行 JSON）
#
# 退出码：0 全过 · 1 有失败
#
# 设计取舍：默认快档要能在一次换班/切换后立刻跑完，覆盖「最常被打断的正确性」
# 而非吞吐；吞吐由 bench/ 下的基准负责。

set -uo pipefail

PORT="${SERVER_PORT:-8899}"
KEY_FILE="${API_KEY_FILE:-$HOME/dsv41-4x-spark/state-tp4/api-key}"
BASE="http://127.0.0.1:${PORT}"
FULL=0
JSON=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    --json) JSON=1 ;;
  esac
done

KEY=""
[[ -f "$KEY_FILE" ]] && KEY="$(cat "$KEY_FILE")"
auth=()
[[ -n "$KEY" ]] && auth=(-H "Authorization: Bearer $KEY")

pass=0; fail=0
declare -a results=()
ok()   { results+=("PASS  $1"); pass=$((pass+1)); echo "[+] $1"; }
bad()  { results+=("FAIL  $1"); fail=$((fail+1)); echo "[x] $1"; }

ask() {  # ask <json-body>
  curl -s --max-time 300 "${auth[@]}" -H 'Content-Type: application/json' \
    -d "$1" "$BASE/v1/chat/completions"
}

# ── 1. 健康 ────────────────────────────────────────────────────────────────
if curl -fsS --max-time 10 "${auth[@]}" "$BASE/health" >/dev/null 2>&1 \
   || curl -fsS --max-time 10 "$BASE/health" >/dev/null 2>&1; then
  ok "engine /health 200"
else
  bad "engine /health 非 200 —— 服务不可交付"
fi

# ── 2. 算术（贪心确定性） ──────────────────────────────────────────────────
r=$(ask '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":16,
  "chat_template_kwargs":{"thinking":false},
  "messages":[{"role":"user","content":"What is 19 + 23? Reply only the number."}]}')
if printf '%s' "$r" | python3 -c 'import json,sys
try:
    sys.exit(0 if json.load(sys.stdin)["choices"][0]["message"]["content"].strip() == "42" else 1)
except Exception:
    sys.exit(1)' 2>/dev/null; then
  ok "算术 19+23=42"
else
  bad "算术失败: ${r:0:120}"
fi

# ── 3. 结构化输出 ──────────────────────────────────────────────────────────
r=$(ask '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":64,
  "chat_template_kwargs":{"thinking":false},
  "response_format":{"type":"json_schema","json_schema":{"name":"a","strict":true,
    "schema":{"type":"object","properties":{"answer":{"type":"integer"}},
              "required":["answer"],"additionalProperties":false}}},
  "messages":[{"role":"user","content":"Return an object whose answer is the integer 42."}]}')
if printf '%s' "$r" | python3 -c 'import json,sys
try:
    c = json.load(sys.stdin)["choices"][0]["message"]["content"]
    sys.exit(0 if json.loads(c) == {"answer": 42} else 1)
except Exception:
    sys.exit(1)' 2>/dev/null; then
  ok "JSON schema 结构化输出"
else
  bad "结构化失败: ${r:0:120}"
fi

# ── 4. 工具调用 ────────────────────────────────────────────────────────────
r=$(ask '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":96,
  "chat_template_kwargs":{"thinking":false},
  "tools":[{"type":"function","function":{"name":"lookup","description":"get value",
    "parameters":{"type":"object","properties":{"key":{"type":"string"}},
                  "required":["key"],"additionalProperties":false}}}],
  "messages":[{"role":"user","content":"Use lookup to get the value for key alpha. Do not guess."}]}')
if printf '%s' "$r" | python3 -c 'import json,sys
try:
    t = json.load(sys.stdin)["choices"][0]["message"].get("tool_calls") or []
    sys.exit(0 if (len(t) == 1 and t[0]["function"]["name"] == "lookup"
                   and json.loads(t[0]["function"]["arguments"]) == {"key": "alpha"}) else 1)
except Exception:
    sys.exit(1)' 2>/dev/null; then
  ok "工具调用（tool_calls，参数正确）"
else
  bad "工具调用失败: ${r:0:160}"
fi

# ── 5/6. corruption + code-gate（复用 gates_suite，容器内跑） ───────────────
gs=""
for c in "/state/gates_suite.py" "$HOME/dsv41-4x-spark/state-tp4/gates_suite.py"; do
  [[ -f "$c" ]] && gs="$c" && break
done
if docker exec dsv41-head test -f /state/gates_suite.py 2>/dev/null; then
  out=$(docker exec -w /state dsv41-head python3 /state/gates_suite.py --corruption --code 2>&1)
  if echo "$out" | grep -q 'U+FFFD=0' && ! echo "$out" | grep -q 'FAIL'; then
    ok "corruption probe 0/0/0"
  else
    bad "corruption 异常: $(echo "$out" | grep -m1 'U+FFFD' || echo 无输出)"
  fi
  if echo "$out" | grep -q 'code-gate total: 12/12'; then
    ok "code-gate 12/12"
  else
    bad "code-gate: $(echo "$out" | grep -m1 'code-gate total' || echo 未完成)"
  fi
else
  echo "[!] gates_suite 不在容器内，跳过 corruption/code-gate"
fi

# ── 全档：终止性 + needle 梯度 ─────────────────────────────────────────────
if [[ "$FULL" -eq 1 ]]; then
  out=$(docker exec -w /state dsv41-head python3 /state/gates_suite.py --termination 2>&1)
  if [[ "$(echo "$out" | grep -c '18/18 stop')" -eq 2 ]]; then ok "终止性 18/18 · 18/18"
  else bad "终止性: $(echo "$out" | tr '\n' ' ')"; fi

  out=$(docker exec -w /state dsv41-head python3 /state/gates_suite.py --needle 2>&1)
  if [[ "$(echo "$out" | grep -c ' PASS')" -eq 3 ]]; then ok "needle 30K/229K/470K 全 PASS"
  else bad "needle: $(echo "$out" | tr '\n' ' ' | cut -c1-160)"; fi
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────
echo
if [[ "$JSON" -eq 1 ]]; then
  printf '{"pass":%d,"fail":%d,"full":%s,"results":[' "$pass" "$fail" "$([[ $FULL -eq 1 ]] && echo true || echo false)"
  for i in "${!results[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    printf '"%s"' "${results[$i]}"
  done
  printf ']}\n'
fi
if [[ "$fail" -eq 0 ]]; then
  echo "[+] GATE PASSED（$pass 项）—— 引擎可信"
  exit 0
else
  echo "[x] GATE FAILED（$pass 过 / $fail 败）—— 不要交付"
  exit 1
fi
