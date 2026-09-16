#!/bin/bash
# armsuite.sh <label> — one A/B arm measured end to end, identical protocol per arm.
#
# Why a fixed suite: the fleet's boot-to-boot variance is on the order of +/-11%
# (see the A/B/A notes in neko-legends/spark-bench), so one arm's numbers are
# only comparable to another arm's when both were produced exactly the same way.
# This script is that "same way". Order matters:
#   1. greedy fingerprint FIRST -- several DSV4/SM12x changes alter the tokens
#      produced (sglang#39235: the SM120 decode path pads query heads to 64 and
#      dropping the pad changes greedy output). A throughput win that moved the
#      fingerprint is not a win, so we find out before measuring speed.
#   2. pure decode (sparkDash protocol) -- post-TTFT tok/s, TTFT excluded.
#   3. prefill gradient -- throughput plus the built-in needle PASS judgement.
#   4. concurrency + TTFT -- aggregate and per-stream at c1/2/4/8.
#   5. idle memory floor -- MemAvailable, the number that used to wedge the box.
#
# Usage (on the head):  scripts/verify/armsuite.sh a0-baseline
#                       scripts/verify/armsuite.sh a0-baseline --long
# Writes bench-results/arms/<label>/{fingerprint,prose,prefill,conc,mem}.txt
# --long adds stage 6 (~15 min): memguard + 600K/900K needles, the measurement
# that decides anything aimed at long-context transient memory. Left off by
# default so the fast suite stays cheap enough to run on every arm; when an arm
# is compared against another, both must have used the same flag set.
set -uo pipefail

LABEL="${1:?usage: armsuite.sh <label> [--long]}"
LONG=0
[[ "${2:-}" == "--long" ]] && LONG=1
ROOT="$HOME/dsv41-4x-spark"
KEY="$(cat "$ROOT/state-tp4/api-key")"
OUT="$ROOT/bench-results/arms/$LABEL"
mkdir -p "$OUT"
export MODEL=deepseek-v4.1-flash

# Node list and ssh user come from the profile, not from literals: the same
# checkout drives a 3-node and a 4-node fleet, and a hard-coded list goes stale.
NODES="$(sed -n 's/^HEAD_IP=//p; s/^WORKER_IPS=//p' "$ROOT/.env.tp4" | tr -d '"' | tr '\n' ' ')"
[[ -n "${NODES// /}" ]] || NODES="$(hostname -I | awk '{print $1}')"
SSH_USER="$(sed -n 's/^WORKER_USER=//p' "$ROOT/.env.tp4" | tr -d '"' | head -1)"
: "${SSH_USER:=$(id -un)}"

{
  echo "=== arm $LABEL @ $(date '+%F %T %Z') ==="
  echo "--- overlay image: $(docker image inspect dsv41-4x-spark:local --format '{{.Id}}' 2>/dev/null)"
  echo "--- engine env: $(docker exec dsv41-head sh -c 'echo "FLASHMLA_BACKEND=$SGLANG_SM120_FLASHMLA_BACKEND"' 2>/dev/null)"
} | tee "$OUT/meta.txt"

echo "--- [1/5] greedy fingerprint"
URL=http://127.0.0.1:8899/v1/chat/completions API_KEY="$KEY" \
  python3 "$ROOT/scripts/verify/fingerprint.py" 2>&1 | tee "$OUT/fingerprint.txt"

echo "--- [2/5] pure decode (sparkDash protocol, n=9)"
URL=http://127.0.0.1:8899/v1 API_KEY="$KEY" MAX_TOKENS=256 ROUNDS=9 \
  python3 "$ROOT/bench/prose_bench_sparkdash.py" 2>&1 | tee "$OUT/prose.txt"

echo "--- [3/5] prefill gradient 8K/32K/100K"
for T in 8000 32000 100000; do
  docker exec dsv41-head python3 /opt/dsv41/scripts/verify/prefill_distinct.py "$T" 1 2>&1 \
    | tee -a "$OUT/prefill.txt"
done

echo "--- [4/5] concurrency + TTFT (direct 8899)"
python3 "$ROOT/bench/bench_tp.py" --base http://127.0.0.1:8899/v1 --model deepseek-v4.1-flash \
  --key "$KEY" --conc 1,2,4,8 --max-tokens 400 --prompt-type code 2>&1 | tee "$OUT/conc.txt"

echo "--- [5/5] idle memory floor (every node in the profile)"
for ip in $NODES; do
  printf '%s ' "$ip"
  if [ "$ip" = "$(hostname -I | awk '{print $1}')" ]; then
    # The head has no key to ssh to itself; read /proc directly.
    awk '/MemAvailable/{print "MemAvailable=" int($2/1024) "MB"}' /proc/meminfo
  else
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_USER@$ip" \
      'free -m | awk "/Mem:/{print \"MemAvailable=\"\$7\"MB\"}"'
  fi
done | tee "$OUT/mem.txt"

if [[ "$LONG" -eq 1 ]]; then
  echo "--- [6/6] long-context needles under memguard (600K, 900K)"
  nohup python3 "$ROOT/scripts/verify/memguard.py" "$OUT/memguard.log" 1.5 >/dev/null 2>&1 &
  GUARD=$!
  sleep 2
  for spec in "600000 21" "900000 22"; do
    set -- $spec
    docker exec dsv41-head python3 /opt/dsv41/scripts/verify/prefill_distinct.py "$1" "$2" 2>&1 \
      | tee -a "$OUT/long.txt"
  done
  kill "$GUARD" 2>/dev/null
  {
    # memguard lines read `HH:MM:SS <avail> min=<floor>`; take the numeric field
    # after the `=`. Sorting the whole `min=N` token would be lexical, which puts
    # min=10.03 before min=4.00 and reports the wrong floor.
    echo "--- memguard floor over the run:"
    awk -F= '/min=/{print $NF}' "$OUT/memguard.log" 2>/dev/null | sort -n | head -1
    echo "--- ABORT count: $(grep -c ABORT "$OUT/memguard.log" 2>/dev/null || echo 0)"
  } | tee -a "$OUT/long.txt"
fi

echo "=== arm $LABEL done @ $(date '+%F %T %Z') ==="