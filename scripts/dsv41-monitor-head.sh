#!/bin/bash
# dsv41 head(rank0) 自愈 monitor — 对齐 vllm-tp4-head 链的职责
# 检测: 容器 healthy 且 8899 /health=200; 连续 3 次失败 -> 全链重建 (serve 流程自带 worker-first)
# 维护标志: state-tp4/maintenance.flag 存在时挂起监控 (不重建)
set -u
ROOT=/home/spark/dsv41-4x-spark
NAME=dsv41-head
URL=http://127.0.0.1:8899/health
FAILS=0
MAINT_FLAG="$ROOT/state-tp4/maintenance.flag"

while :; do
  if [ -f "$MAINT_FLAG" ]; then
    sleep 300; continue
  fi
  ok=1
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    h=$(docker inspect -f '{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo starting)
    code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || echo 000)
    [ "$h" = healthy ] && [ "$code" = 200 ] || ok=0
  else
    ok=0
  fi
  if [ "$ok" = 1 ]; then FAILS=0; sleep 60; continue; fi
  FAILS=$((FAILS+1))
  if [ "$FAILS" -lt 3 ]; then sleep 60; continue; fi
  echo "[dsv41-monitor] $(date) rebuild: container=$(docker ps -q --filter name=$NAME | wc -l) /health=$(curl -s -m 8 -o /dev/null -w '%{http_code}' $URL 2>/dev/null)" >> "$ROOT/logs-tp4/monitor.log"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  for h in 192.168.1.56 192.168.1.58 192.168.1.57; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 spark@$h 'docker rm -f dsv41-worker >/dev/null 2>&1' 2>/dev/null || true
  done
  cd "$ROOT" && ENV_FILE=.env.tp4 ./start-tp4.sh serve >> "$ROOT/logs-tp4/monitor.log" 2>&1 || true
  FAILS=0
  sleep 60
done
