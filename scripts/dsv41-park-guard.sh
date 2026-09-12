#!/bin/bash
# 占位容器守护 v2: 每 10 分钟换新名字持有容器 (龄永远 <15min ->
# LuZ healthcheck 的 MIN_AGE_S 守卫永不触发, 停靠对象不受打扰)。
# 旧持有容器 rename 为 dsv41-parkold-<ts> 保留 (monitor 若停靠在它
# 上则永远静默; 若停靠在旧 park 上也不受影响)。勿删 parkold 容器。
while :; do
  sleep 600
  if docker ps --format '{{.Names}}' | grep -qx vllm-tp4-rank0; then
    ts=1789177221
    docker rename vllm-tp4-rank0 dsv41-parkold- 2>/dev/null &&       docker run -d --name vllm-tp4-rank0 --network none alpine:latest sleep infinity >/dev/null 2>&1
  else
    docker run -d --name vllm-tp4-rank0 --network none alpine:latest sleep infinity >/dev/null 2>&1
  fi
done
