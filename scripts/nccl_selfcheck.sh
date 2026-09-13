#!/bin/bash
# nccl_selfcheck.sh — 校验 NCCL 初始化是否符合无交换机环网预期
#
# 检查项（任一失败即非零退出）：
#   1. RING-ONLY 补丁生效（日志出现 "RING-ONLY v"）
#   2. 算法矩阵中 Tree 全部为 0、Ring 全部为 1（Tree 未参与，Ring 强制）
#   3. 通道数 > 0
#   4. 期望的 RoCE 设备全部被 NCCL 采用
# 仅记录（不判失败）：GDR 状态、协议集。
#
# 用法：nccl_selfcheck.sh [debug_dir] [expected_hca_count]
#   默认 debug_dir=$HOME/nccl-debug, expected_hca_count=4

set -uo pipefail

DBG_DIR="${1:-$HOME/nccl-debug}"
EXPECT_HCA="${2:-4}"
fail=0

log=$(ls -t "$DBG_DIR"/*.log 2>/dev/null | head -1)
if [[ -z "${log:-}" ]]; then
  echo "[x] 未找到 NCCL 调试日志（$DBG_DIR/*.log）——确认 NCCL_DEBUG_FILE 已设且服务已启动"
  exit 1
fi
echo "[i] 检查日志: $log"

# 1. RING-ONLY 补丁
if grep -q 'RING-ONLY v' "$log"; then
  ver=$(grep -oE 'RING-ONLY v[0-9a-zA-Z._-]+' "$log" | head -1)
  echo "[+] $ver 补丁生效"
else
  echo "[x] 未发现 RING-ONLY 补丁标记——ring-only 未生效！"
  fail=1
fi

# 2. 算法矩阵：Tree 全 0 / Ring 全 1
#    表行形如： "    AllReduce |      1         1         1   |      0       1  ..."
matrix_rows=$(grep -cE '^\s+(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce)\s+\|' "$log" || true)
if [[ "${matrix_rows:-0}" -ge 5 ]]; then
  bad=$(grep -E '^\s+(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce)\s+\|' "$log" \
        | awk -F'|' '{n=split($3,a," "); tree=a[1]; ring=a[2];
                      if (tree+0 != 0 || ring+0 != 1) print "  BAD: " $0}')
  if [[ -z "$bad" ]]; then
    echo "[+] 算法矩阵 OK：Tree=0 / Ring=1（$matrix_rows 个集合函数全部强制 Ring）"
  else
    echo "[x] 算法矩阵异常：Tree 未禁用或 Ring 未启用"
    echo "$bad"
    fail=1
  fi
else
  echo "[!] 日志中未找到算法矩阵（NCCL_DEBUG 可能不是 INFO）——跳过"
fi

# 3. 通道数
chans=$(grep -oE '[0-9]+ coll channels' "$log" | head -1 | grep -oE '^[0-9]+' || echo 0)
if [[ "${chans:-0}" -gt 0 ]]; then
  echo "[+] 集合通道数: $chans"
else
  echo "[x] 未发现集合通道"
  fail=1
fi

# 4. RoCE 设备采用数
nics=$(grep 'NET/IB : Using' "$log" | head -1 | grep -oE '\[[0-9]+\]ro' | wc -l | tr -d ' ')
if [[ "${nics:-0}" -ge "$EXPECT_HCA" ]]; then
  echo "[+] NCCL 采用 RoCE 设备数: $nics（期望 >= $EXPECT_HCA）"
else
  echo "[x] NCCL 仅采用 $nics 个 RoCE 设备（期望 $EXPECT_HCA）——检查 NCCL_IB_HCA"
  fail=1
fi

# 5. 仅记录：GDR 状态（GB10 上预期为 0，全走 host-staging）
gdr=$(grep -oE 'cuMemGdrSupport [0-9]+' "$log" | head -1 || true)
echo "[i] ${gdr:-cuMemGdrSupport 未知}（GB10 预期 0 = 无 GDR，CPU-path）"

# 6. 补丁谱系识别（仅记录）：区分两套已知的 switchless 补丁路线
#    - SparkRing/PR#3：源码加 NCCL_PARAM(SwitchlessRingOnly,…)，运行期打印
#      "Tree/PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY"
#    - LuZ 路线（本栈）：环邻过滤 + 算法矩阵 Tree=0（Tree 仍建连但永不被选中）
lib="${NCCL_HOST_DIR:-/opt/nccl-ringonly}/libnccl.so.2.30.7"
if [[ -f "$lib" ]]; then
  # grep -c prints "0" and exits 1 on no match, so do not append another 0.
  sw=$(strings "$lib" 2>/dev/null | grep -c SWITCHLESS_RING_ONLY)
  sk=$(strings "$lib" 2>/dev/null | grep -c SKIP_TREE_CONNECT)
  dis=$(grep -icE 'transport setup disabled' "$log")
  sw=${sw:-0}; sk=${sk:-0}; dis=${dis:-0}
  if [[ "$sw" -gt 0 ]]; then
    echo "[i] NCCL 补丁谱系: SparkRing/PR#3 路线（SWITCHLESS_RING_ONLY>0, skip-tree=${sk}）"
  elif [[ "$dis" -eq 0 ]]; then
    echo "[i] NCCL 补丁谱系: LuZ 路线（无 SWITCHLESS_RING_ONLY；靠算法矩阵 Tree=0 达成 ring-only）"
  else
    echo "[i] NCCL 补丁谱系: 未知/混合"
  fi
else
  echo "[i] 未找到 $lib，跳过谱系识别"
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "[+] NCCL 自检通过"
else
  echo "[x] NCCL 自检发现问题（见上）"
fi
exit "$fail"
