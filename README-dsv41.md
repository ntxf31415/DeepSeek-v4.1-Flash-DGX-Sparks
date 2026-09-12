# dsv41 — DeepSeek-V4.1-Flash on the 4× DGX Spark ring (TP4)

4DGX 环网适配版。上游：`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`
（3×Spark TP3 主线 + TP4 profile 未 boot 验证）；源头 `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000`（MIT）。

部署方案与背景：4DGX-DSV41-Flash-部署方案-20260911

## 补丁清单（相对上游 main@e59e6eb）

| 编号 | 文件 | 改动 |
|---|---|---|
| P1 | start.sh | `HOST` 可配（上游硬编码 0.0.0.0）；.env.tp4 设 127.0.0.1 对齐 loopback 门禁 |
| P2 | start.sh | `NCCL_IB_GID_INDEX_FORCE` 定义时跳过 GID 自动探测（内核 1031 GID 表重排铁律 = -1） |
| P3 | start.sh | `WEIGHTS_MODE=local`：worker 挂本地权重，跳过 NFS share/volume 全链路 |
| P4 | start.sh | `EXTRA_DOCKER_ENV` 通用透传（静态 NCCL 全集，追加在默认值之后可覆盖） |
| P5 | boot.py | served name 逗号别名时 smoke/warmup 取第一个名字 |
| P6 | start.sh | 宿主 ring-only NCCL 2.30.7 + libncclpin v9 的 `LD_PRELOAD` 挂载 + ~/nccl-debug 取证目录 |
| P7 | start.sh | per-rank `NCCL_IB_PEER_HCA`（.env.tp4 的 `PEER_HCA_RANK0..3`） |

## 环网契约（.env.tp4）

- rank 映射：0=.55、1=.56、2=.58、3=.57（`WORKER_IPS` 顺序必须如此，上游按顺序分配 rank）
- GID=-1 / PEER_HCA per-rank / 四口 HCA（含第二张卡 roceP2p1s0f0/f1）/ MERGE_NICS=0 / SUBNET_AWARE_ROUTING=1 —— 全部沿用 v2.1-vl-r1 生产值
- 引擎位 8899 loopback + key YOUR_API_KEY + served name 别名保留 `deepseek-v4-flash-vision-exp`（proxy/门户零改动）
- `--enable-metrics` 必带（RK3588 sparkdash 面板 tok/s 计数器路径）
- 明确不移植：`VLLM_*` 全系、ringonly-v5/`NCCL_RING_MAP=quad`（上游实测不可用）、`PYTORCH_CUDA_ALLOC_CONF=True`（dsv41 会 NaN）

## 首次 boot 保守值

CONTEXT_LENGTH=262144、MAX_TOTAL_TOKENS=2000000、CHUNKED_PREFILL_SIZE=2048。爬坡到 1M/4M 前需 memguard 护长 prompt 实测。
