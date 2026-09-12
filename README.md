# DeepSeek-V4.1-Flash (SGLang) on 4× DGX Spark, TP4, Switchless Ring

Production recipe for serving **deepseek-ai/DeepSeek-V4.1-Flash** (552B MoE, 8B/16B
active, MXFP4 experts, 1M context, DSpark speculative decoding) with **SGLang TP4**
across **4× NVIDIA DGX Spark (GB10)** connected as a **switchless RoCE ring** (no
400G switch).

> 中文说明：[README.zh-CN.md](README.zh-CN.md) · 部署方案与基准对比：[docs/](docs/)

**Measured on 4× DGX Spark (GB10, sm_121a, switchless ring), 1M ctx / 4M KV pool:**

| benchmark | value |
|---|---|
| decode peak / mean (code, temp 0) | **89.6 / 72.1** tok/s |
| prose OFF / ON | 30.8 / 33.6 tok/s |
| prefill 8K / 32K / 100K | **2814 / 3149 / 3119** t/s |
| aggregate c1 / c4 / c8 / c12 | 74 / 210 / 264 / 271 tok/s |
| 470K needle / 900K cold prefill | ✅ / 1271 t/s (738 s) |
| quality gates | needle 30K-470K ✅ · corruption 0/0/0 · termination 18/18+18/18 · code-gate 12/12 · GSM8K n=50 **1.00** |
| DSpark acceptance | 3.95 tok/step (code) → 6.0 (math, saturated) |
| cold start | ~9 min, **no cold-start penalty** |

Full six-stack comparison incl. LuZ / Vision-Exp / GLM: [docs/4DGX-dsv41-基准测试-横向对比-20260912.md](docs/4DGX-dsv41-基准测试-横向对比-20260912.md).

---

> **Sister projects:** [DeepSeek-V4-Flash-Vision-Exp TP4 switchless-ring](https://github.com/ntxf31415/deepseek-v4-vision-exp-dgxspark-tp4-switchless-ring) (vLLM, the same ring base) · [GLM-5.3-Flash NVFP4 TP4 switchless-ring](https://github.com/ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless) (companion recipe).

## What this is

A **ring adaptation + operations layer** on top of the upstream SGLang recipe.
The repo ships launcher scripts, SGLang monkey-patches (Engram NVMe row store,
MXFP8 b12x, prefill empty-cache), a self-heal monitor, and the benchmark gate
suite. **No weights, no images, no NCCL binaries.**

| Component | Origin | License |
|---|---|---|
| SGLang serving recipe (boot, adapters, Engram row store, DSpark setup) | [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | AGPL-3.0-or-later |
| Recipe lineage / benchmark methodology | [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) | MIT |
| Model weights | `deepseek-ai/DeepSeek-V4.1-Flash` (Hugging Face) | see model card |

## Ring adaptation delta (vs upstream TP4 profile)

- Ring-only NCCL 2.30.7 + libncclpin core-pinning shim via `LD_PRELOAD`;
  `NCCL_IB_GID_INDEX=-1` iron rule; per-rank `PEER_HCA` for the 4-edge wiring
- Node-local weights (no NFS), loopback engine behind a concurrency proxy,
  multi-alias served names (old served name kept for zero-touch consumers)
- Static verify mode + profiled SPS table (upstream compact mode trips an
  engram target-verify assertion on concurrent short requests)
- Local packed Engram shards (~47 GiB/node), weight load 49 s → 24 s
- Full patch list: [docs/4DGX-DSV41-Flash-部署方案-20260911.md](docs/4DGX-DSV41-Flash-部署方案-20260911.md) §三

## Repo contents

- `start.sh / start-tp4.sh / stop.sh / boot.py` — serving orchestration, pinned checkpoint boot, smoke + warm-up
- `adapter/` — SGLang patches (Engram row store C++, MXFP8 backend, prefill cache hook)
- `scripts/` — SSH helper, verify/ probe kit, self-heal monitor + systemd unit template
- `bench/` — gate suite (needle / corruption / termination / code-gate), vision gate, event-timeline matrix + common-window analysis, prose, GSM8K spot
- `docs/` — deployment plan + benchmark comparison (sanitized export)

## Sanitization

Internal IPs/hostnames are replaced with placeholders and API keys are removed
(`YOUR_API_KEY`); site `.env.tp4` is excluded (`.gitignore`). This fork keeps
the upstream `main` branch untouched — the adaptation lives on `4dgx-ring`.
