# DeepSeek-V4.1-Flash (SGLang) on 4× DGX Spark, TP4, Switchless Ring

Production recipe for serving **deepseek-ai/DeepSeek-V4.1-Flash** (552B MoE, 8B/16B
active, MXFP4 experts, 1M context, DSpark speculative decoding) with **SGLang TP4**
across **4× NVIDIA DGX Spark (GB10)** connected as a **switchless RoCE ring** (no
400G switch).

> 中文说明：[README.zh-CN.md](README.zh-CN.md) · 部署方案与基准对比：[docs/](docs/)

**Measured on 4× DGX Spark (GB10, sm_121a, switchless ring), 1M ctx / 5M KV pool.**
Thinking mode is given as `OFF · ON`; the build these numbers came from is pinned in
[BUILD-IDENTITY.md](BUILD-IDENTITY.md).

| benchmark | value |
|---|---|
| decode peak / mean (code, temp 0) | **100.3 / 81.4** · 99.8 / 81.2 tok/s |
| prose OFF · ON | **33.3 · 36.6** tok/s |
| prefill 8K / 32K / 100K | **3102 / 3443 / 3253** t/s |
| aggregate c1 / c4 / c8 / c12 | 82 / 223 / 295 / **398** tok/s (ON: 83 / 225 / 298 / 403) |
| quality gates | needle 30K-470K ✅ · corruption 0/0/0 · termination 18/18+18/18 · code-gate 12/12 · GSM8K n=50 **1.00** |
| DSpark acceptance | 3.98 tok/step, rate 0.595 (6.0 saturated on math) |
| cold start | ~9 min, **no cold-start penalty** |

**Long context** (cold prefill, needle-checked):

| depth | result |
|---|---|
| 470K | ✅ 211.7 s · 2144 t/s |
| 600K | ✅ 341.7 s · MemAvailable floor 4.92 GB |
| **900K** | ⚠️ **fails** — the Engram row cache and the shared-expert pad buffer cost ~2 GB of deep-context headroom. 600K and below are unaffected. |

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
| Host ring-only NCCL 2.30.7 build + `libncclpin` core-pinning shim (host-side, not shipped) | [luxingcom/aicad-nccl-optimization](https://github.com/luxingcom/aicad-nccl-optimization) (LuZ lineage) | **no license declared** |
| Model weights | `deepseek-ai/DeepSeek-V4.1-Flash` (Hugging Face) | see model card |

## Ring adaptation delta (vs upstream TP4 profile)

**Transport / topology**

- Ring-only NCCL 2.30.7 + libncclpin core-pinning shim via `LD_PRELOAD`;
  `NCCL_IB_GID_INDEX=-1` iron rule; per-rank `PEER_HCA` for the 4-edge wiring
  (`./start-tp4.sh ncclcheck` verifies the ring-only path came up). The library is
  the **LuZ lineage** build (ring-only via NCCL's algorithm matrix, `Tree=0 / Ring=1`),
  *not* a SparkRing patched library — see [BUILD-IDENTITY.md](BUILD-IDENTITY.md)
- Node-local weights (no NFS), loopback engine behind a concurrency proxy,
  multi-alias served names (old served name kept for zero-touch consumers)

**Tuned for the ring** (each A/B'd in isolation; `[measured]` deltas in the config header)

| setting | why |
|---|---|
| `EP_SIZE=2` (was 4) | kills the expert-parallel straggler; long-context prefill 595 s → 345 s at 500K |
| `MAX_RUNNING_REQUESTS=12` (was 8) | with `--min-free-slots-delay 1` the 12th slot actually runs: c12 271 → 398 |
| `DSV41_CACHE_GIB=1` / 16-way | Engram row cache: hit rate 0 → 99.1 %, c12 +6 %, prefill 100K +10.5 % |
| `DSV41_SHARED_PAD_K=1` | upstream PR #17: keeps the shared expert's K=576 shape eligible for b12x (bit-identical) |
| static verify mode | upstream compact/ragged mode trips an engram target-verify assertion on V4.1 (sgl-project/sglang#39173) |

Tried and reverted: `--enable-deepseek-v4-fp4-indexer` costs ~11 % on 500K cold
prefill and 6.4 GB of unified memory for no c12 gain. Single regression kept:
c6 260 → 236 (an EP2 side effect; c8/c12 rise far more).

## Repo contents

- `start.sh / start-tp4.sh / stop.sh / boot.py` — serving orchestration, pinned checkpoint boot, smoke + warm-up
- `adapter/` — SGLang patches (Engram row store C++, MXFP8 backend, shared-expert K pad, prefill cache hook)
- `scripts/` — SSH helper, verify/ probe kit, self-heal monitor + systemd unit, `gate.sh`, `nccl_selfcheck.sh`
- `bench/` — gate suite (needle / corruption / termination / code-gate), vision gate, event-timeline matrix + common-window analysis, prose, GSM8K spot, third-party-shaped sweep
- `.env.tp4.ring.example` — the configuration this repo actually runs (sanitized, with the measured rationale for each deviation)
- `BUILD-IDENTITY.md` — image IDs, SGLang commit, component versions, content md5s
- `docs/` — deployment plan, upstream ISSUE/PR survey, benchmark comparison (sanitized export)

## Sanitization

Internal IPs/hostnames are replaced with placeholders and API keys are removed
(`YOUR_API_KEY`); site `.env.tp4` is excluded (`.gitignore`). This fork keeps
the upstream `main` branch untouched — the adaptation lives on `4dgx-ring`.
