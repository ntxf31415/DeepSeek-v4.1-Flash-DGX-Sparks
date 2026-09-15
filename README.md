# DeepSeek-V4.1-Flash (SGLang) on 4× DGX Spark, TP4, Switchless Ring

Production recipe for serving **deepseek-ai/DeepSeek-V4.1-Flash** (552B MoE, 8B/16B
active, MXFP4 experts, 1M context, DSpark speculative decoding) with **SGLang TP4**
across **4× NVIDIA DGX Spark (GB10)** connected as a **switchless RoCE ring** (no
400G switch).

> 中文说明：[README.zh-CN.md](README.zh-CN.md) · 部署方案与基准对比：[docs/](docs/)

**Measured on 4× DGX Spark (GB10, sm_121a, switchless ring), 1M ctx / 8M KV pool.**
The build these numbers came from is pinned in [BUILD-IDENTITY.md](BUILD-IDENTITY.md).

| benchmark | value |
|---|---|
| **prose decode, 1 stream** (sparkDash protocol) | **48** tok/s · TTFT ~185 ms |
| **aggregate c1 / c2 / c4 / c6 / c8 / c12** (same protocol) | **48 / 71 / 119 / 148 / 163 / 220** tok/s |
| decode peak / mean (code, temp 0) | 100.3 / 81.4 · 99.8 / 81.2 tok/s |
| prefill 8K / 32K / 100K | 3102 / 3443 / 3253 t/s |
| quality gates | needle 30K-470K ✅ · corruption 0/0/0 · termination 18/18+18/18 · code-gate 12/12 · GSM8K n=50 **1.00** |
| DSpark acceptance | 3.98 tok/step, rate 0.595 (6.0 saturated on math) |
| cold start | ~9 min, **no cold-start penalty** |

The prose and aggregate rows use the [sparkDash](https://github.com/MiaAI-Lab/sparkDash)
`DecodeBench` protocol — the one the upstream fleet publishes. One fixed prose prompt,
streaming, `min_tokens = max_tokens = 256` with `ignore_eos`, decode measured as
`(completion_tokens − 1) / (t_last − t_first content chunk)`: post-TTFT, so prefill and
queueing stay out of the denominator. Thinking off. The same prompt timed by wall clock
reads ~30 % lower — **the two bases are not comparable**, so quote the protocol with the
number.

`decode peak / mean` is the *ordinary-output* row: single stream over the code and mixed
prompts in [`bench/bench_tp.py`](bench/bench_tp.py), timed by **wall clock**, so its
denominator still carries the prefill. That is the basis this table used before
sparkDash; it is kept for continuity and has **not** been re-baselined. `prefill` is a
separate measurement.

**Long context** (cold prefill, needle-checked):

| depth | result |
|---|---|
| 470K | ✅ 211.7 s · 2144 t/s |
| 600K | ✅ 341.7 s · MemAvailable floor 4.92 GB |
| **900K** | ✅ **completes with adaptive chunking** (2367 s, MemAvailable floor 1.96 GB). A fixed 2048 chunk wedges the engine here: the indexer's per-chunk transient is ~14 B × chunk × prefix, ~26 GB at a 900K prefix. See the adaptive-chunk adapter below. |

Long-prefill *speed* is the trade for that headroom: 600K costs 1.08× the fixed-2048
time and 900K costs 3.4×. Prefixes below the adaptive low-water mark keep the full
static chunk, so ordinary traffic is unaffected.

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
| `MEM_FRACTION_STATIC=0.85` + `MAX_TOTAL_TOKENS=8M` (was 0.90 / 5M) | the KV pool was not the binding limit — the prefill transient is. 0.85 frees ~16 GB outside the static pool for it, which pays for both the larger pool and the long-prefill peak |
| adaptive chunking (`DSV41_ADAPTIVE_CHUNK=1`, LOW 400K / HIGH 800K / FLOOR 1024) | sizes each prefill chunk from the prefix length, so the transient stays bounded without paying the small-chunk step penalty on short prompts. Reuses the `dynamic_chunk_sizer` hook SGLang already has (it only installs under `pp_size > 1`) |

Tried and reverted: `--enable-deepseek-v4-fp4-indexer` costs ~11 % on 500K cold
prefill and 6.4 GB of unified memory for no c12 gain. Single regression kept:
c6 260 → 236 (an EP2 side effect; c8/c12 rise far more).

## Repo contents

- `start.sh / start-tp4.sh / stop.sh / boot.py` — serving orchestration, pinned checkpoint boot, smoke + warm-up
- `adapter/` — SGLang patches (Engram row store C++, MXFP8 backend, shared-expert K pad, prefill cache hook, prefix-sized chunking, KV-pool byte accounting)
- `runtime/` — build-time patches applied over the base image (`patch_encoding_dsv41.py` tolerates the image placeholder token in message text, which upstream rejects with a self-sustaining HTTP 500 on the Anthropic endpoint)
- `scripts/` — SSH helper, verify/ probe kit, self-heal monitor + systemd unit, `gate.sh`, `nccl_selfcheck.sh`
- `bench/` — gate suite (needle / corruption / termination / code-gate), vision gate, event-timeline matrix + common-window analysis, prose, GSM8K spot, third-party-shaped sweep
- `.env.tp4.ring.example` — the configuration this repo actually runs (sanitized, with the measured rationale for each deviation)
- `BUILD-IDENTITY.md` — image IDs, SGLang commit, component versions, content md5s
- `docs/` — deployment plan, upstream ISSUE/PR survey, benchmark comparison (sanitized export)

## Sanitization

Internal IPs/hostnames are replaced with placeholders and API keys are removed
(`YOUR_API_KEY`); site `.env.tp4` is excluded (`.gitignore`). This fork keeps
the upstream `main` branch untouched — the adaptation lives on `4dgx-ring`.
