# 4dgx-dsv41 — ring adaptation notes (sanitized export)

4dgx home-lab adaptation of the upstream recipe in this repository, for a
**4-node DGX Spark switchless ring** (no Ethernet switch). See the project
intro, layout and disclaimers in [README-dsv41.md](README-dsv41.md) and the
full deployment plan + benchmark comparison in [docs/](docs/).

## Quick facts

- SGLang TP4, 1M ctx / 4M KV pool, static verify + SPS table, packed Engram
- Ring-only NCCL 2.30.7 + libncclpin v9 via LD_PRELOAD; GID iron rule;
  per-rank PEER_HCA
- prefill 8K/32K/100K = 2814/3149/3119 t/s · decode 89.6 t/s (code) ·
  quality gates all green · cold boot ~9 min, no cold-start penalty
- Patches P1–P7 and ring contracts: [README-dsv41.md](README-dsv41.md)

## What was sanitized

Internal IPs/hostnames replaced with placeholders, API keys removed
(`YOUR_API_KEY`), site-specific `.env.tp4` excluded — see `.gitignore`.
