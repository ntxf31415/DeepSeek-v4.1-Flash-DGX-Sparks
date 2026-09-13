# Build identity — what the reference fleet actually runs

Recorded 2026-09-13 from the live deployment. Use this to tell whether your
build is the same one the `[measured]` numbers in `.env.tp4.ring.example`
and `README-dsv41.md` came from.

## Images

| | value |
|---|---|
| Overlay | `dsv41-4x-spark:local`, **image ID `sha256:6749a6de1c34a3f20f0b7762107540922defde69b836d3feb10cdcd10ece18d1`**, 33.2 GB |
| Overlay repo digest | **`<none>`** — built locally with `docker build`, never pushed to a registry, so there is no `RepoDigest` |
| Base | `lmsysorg/sglang:dev-dsv41`, image ID `sha256:63700570275c75dab96621f38805682d69e97ce5bb2ef6adba4ded5fcc295c53` |
| Base repo digest | **`<none>`** — the base was obtained by tag/`docker load`, not by digest, so `docker images --digests` shows none either |

**There is no registry digest for either image.** The content-addressed image ID
is the reproducible identifier; the base tag `dev-dsv41` is mutable upstream, so a
later rebuild can differ. Lock it yourself before rebuilding:

```bash
docker image inspect lmsysorg/sglang:dev-dsv41 --format '{{.Id}}'   # pin this
docker image inspect dsv41-4x-spark:local  --format '{{.Id}}'
```

If you rebuild the overlay on another host, the image ID will differ — `docker
build` embeds build timestamps. Compare the *contents* instead:

```bash
docker run --rm --entrypoint sh dsv41-4x-spark:local -c \
  'md5sum /opt/dsv41/boot.py /opt/dsv41/adapter/librow_store.so \
          /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py'
# reference fleet:
#   74025c52d5011316a5fad6a32243c279  /opt/dsv41/boot.py
#   1f820aef687d0f9685c0e4b51585ece6  /opt/dsv41/adapter/librow_store.so
#   d91b0cda319e8a9b57e73e71020fb7bb  .../attention/flash_mla_sm120.py
```

## Software stack (inside the overlay image)

| Component | Version |
|---|---|
| SGLang | `0.0.0.dev1+ge087e662b` — **commit `e087e662b`** |
| FlashInfer | `0.6.18` (`flashinfer-python`) |
| PyTorch | `2.13.0+cu130` (CUDA 13.0) |

## Host stack (all four nodes identical)

| | value |
|---|---|
| GPU | NVIDIA GB10 (SM121), 121.7 GiB unified per node |
| Driver | `580.173.02` |
| Kernel | `6.17.0-1031-nvidia` |
| NCCL (host, ring-only) | `/opt/nccl-ringonly/libnccl.so.2.30.7` — **LuZ lineage**, 2.30.7 |

The NCCL is the LuZ ring-only build, *not* a sparkring patched library: it
contains neither `SWITCHLESS_RING_ONLY` nor `SKIP_TREE_CONNECT` and achieves
ring-only by disabling the Tree algorithm in NCCL's algorithm matrix
(`Tree=0 / Ring=1` for all five collectives) rather than by skipping the tree
transport connect. `scripts/nccl_selfcheck.sh` identifies which lineage a given
library is.
