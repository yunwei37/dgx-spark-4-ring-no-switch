# Four DGX Sparks, one direct ring, no switch

Reproducible large-model inference on four NVIDIA DGX Spark systems (GB10,
sm_121, 128 GB unified memory each) wired as a direct ConnectX ring with no
switch. The repository publishes the serving profiles, ring launchers,
pinned ARM64 runtime images, benchmark records and configuration decisions
for every layout exercised on that ring. It ships no model weights and
modifies no host management networking.

Tested results are always ordered **newest first**: the most recently tested
checkpoint leads, older generations follow in descending test date, and
superseded narratives move to `docs/` history.

## Tested results — newest first

Every number names its measurement boundary; `partial`, `failed`, `unsafe`
and `not measured` stay distinct from success.

| Tested | Checkpoint | Layout | Context boundary | Stable decode | Result |
| --- | --- | --- | --- | ---: | --- |
| 2026-09-11 | `deepseek-ai/DeepSeek-V4.1-Flash` + DSpark k=5, gmu 0.83 | TP=4 | 993,435-token needle correct at 1,048,576 max model length (2,058,026-token KV pool) | 50.14 tok/s C1 mean, 68.06 code; 144.71 tok/s at 8 requests | passed and deployed; 87–92% production prefix-cache hit rate ([record](docs/deepseek-v41-flash.md)) |
| 2026-09-11 | same, gmu 0.88 | TP=4 | 3,819,333-token KV pool | not measured | unsafe: the 8,192-token autotune warmup exhausted host memory |
| 2026-09-11 | same + vLLM filesystem prefix-cache tier | TP=4 | not reached | not measured | unsupported: the offload scheduler rejects V4.1's compressor-ring KV group; the per-node NVMe tier replaces it |
| 2026-09-02 | `Tech2wild/GLM-5.3-Int4-Int8Mix` | TP=4 | 199,489-token input attempted | not measured | startup and short requests passed; near-200K prefill crossed the memory safety floor and was stopped |
| 2026-08-31 | same formal Int4/Int8Mix checkpoint | TP=4 | 8,192 bounded baseline | 12.90 tok/s mean across 3 trials | formal GLM-5.3 loaded and answered real requests; full 1M context not yet passed |
| 2026-08-29 | `zai-org/GLM-5.3-Flash` FP8 | TP=4/EP=4 | 240,000-input retrieval passed | 20.16 tok/s single stream | bounded inference passed |
| 2026-08-29 | same Flash FP8, MTP=5 | TP=4/EP=4 | 78,000-input retrieval passed | 25.57 tok/s forced 512-token decode | bounded pass with a smaller KV pool; not formal GLM-5.3 |
| 2026-08-31 | `RadixArk/Qwen3.8-Flash-Next-NVFP4` + MTP file view | TP=2/EP=2 | 262,144 native window | 40.20 tok/s single; 102.19 tok/s at 4 requests | strict final-key retrieval passed and restored |
| 2026-08-28 | same checkpoint, MTP | TP=2/EP=2 | 262,144 native window | 37.60 tok/s single; 99.66 tok/s at 4 requests | exact mid-context retrieval passed and cleaned up |
| 2026-08-27 | same checkpoint, MTP | TP=4 | 262,144 native window | 48.81 tok/s short cached natural output | exact mid-context retrieval passed; separate decode workload, restored |
| 2026-08-29 | `Qwen/Qwen3.8-Flash-Next` BF16 | TP=4/EP=4 | 32,768 cold input passed; 32,769 failed | 25.87 tok/s short request | long-context correctness failure, not a full-window success |

Formal **GLM-5.3 (not Flash)** remains the first open priority: the bounded
8K pass above is a functional baseline, not the requested 1,048,576-token
result, and formal NVFP4 weights have never completed a load. Prefer two
nodes whenever the complete checkpoint, context, correctness and memory
reserve fit — node count is not a success criterion. See the
[evidence summary](docs/benchmarks.md#recorded-matrix-and-node-count).

### Retained safety boundaries

- The early NVFP4 60,469-token attempt was **unsafe**: unified-memory
  pressure disturbed the management plane and one rank had to reboot. It is
  retained as a limit in the
  [2026-08-25 record](benchmarks/glm52-nvfp4-2026-08-25.json), never reported
  as a successful 60K result.
- gmu 0.88 with an 8,192-token batch is unsafe on GB10 (table above).
- Correctness gates are exact-token retrieval or verbatim answers, never
  sole throughput. Exact records live under [`benchmarks/`](benchmarks/).

Older generation measurements, loader experiments and the full narrative are
history under [`docs/benchmarks.md`](docs/benchmarks.md),
[`docs/loader-memory-results.md`](docs/loader-memory-results.md) and
[`docs/blog/`](docs/blog/); they are not featured results.

## Topology

```text
rank 0 <====> rank 1
  ^             |
  |             v
rank 3 <====> rank 2
```

Each node has two direct ConnectX neighbors. The repository assumes ordinary
connected subnets already exist on those links. It installs no routes,
dispatchers, timers, firewalls, DHCP overrides, or Tailscale configuration.

## Reproduce a tested layout

### Prerequisites (every profile)

- four ARM64 DGX Spark nodes with Docker, the NVIDIA Container Toolkit and
  `/dev/infiniband` ConnectX devices present;
- SSH using normal host-key verification;
- the exact checkpoint revision mounted at the same path on every node
  (read-only) and a writable compilation-cache directory on every node;
- NCCL socket interface/subnet selection appropriate for the current ring.

### Newest profile — DeepSeek-V4.1-Flash, TP=4

Driven by [`profiles/deepseek-v41-flash-tp4.sh`](profiles/deepseek-v41-flash-tp4.sh),
one rank per node in physical ring order. The full measurement record and
checkpoint pins are in [`docs/deepseek-v41-flash.md`](docs/deepseek-v41-flash.md).

Per node (rank `i` = 0..3) the serving container needs:

| Input | Value |
| --- | --- |
| Environment | `NODE_RANK=i`, `MASTER_ADDR=<rank-0 address>`, optional `KV_OFFLOAD_BYTES` (default 64 GiB) |
| Image | see the [runtime images](#runtime-images) table |
| Mounts | `/models/DeepSeek-V4.1-Flash` (checkpoint, read-only); `/kv-offload` (local NVMe directory for the prefix-cache tier); the seven recipe patch files and [`images/runtime/deepseek41/`](images/runtime/deepseek41/) exactly as its README tables them |
| Runtime env | `LD_LIBRARY_PATH=/opt/nccl-mesh/lib:/usr/local/cuda/lib64:...`, `PYTHONPATH=/opt/spark-manage/py` (or wherever `dsv41_kv_nvme.py` is mounted) |

The script `exec`s `vllm serve` with the tested flags: TP4, DSpark k=5
speculative decoding, `--max-model-len 1048576`, `--block-size 128`,
`--max-num-seqs 8`, deepseek_v41 tool/reasoning parsers, and the
`CacheableGroupsOffloadingConnector` NVMe tier. Reasoning effort accepts
`low`, `medium`, `high`, `xhigh` and `max`; thinking stays off unless a
request enables it. A 64 GiB tier file holds roughly 1.7M prefix tokens per
node (~40 KB per token per rank).

### Ring launchers — SGLang and vLLM profiles

Both launchers SSH to the four nodes in `RING_NODES` order, start ranks
1–3 first and rank 0 last, and support `--dry-run` and `--stop`.

Required environment (never committed):

| Variable | Meaning |
| --- | --- |
| `RING_NODES` | Four comma-separated SSH hosts in physical ring order |
| `MASTER_ADDR` | Address reachable by every rank for rendezvous |
| `SOCKET_IFNAME` | Management interface for rendezvous/control traffic |
| `MODEL_HOST_PATH` | Identical existing model directory on every node |
| `CACHE_HOST_PATH` | Existing writable cache directory on every node |
| `VLLM_API_KEY` | vLLM launcher only: API key passed to each container |

Optional: `SSH_USER` (default: current user), `IMAGE` (immutable tag or
digest override).

```bash
# SGLang profile (Qwen3.8 Flash Next NVFP4)
export RING_NODES=rank0.example,rank1.example,rank2.example,rank3.example
export MASTER_ADDR=192.0.2.10
export SOCKET_IFNAME=management-interface
export MODEL_HOST_PATH=/srv/models/Qwen3.8-Flash-Next-NVFP4
export CACHE_HOST_PATH=/srv/cache/qwen38-nvfp4
./scripts/launch-sglang-ring.sh profiles/qwen38-flash-next-nvfp4.sh

# vLLM profile
export VLLM_API_KEY='replace-me'
./scripts/launch-ring.sh profiles/glm52-int4-int8mix.sh

# Stop exactly the four containers of a profile
./scripts/launch-ring.sh --stop profiles/glm52-int4-int8mix.sh
```

The launchers never download weights, change host networking, create swap,
drop caches, install an OOM daemon, or alter existing services. They are
portable Docker reproduction helpers, not the production GitOps deployment.

### Profiles

| Profile | Engine | Layout | Notes |
| --- | --- | --- | --- |
| [`deepseek-v41-flash-tp4.sh`](profiles/deepseek-v41-flash-tp4.sh) | vLLM | TP=4, 1M context, DSpark k=5 | newest; per-node NVMe prefix-cache tier |
| [`qwen38-flash-next-nvfp4.sh`](profiles/qwen38-flash-next-nvfp4.sh) | SGLang | TP=4 (or TP=2/EP=2), 262K window | EAGLE MTP 1-step/2-draft; the passing configuration |
| [`glm52-int4-int8mix.sh`](profiles/glm52-int4-int8mix.sh) | vLLM | TP=4, 8K | historical first profile; optional `MTP_TOKENS` |

Each profile defines `PROFILE_NAME`, `CONTAINER_NAME`, `IMAGE` (immutable
tag, overridable), the model repository/revision and container path, the
engine arguments, and the `CONTAINER_ENV` NCCL/Mesh environment. The Qwen
SGLang compatibility patches are source-hash-guarded files under
[`images/runtime/`](images/runtime/); they are test-scoped patches for the
recorded immutable image, not host modifications.

## Runtime images

All images are `linux/arm64`, built from digest-pinned bases with NCCL
2.29.7 and the Mesh plugin at the exact tested commits, and published to
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch` by the `workflow_dispatch`
workflows under [`.github/workflows/`](.github/workflows/). Publication
digests and verification status for every tag are recorded in
[`docs/image.md`](docs/image.md).

| Tag | Dockerfile | Serves |
| --- | --- | --- |
| `deepseek-v41-flash-20260911` | `Dockerfile.deepseek-v41-flash` | DeepSeek-V4.1-Flash (newest) |
| `glm53-intmix-nvfp4-dflash2-<sha>` | `Dockerfile.vllm-glm53-intmix-nvfp4-dflash2` | formal GLM-5.3 IntMix NVFP4 DFlash2 experiments |
| `glm53-intmix-nvfp4-dcp4-<sha>` | `Dockerfile.vllm-glm53-intmix-nvfp4` | formal GLM-5.3 IntMix NVFP4 experiments |
| `glm53-intmix-router-<sha>` | `Dockerfile.vllm-glm53-intmix` | formal GLM-5.3 INT4/INT8 router candidate |
| `glm53-nvfp4-loader-20260829` | `Dockerfile.sglang-glm53` | GLM-5.3 NVFP4 loader experiments |
| `qwen38-flash-next-nvfp4-20260827` | `Dockerfile.sglang-qwen38` | Qwen3.8 Flash Next NVFP4 |
| `int4-int8mix-20260824` | `Dockerfile` / `Dockerfile.package` | the first historical INT4 runtime |

Images contain runtimes only, never model weights. Until
[`docs/image.md`](docs/image.md) records an inference smoke for a tag, its
benchmark data proves the component recipe, not the assembled package.

## Documentation

- [`docs/deepseek-v41-flash.md`](docs/deepseek-v41-flash.md) — newest
  record: DeepSeek-V4.1-Flash inputs, deviations, measurements, NVMe tier.
- [`docs/architecture.md`](docs/architecture.md) — the ring boundary.
- [`docs/benchmarks.md`](docs/benchmarks.md) — measurements and limitations.
- [`docs/configuration-decisions.md`](docs/configuration-decisions.md) —
  every retained non-default tied to evidence.
- [`docs/image.md`](docs/image.md) — image publication digests and status.
- [`docs/loader-memory-results.md`](docs/loader-memory-results.md) — tested
  loader alternatives.
- [`docs/glm53-community-experiments.md`](docs/glm53-community-experiments.md)
  — external formal GLM-5.3 experiments audited separately from our results.
- [`docs/blog/`](docs/blog/) — experiment narratives.

## Validate

```bash
python3 tests/validate_repo.py
bash -n scripts/launch-ring.sh scripts/launch-sglang-ring.sh profiles/*.sh
```

Repository-authored files are MIT licensed. Third-party components keep
their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
