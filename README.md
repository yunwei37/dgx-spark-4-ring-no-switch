# Four DGX Sparks, one direct ring, no switch

Reproducible GLM and Qwen inference on four NVIDIA DGX Spark systems connected
as a direct ConnectX ring. This repository publishes only configurations and
results that were exercised on the four-node ring. It does not ship model
weights or modify host management networking.

![Measured GLM-5.2 decode throughput](docs/assets/glm52-decode-throughput.svg)

## Tested results

| Checkpoint | Layout | Context | Stable decode | Result |
| --- | --- | ---: | ---: | --- |
| `QuantTrio/GLM-5.2-Int4-Int8Mix` | TP=4 | 8,192 | 12.99 tok/s | passed and restored |
| same checkpoint, MTP=4 | TP=4 | 8,192 | 33.42 tok/s fixed workload | passed; low memory reserve |
| `0xSero/GLM-5.2-504B-Nvidia` | PP=4 | 32,005 | 4.40 tok/s | passed and restored |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` + MTP | TP=4 | 262,144 native window | 48.81 tok/s short cached natural output | exact mid-context retrieval passed; separate decode workload, restored |
| same checkpoint, MTP | TP=2/EP=2 | 262,144 native window | 37.60 tok/s single; 99.66 tok/s at 4 requests | exact mid-context retrieval passed and cleaned up |
| same full checkpoint, MTP file view | TP=2/EP=2 | 262,144 native window | 40.20 tok/s single; 102.19 tok/s at 4 requests | strict final-key retrieval passed and restored |
| `Qwen/Qwen3.8-Flash-Next` BF16 | TP=4/EP=4 | 32,768 cold input passed; 32,769 failed | 25.87 tok/s short request | long-context correctness failure, not full-window success |
| `zai-org/GLM-5.3-Flash` FP8 | TP=4/EP=4 | 240,000 input retrieval passed | 20.16 tok/s single | bounded inference passed |
| same Flash FP8, MTP=5 | TP=4/EP=4 | 78,000 input retrieval passed | 25.57 tok/s forced 512-token decode | bounded pass; less KV capacity, not formal GLM-5.3 |

Formal **GLM-5.3 (not Flash)** remains the first priority and has **zero**
successful local inference requests. August31 native NVFP4 loading failed
during initial reads. A follow-up pre-read ownership/sliced-vocabulary patch
saved about1.74GiB RSS in a byte-identical GPU component but loaded more slowly;
the actual full-model follow-up still hit the host-memory floor on its final
pipeline rank. [Failure data and graphs](docs/loader-memory-results.md) are
published; constructor or component savings are not inference.
The requested INT4/INT8 checkpoint finished full upstream
checksum/index/size verification on August31; its temporary verifier was removed.
It has not completed local inference;
its [four-rank router/capacity preflight](docs/benchmarks.md#formal-glm-53-int4int8-preflight-2026-08-31)
measured 94.65 GiB/rank of static tensors with the FP32-router candidate, not a
full checkpoint load or serving peak. The first actual TP4 attempt at21:05UTC
then failed at fastsafetensors' initial4.65GiB GPU file-buffer allocation after
94.91GiB Torch allocation; its97GiB test cap cannot fit that combination.
All four ranks had joined; no API/inference completed. Original service was
restored and authenticated generation passed. See
[the full-load failure record](benchmarks/glm53-intmix-first-load-2026-08-31.json).
A subsequent native `safetensors` run completed all weights and postprocessing
on all four ranks in477-536seconds, using94.5GiB each. Initialization then
failed on an896MiB MLA profiling workspace rejected by our97GiB test allocator
limit. This limit is not an upstream requirement; it is being removed rather
than promoted as a serving default. One driver allocation warning is retained.
There is still no successful formal-model inference or full-context result.
See [the native-loader evidence](benchmarks/glm53-intmix-native-load-2026-08-31.json).
The requested final context is the checkpoint's native1,048,576tokens;
the8K diagnostic and community200K/300K configurations are not that result.
Prefer two nodes when the complete checkpoint, context, correctness and memory
reserve fit; node count is not a success criterion. The formal ~465GB NVFP4 and
~405GB INT4/INT8 checkpoints do not fit entirely in two 128GB nodes.
See the [evidence summary](docs/benchmarks.md#recorded-matrix-and-node-count).

The INT4 MTP mixed-prompt runs measured 16.37-23.71 tok/s. The NVFP4 60,469
token attempt was unsafe: unified-memory pressure affected the management plane
and one rank had to reboot. That failure is retained as a limit, not reported as
a successful 60K result. Qwen3.8 Flash Next exercised its full 262,144-token
native window with 261,888 prompt tokens plus a 256-token output budget. Exact
records are under [`benchmarks/`](benchmarks/).

The exact-token retrieval client is
[`scripts/max_context_probe.py`](scripts/max_context_probe.py). The two
source-hash-guarded SGLang compatibility patches used by the Qwen run are under
[`images/runtime/`](images/runtime/); they are test-scoped patches for the
recorded immutable image, not host modifications.

The bounded Qwen TP2 native-window run recovered its mid-prompt marker with
2,249.41 prefill tok/s, 31.22 tok/s cold full-context decode, and 38.58 tok/s
after a 258,048-token cache hit. Its 35-minute cold start remains an explicit
loader bottleneck, not a production-ready startup result.

The [MTP file-selection experiment](docs/blog/2026-08-31-qwen-mtp-file-selection.md)
subsequently reduced native draft loading from the historical 839–842 seconds
to 19–20 seconds, with strict 262K final-key retrieval passing. Full target
loading still took about 20 minutes; total container-start-to-ready was 22m21s.
This is a historical comparison, not a matched-cache A/B or one-minute startup.

The matched Qwen full-depth natural-output run decoded at 48.81 tok/s with
86.86% MTP acceptance. A 3-step/4-draft candidate appeared faster at 63.67
tok/s but deterministically degenerated to repeated punctuation and failed
retrieval twice, so the reusable profile keeps the passing 1-step/2-draft
configuration.

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

## Reproduce the tested INT4 profile

Requirements:

- four ARM64 DGX Spark nodes with Docker and NVIDIA Container Toolkit;
- SSH using normal host-key verification;
- the exact checkpoint revision mounted at the same path on every node;
- a writable compilation-cache path on every node;
- NCCL socket interface/subnet selection appropriate for the current ring.

Set the environment without committing it:

```bash
export RING_NODES=rank0.example,rank1.example,rank2.example,rank3.example
export MASTER_ADDR=192.0.2.10
export SOCKET_IFNAME=management-interface
export MODEL_HOST_PATH=/srv/models/GLM-5.2-Int4-Int8Mix
export CACHE_HOST_PATH=/srv/cache/glm52-int4
export VLLM_API_KEY='replace-me'
./scripts/launch-ring.sh profiles/glm52-int4-int8mix.sh
```

Stop exactly these four containers with:

```bash
./scripts/launch-ring.sh --stop profiles/glm52-int4-int8mix.sh
```

The launcher never downloads weights, changes host networking, creates swap,
drops caches, installs an OOM daemon, or alters existing services. It is a
portable Docker reproduction helper, not the production GitOps deployment.

## Container image

`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:int4-int8mix-20260824`

The image combines the exact tested INT4 base runtime with the pinned NCCL Mesh
plugin and the one measured loader memory-lifetime fix. Publication digest and
verification status are recorded in [`docs/image.md`](docs/image.md). Until that
file says `inference smoke: passed`, the old benchmark data proves the component
recipe, not the newly assembled package.

The Qwen SGLang package is built by `Dockerfile.sglang-qwen38`; the exact
four-rank arguments are in `profiles/qwen38-flash-next-nvfp4.sh` and the bounded
Docker launcher is `scripts/launch-sglang-ring.sh`. The image contains the
runtime and compatibility fixes, never the model weights.

`Dockerfile.sglang-glm53` packages the exact SGLang digest used by the formal
GLM-5.3 NVFP4 loader experiment, `fastsafetensors 0.3.3`, NCCL Mesh, and the
source-hash-guarded fixes. Experimental builds and constructor diagnostics are
recorded in `docs/image.md`; GHCR publication and corrected full TP4 inference
remain incomplete. They are not successful formal GLM inference evidence.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) explains the ring boundary.
- [`docs/benchmarks.md`](docs/benchmarks.md) gives measurements and limitations.
- [`docs/configuration-decisions.md`](docs/configuration-decisions.md) ties each
  retained non-default to evidence.
- [`docs/loader-memory-results.md`](docs/loader-memory-results.md) records the
  tested loader alternatives.
- [`docs/glm53-community-experiments.md`](docs/glm53-community-experiments.md)
  audits external formal GLM-5.3 experiments, including switchless-ring reports,
  and separates them from our own results and unvalidated recipes.
- [`docs/blog/2026-08-25-glm52-on-four-dgx-sparks.md`](docs/blog/2026-08-25-glm52-on-four-dgx-sparks.md)
  is the experiment narrative.

## Validate

```bash
python3 tests/validate_repo.py
bash -n scripts/launch-ring.sh profiles/*.sh
```

Repository-authored files are MIT licensed. Third-party components keep
their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
