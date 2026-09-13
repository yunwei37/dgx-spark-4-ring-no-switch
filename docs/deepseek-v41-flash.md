# DeepSeek-V4.1-Flash on the four-Spark ring

Trial date: 2026-09-10/11 UTC. Four DGX Spark (GB10, sm_121, 128 GB unified
memory) in the repository's direct ConnectX ring, TP=4, one rank per node in
physical ring order. This is a recorded trial, not a promoted profile.

## Inputs

| Input | Pin |
| --- | --- |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277` (88 files, 510.31 GB; every LFS file SHA-256 checked on every node) |
| vLLM tree | `vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58` (`deepseekv41-flash-0909-arm64`, `0.1.dev20904+g179dd0fa9`) |
| Recipe | `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark@592540c` (MIT): boot-10 flags, seven bind-mounted patch files, overlay3/4/5 image chain |
| Serving image | recipe overlay3/4/5 built on the day-0 image (FlashInfer 0.7.0rc1 `07869c61`; prebuilt `mxfp8_gemm_cutlass_sm120` and `sparse_mla_sm120`, `-O3 -DNDEBUG`) |
| Transport | NCCL 2.29.7 (`b91894b`) + `autoscriptlabs/nccl-mesh-plugin@19924dcc`, `NCCL_ALGO=Ring`, runtime connect |

## Deviations from the recipe

- Switchless ring: NCCL Mesh plugin over the direct links instead of the
  recipe's single-subnet `NCCL_NET=IB`. The plugin library directory must be
  on `LD_LIBRARY_PATH`, otherwise NCCL reports `Could not find: mesh
  libnccl-net-mesh.so` and `ncclCommInitRank` fails with `invalid usage`.
- The day-0 image replaces the recipe's overlay1: every recipe diff dry-runs
  cleanly against its files, and its sm_120 `_C_stable_libtorch` runs on GB10.
- The day-0 `nvidia/model.py` imports `gather_engram_hashes` (Engram data
  parallelism), which the recipe's disk-backed `engram.py` predates. An AST
  scan of all imports found it to be the only missing name. The trial appends
  an identity implementation that raises when Engram DP size is above 1.
- `--max-model-len 1048576` with CUDA graphs (recipe serving boot: 300K).
- Full checkpoint on every node's local NVMe (no NFS).

## Serving flags

TP4, `--block-size 128`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192`,
FP8 KV (default), DSpark k=5 (`probabilistic` draft, `block` rejection,
adaptive verification off), `FULL_AND_PIECEWISE` graphs at the exact k and
k+1 multiples, the vision tower with vLLM's default multimodal input handling
(the earlier `--language-model-only` builds stubbed it out), deepseek_v41 tool and
reasoning parsers, Engram rows staged from disk before the forward
(32 threads).

## Results

### gmu 0.83, GPU prefix cache only

- Load 81.36 GiB per rank in 275 s; pod start to ready 8 min.
- KV 7.24 GiB per rank, pool 1,955,534 tokens (1.86x at 1M).
- 512-token code answer: TTFT 0.40 s, 62.2 tok/s decode.
- Recipe prompt set v1, temperature 0, thinking off
  ([raw metrics](../benchmarks/deepseek-v41-flash-tp4-1m-gmu083-2026-09-11.json)):

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---:|---:|---:|
| 1 | 43.05 | 48.42 | 0.372 |
| 2 | 69.42 | 41.04 | 0.729 |
| 4 | 97.73 | 28.16 | 0.469 |
| 6 | 119.61 | 23.08 | 0.494 |
| 8 | 146.81 | 21.18 | 0.645 |

  Coding: 67.57 tok/s at C1, 229.61 tok/s aggregate at C8. Prefill
  1,070-1,168 tok/s (46,810-token prompt, TTFT 40.1 s).

### gmu 0.88: unsafe

KV reached 13.94 GiB per rank (3,819,333 tokens), but the FlashInfer autotune
warmup at 8,192 tokens exhausted host memory: two nodes stopped reporting to
the cluster and the trial ranks were evicted. Do not use gmu 0.88 with an
8,192-token batch on GB10.

### Filesystem prefix-cache tier: not supported on this tree

vLLM's `OffloadingConnector` with a `TieringOffloadingSpec` filesystem tier
was tried at gmu 0.85, 0.83 and 0.80. It needs `--enable-cumem-allocator`
(it rejects `expandable_segments` otherwise) and `--prefix-match-unit 8`
(V4.1 has an 8-token KV group), after which the offload scheduler still
asserts `FullAttentionSpec` for V4.1's `CircularBufferSpec` compressor ring:
one reused block per request that is not prefix-cacheable. Supporting it
needs a connector change that skips such scratch groups, validated by a
cached-versus-cold greedy comparison. At gmu 0.83 the tier's extra host
allocations also left the head node with about 3 GiB available.

### Verified serving configuration (gmu 0.83, GPU prefix cache)

Same flags as above, re-run with a host-memory guard:

- KV pool 2,058,026 tokens (1.96x at 1M); idle MemAvailable 6 GiB on the head
  node, 8 GiB on the others; benchmark lows 3 GiB (head) and 4 GiB.
- 512-token code answer: TTFT 0.37 s, 59.9 tok/s.
- ([raw metrics](../benchmarks/deepseek-v41-flash-tp4-1m-gmu083-verify-2026-09-11.json)):

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---:|---:|---:|
| 1 | 44.21 | 50.14 | 0.376 |
| 2 | 69.13 | 39.51 | 0.373 |
| 4 | 99.13 | 28.44 | 0.476 |
| 6 | 124.35 | 23.84 | 0.590 |
| 8 | 144.71 | 21.08 | 0.638 |

  Coding 68.06 and math 68.36 tok/s at C1. Prefill 1,371-1,590 tok/s.
- Needle at depth 0.5: 993,435 prompt tokens, TTFT 982.4 s (1,011 tok/s),
  correct answer.
- A synthetic prefix probe (an identical 115,519-token prompt sent twice)
  kept a ~59 s TTFT with `vllm:prefix_cache_hits_total` at 0. Production
  traffic contradicts it as a general result: after deployment, 87-92% of
  prompt tokens hit the GPU prefix cache, and agent turns of 130-164K tokens
  reached their first token in 0.9-3.8 s against 65-83 s when the prefix
  changed. The probe's miss is unexplained and specific to that workload.

The published deployment uses gmu 0.83. Vision profiling at 0.80 left too
little KV capacity for one 1M-token request; 0.83 restored that capacity while
keeping the measured head-node memory floor above 3 GiB. The service uses
vLLM's default multimodal item handling rather than a locally chosen
per-prompt image count.

### Per-node NVMe prefix-cache tier

vLLM's `TieringOffloadingSpec` cannot serve this layout: its secondary-tier
I/O runs in the scheduler process through that node's `/dev/shm` staging
region, which holds only rank 0's slice when every TP rank is on a different
node. [`images/runtime/deepseek41/dsv41_kv_nvme.py`](../images/runtime/deepseek41/)
replaces it: the scheduler keeps vLLM's CPU LRU manager as a slot allocator,
and every worker copies its own KV slice between device memory and a
preallocated file on its node's NVMe. The connector subclass skips V4.1's
compressor ring (`CircularBufferSpec`), which the GPU prefix cache also
excludes. [`profiles/deepseek-v41-flash-tp4.sh`](../profiles/deepseek-v41-flash-tp4.sh)
adds `--enable-cumem-allocator` (required with `expandable_segments`) and the
connector configuration.

Checks at gmu 0.80 with a 64 GiB file per node (GPU pool 1,237,719 tokens):

| Check | Result |
| --- | --- |
| 69,289-token prompt, cold | TTFT 39.5 s |
| Same prompt after resetting only the GPU prefix cache | TTFT 0.7 s, 871 MiB read from NVMe, identical greedy answer |
| 230,806-token prompt, cold | TTFT 143.9 s |
| Same prompt after GPU reset | TTFT 1.0 s, 2,868 MiB read, identical greedy answer |
| Follow-up turns after a reset | 0.7 s and 1.0 s, correct |
| 512-token decode | 62.4 tok/s (60.2 without the tier) |

Every offloaded group stores a full block per slot, about 40 KB per token per
rank, so a 64 GiB file holds roughly 1.7M tokens of prefix per node.

### Reasoning effort levels

V4.1 names four effort levels (low 25, high 50, xhigh 75, max 100) and
rejects others with HTTP 400. The patched tokenizer in
`images/runtime/deepseek41/` also accepts `medium` (budget 37) and
`minimal` (low), so OpenAI-style clients can send any of low, medium, high,
xhigh and max. Thinking stays off unless the request enables it. Its post-deployment check: all ranks ready about seven
minutes after start, idle MemAvailable 6 GiB on the head node and 8-9 GiB on
the others, `max_model_len` 1,048,576, and a 512-token code answer at TTFT
0.40 s and 60.2 tok/s decode.
