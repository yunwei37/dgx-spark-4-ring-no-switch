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
k+1 multiples, `--language-model-only`, deepseek_v41 tool and reasoning
parsers, Engram rows staged from disk before the forward (32 threads).

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

### gmu 0.85 with a filesystem prefix-cache tier

In progress.
