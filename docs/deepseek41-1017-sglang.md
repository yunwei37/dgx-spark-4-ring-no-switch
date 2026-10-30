# DeepSeek V4.1 Flash 1017 on four DGX Sparks: SGLang qualification

Tested 2026-10-30. This is the **official** October checkpoint, not an
uncensored or abliterated derivative. The runtime is serving through the
existing gateway, with real client and near-1M acceptance. The throughput
goal remains incomplete.

## What actually ran

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, native FP8/FP4.
- Four DGX Sparks, TP4/EP1, one rank per node, direct ConnectX ring.
- [Recipe v2.6.2](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa),
  SGLang `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9` and its ordered backports.
  The inherited wheel version is stale; source provenance is authoritative.
- NCCL 2.29.7, Mesh `19924dcc`. This is the recipe's documented **NCCL ring
  fallback**, not its switched RoCEnante transport.
- 1,048,576 context; 16 running requests; static memory fraction 0.78;
  4,194,304 KV tokens; 8192-token prefill chunks; DSpark block size 5.
- Packed local Engram, 4 GiB cache per node, 16-way, 96 I/O threads,
  no resident scales and prefetch enabled.
- Production compute/prefill adapters, native extra-page kernels and normal
  FlashInfer autotune enabled. Encoder-only replay is disabled; decoder replay
  follows upstream. Thinking is enabled by default.
- No GPU-clock cap, reboot, host compaction, new forwarding rule or daemon was
  added for this qualification.

The profile is generated from the pinned recipe's Production block and latest
README release gate. It is not a second handwritten list of tuning values.
Upstream defaults, the example's older values and its measured Production
profile are different things.

## Matched throughput comparison

Pinned speedbench `e95a638b5f92d26cc0724e7941abb8053cde3e20` (0.3.0),
the same frozen 73 prompts plus four sentinels, temperature 0, thinking off,
256 output-token budget, C1; load phases use 90 seconds each. The same client
hardware was used for the local before/after runs. There was no benchmark
overlap with Engram packing or profiling.

| Metric | Earlier stripped EP1 profile | Production-profile NCCL ring | Requested reference |
|---|---:|---:|---:|
| C1 overall geometric mean, tok/s | 51.21 | **67.49** | 89.3 |
| C1 code geometric mean, tok/s | 67.10 | **86.18** | 112.2 |
| C1 90-second load aggregate, tok/s | 45.64 | **61.70** | separate from C1 geometric mean |
| C4 load aggregate, tok/s | 84.89 | **106.12** | not used as the target |
| C16 load aggregate, tok/s | 159.28 | **194.40** | 211 |
| Cold 64K prefill, tok/s | 3705.2 | **5377.6** | about 5300–6000 |
| Cold 64K TTFT, seconds | 17.716 | **12.207** | workload-dependent |

Both SGLang C1 runs had 73/73 valid samples and zero failed requests. New C16
95% block-bin interval was 172.3–218.3 tok/s; overlap with 211 is not evidence
that the point target was achieved. New C1 median TTFT was 215 ms and median
decode step 47.03 ms. Four repeated sentinels returned identical text; their
timing drift was -2.8%.

The new profile improves overall C1 about 32%, code about 28%, C16 about 22%
and cold 64K prefill about 45%. The earlier vLLM overall 44.53/code 59.92 run
had only 71 valid decode samples; it is not a fully matched 73-sample ratio.

The requested 89.3/112.2/211 figures came from upstream measurements, not a
guarantee for another transport. The pinned recipe's latest official-1017
release gate itself reports C1 88.38, C16 207.62 and cold 64K 6054 in its
v2.6.2 arm. Do not combine that release gate with older checkpoint results.
See [upstream current results](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa/README.md).

## Correctness and clients

- All eight business checks passed: default/disabled thinking, thinking alias,
  tool invocation and continuation, strict JSON schema, streaming finish/usage,
  and native image understanding.
- A real **1,039,058-input-token** request recovered all three planted values
  at different depths, returning 13 output tokens in **247.915 seconds**.
  This is retrieval acceptance, not an isolated cold-prefill throughput number.
- Frozen 114-question native-probability MMLU pilot: **104/114**. Earlier
  stripped SGLang and vLLM one-forward pilots were 103/114. Three winners
  differ from the vLLM reference; numerical or full-quality equivalence is
  not established. The earlier full vLLM score was 12077/14042 (86.006%).
  Full SGLang and derivative-model comparisons remain pending.
- OMP 18.13.3 completed a real multi-turn tool task, including recovering from
  a missing file and locating the correct manifest. It changed no project code.
- The gateway's stable names remain `local-large` (1M) and `local-small`
  (200K). OMP now discovers context/reasoning/vision from native LiteLLM
  metadata, with no copied model catalog or manually supplied client limits.
  A smaller member was removed from the large alias because its metadata
  overwrote the 1M window. Small remains independently available.
- The actual public curl client received 401 for missing/invalid auth and 200
  for an existing valid credential. Python's default urllib User-Agent received
  Cloudflare 1010; no edge policy or credential was changed.
- All four ranks remained Ready with zero restarts through the full benchmark
  and near-1M check. This is a bounded stability window, not a 24-hour soak.

## Why the first attempt lagged, and what remains

The first custom launcher omitted much of the selected Production profile.
Correcting that omission is the measured before/after improvement above.
It is inaccurate to call the current setup identical to upstream's fastest
deployment: compute, memory and Engram follow the profile; transport differs.

A new 32-step CPU/GPU trace was collected after the change, separately from
throughput measurements. Per rank, NCCL occupies **28–31% of summed CUDA-kernel
durations**. The 2873 observed AllReduce kernels average about **150–182 us**;
upstream documents about 56 us for its patched NCCL ring path. These are
cross-site observations, not a controlled transport A/B, and summed kernel
time is not critical-path wall time. MoE and dense GEMMs remain substantial.
The evidence supports investigating collectives next; it does not establish
that the network explains the entire gap.

RoCEnante's full ring path needs opposite-node hardware forwarding and its
own qualification; its switched two-shot settings cannot simply be enabled
on a plain ring. Those host/network changes have not been applied.

## Minimal local differences and failure history

The build preserves credited upstream backports, the producer claim-barrier
fix and prebuilt native-page kernels. Local changes redact auth diagnostics,
keep source digests as provenance rather than additional acceptance gates,
avoid persisting a second API-key copy, and leave request length/loop abort
client-controlled. Functional shape/dtype/parity checks remain.

Early native-default request-capacity and memory experiments failed; raising
the memory fraction caused driver memory pressure and node loss. Those
experiments were retired and are not a recommended profile. A temporary
encoder-replay workaround excluded vision; the current Production-profile
candidate removes it and passes native vision. Diagnostic autotune-disable
was also removed.

## Image identity and reproduction

Measured OCI manifest:
`sha256:234754c5b53aa21c29d96afe2c0e9361d81509c34fc0707eef7a1b40f3797579`.

Image config:
`sha256:a1912a21e0b3df0f9c474adb37f2c58b359bf68eb0ba801681e5bb8ed5492329`.

Publication target:
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:sglang-cfe3cfb-production-20261030c`.
Publication completed through [Actions run 41022059900](https://github.com/yunwei37/dgx-spark-4-ring-no-switch/actions/runs/41022059900).
Anonymous registry readback reports the exact manifest and config above.
Native containerd pulls of both published references completed on all four
serving nodes using their identical cached content.

The companion image is
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:nccl-2.29.7-mesh-19924dcc@sha256:ee6e0bb0ea8e36a10e3d9ed85745427deb1109e5b2e5ab27f32b0b7c15ed309e`.

The [release assets](https://github.com/yunwei37/dgx-spark-4-ring-no-switch/releases/tag/sglang-native-1017-20261030)
retain the exact overlay, companion OCI archive, generated profile and build
provenance. Neither image was rebuilt during publication. Registry readback
and deployment health are separate from the still-unmet throughput goal.

The public upstream base is
`lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9`.
Only about 50.3 MB of additional compressed OCI layers are local changes.
A release overlay plus that immutable public base reconstructs the measured
manifest without recompiling or redistributing weights.

See [build sources](../images/sglang-native/README.md). Models, credentials,
private topology and raw request/response logs are not part of this repository.
