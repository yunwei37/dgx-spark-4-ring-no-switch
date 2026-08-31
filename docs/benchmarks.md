# GLM-5.2 benchmark results

All numbers are direct measurements from one four-node DGX Spark ring. They are
not vendor specifications and should not be generalized to other images,
revisions, prompts, or physical link layouts.

## INT4/INT8Mix

The exact `QuantTrio/GLM-5.2-Int4-Int8Mix` revision ran TP=4 with an 8,192-token
maximum context. The base profile loaded in 589.25 seconds, occupied 94.38 GiB
per rank, and averaged 12.99 decode tokens/s across three trials.

MTP with four speculative tokens averaged 33.42 tokens/s on the fixed workload
and accepted 413 of 420 proposed draft tokens. Mixed prompts measured 17.7790,
16.3667, and 23.7124 tokens/s with roughly 43% acceptance. MTP raised load time
to 1,183.05 seconds and memory to 96.92 GiB per rank, leaving only 3.3-5.0 GiB;
it passed the bounded test but is not a generous service envelope.

## ModelOpt NVFP4

The exact `0xSero/GLM-5.2-504B-Nvidia` revision ran PP=4/TP=1. At 32,005 input
tokens it loaded in 891.98 seconds, became API-ready in roughly 1,080 seconds,
prefilled at 221.13 tokens/s, and stabilized at 4.40 decode tokens/s. The agent
and tool-call path passed.

The 60,469-token attempt caused unsafe unified-memory pressure, management-plane
latency, and a reboot of one rank. It did not pass. MTP was unsupported with PP
in the tested runtime.

Machine-readable inputs are in:

- [`glm52-int4-int8mix-2026-08-24.json`](../benchmarks/glm52-int4-int8mix-2026-08-24.json)
- [`glm52-nvfp4-2026-08-25.json`](../benchmarks/glm52-nvfp4-2026-08-25.json)

## Qwen3.8 Flash Next NVFP4

The exact `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision ran TP=4 with native MTP
and the full 262,144-token window. A 261,888-token prompt placed a unique marker
at token offset 130,932 and reserved 256 tokens for generation. The server
reported all 261,888 input tokens and recovered the exact marker.

TTFT was 173.94 seconds, equivalent to 1,505.59 prompt tokens/s; end-to-end time
was 174.59 seconds. MTP accepted 83.33% of proposed draft tokens. Engine startup
was 460.44 seconds and the API/tokenizer path was ready in 465.25 seconds. The
lowest bounded host `MemAvailable` sample was 15.36 GiB. This is a synchronous
sample rather than continuous telemetry.

A second request reused 258,048 cached prompt tokens and forced generation past
the natural stop boundary to isolate full-depth decode. It produced 254 tokens
at 26.59 tok/s after the first event; MTP acceptance was 28.43%. Because
`ignore_eos` was set, this is a decode stress result rather than a response
quality result. It also shows that the short-context 47.98 tok/s measurement
must not be presented as full-depth throughput.

For the matched natural-long-output workload, the same 1-step/2-draft MTP
profile generated 256 tokens at 48.81 tok/s after the first event and recovered
the marker. MTP acceptance was 86.86%. This request reused 261,824 prompt tokens
to isolate decode at the full context depth; it is the decode baseline for the
subsequent 3-step/4-draft A/B, not a second prefill result.

The matched 3-step/4-draft candidate reached 63.67 tok/s with a 100% reported
acceptance rate, but both the cold and cached trials produced degenerate repeated
punctuation and failed to recover the marker. It is a quality failure, not a
throughput win. The published profile therefore retains 1-step/2-draft, whose
48.81 tok/s result recovered the marker and produced the requested response.

The immutable SGLang image needed source-hash-guarded, test-local compatibility
fixes for gated-MoE TP padding and for the QSA sparse-decode architecture gate.
The latter lets SM121 use the already-present TRT-LLM sparse decode kernel rather
than the broken FlashAttention-4 CuTe fallback. No host daemon, cache-drop loop,
swap setting, route, or management-network change was added.

A bounded TP2/EP2 rerun also exercised the full native window. Its 261,888-token
prompt plus 256-token output budget recovered the exact mid-prompt marker. Cold
TTFT was 116.42 seconds at 2,249.41 prefill tok/s and 31.22 decode tok/s. The
immediate cache-hit run reused 258,048 tokens, reached 2.67-second TTFT, and
decoded at 38.58 tok/s. A separate 512-token single request reached 37.60 tok/s;
four concurrent 512-token requests reached 99.66 tok/s aggregate.

This TP2 correctness pass took 2,140.59 seconds to become tokenizer-ready. Both
the target and native MTP passes scanned all 206 checkpoint shards, so the run
is not an optimized cold-start result. Request-local
`chat_template_kwargs.enable_thinking=false` returned clean output without a
server response-rewriting patch.

Machine-readable inputs:

- [`qwen38-flash-next-nvfp4-2026-08-27.json`](../benchmarks/qwen38-flash-next-nvfp4-2026-08-27.json)
- [`qwen38-flash-next-nvfp4-tp2-context-2026-08-28.json`](../benchmarks/qwen38-flash-next-nvfp4-tp2-context-2026-08-28.json)

## Qwen TP2 MTP file view, 2026-08-31

The identical complete target checkpoint with a temporary three-file native
draft view passed two short arithmetic cases and strict final-key retrieval
at 261888 input tokens plus a 256-token budget. Target loads were 1215.30/1209.90s;
MTP loads were 19.00/20.23s. Single/four-concurrent synthetic completion measured
40.20/102.19 tokens/s. Native CUDA graphs remained enabled. The historical
comparison is not matched-cache A/B; the old conflicting long-response fixture
is not counted as a strict final-output pass. The TileLang JIT checker warning
is retained as unresolved, and total token pool/concurrency changed to 308416/9.

See [the full result](../benchmarks/qwen38-mtp-view-tp2-2026-08-31.json),
[loading-stage chart](assets/qwen38-mtp-load-stages.svg), and
[profile and explanation](blog/2026-08-31-qwen-mtp-file-selection.md).

## Recorded matrix and node count

The 2026-08-31 user summary was checked against the original 2026-08-27/29
measurements, not rerun or treated as interchangeable workloads:

| Complete model | Nodes | Measured result | Qualification |
| --- | ---: | --- | --- |
| Qwen3.8 Flash Next NVFP4 + MTP | 4 | 48.81399 tok/s full-depth natural output; 262K retrieval | 261824 cached prompt tokens in this decode measurement; cold prefill measured separately above |
| Qwen3.8 Flash Next BF16 | 4 | 25.8662 tok/s short request | Cold 32768-input retrieval passed; 32769 failed with repeated token0. Smaller chunks and disabling radix did not fix quality |
| GLM-5.3 Flash FP8 | 4 | 20.162 tok/s, 512-token single stream | 240000-input retrieval passed; pool252352, not the advertised1M window |
| GLM-5.3 Flash FP8 MTP5 | 4 | 25.56635 tok/s forced512; 24.76421 natural256 | MTP acceptance37.75% forced; pool78528; 78000-input retrieval passed |
| Qwen NVFP4 + native MTP file view | 2 | 40.20 tok/s single; 102.19 aggregate4; strict262K passed | See dated 2026-08-31 result; not a matched throughput comparison to TP4 |

Original private evidence filenames are `qwen38-bf16-tp4-serving-20260829.json`,
`glm53-fp8-tp4-20260829.json`, and `glm53-fp8-tp4-mtp5-20260829.json`; public
sanitized matrix data preserves their model/runtime and measurement boundaries.
Use two nodes where complete-model correctness, capacity and performance allow;
do not run four solely to fill a matrix cell. Formal GLM-5.3 remains separate
and unpassed. Its ~465GB NVFP4 and ~405GB INT4/INT8 formats require more than
two128GB nodes for full accelerator residency before runtime/KV overhead.

## Formal GLM-5.3 INT4/INT8 preflight, 2026-08-31

All four actual GPU ranks completed the synthetic router comparison and the
full 78-layer FakeTensor constructor audit. This is **not full-model inference**:
no checkpoint weights were loaded and no serving throughput was measured.
The pinned model revision is `206507bbb047d8223964a0414cd83230c59428f9`.

| Constructor | Unique registered tensor storage per rank | Scope |
| --- | ---: | --- |
| Existing BF16 router | 94.43058 GiB | Full 78 layers, no MTP, no actual large allocation |
| FP32-router candidate | 94.65031 GiB | Adds 225 MiB; same model/quantization |

The packaged GateLinear's BF16 raw logits differed from GPU FP32 F.linear:
top-8-set agreement was 100%, 90.625%, 97.65625% for synthetic 1/32/128-token
batches. Setting both parameter/output dtype to FP32 made outputs bitwise
identical to the FP32 reference on every rank. This does not establish actual
checkpoint quality, grouped/bias-corrected expert choices or overall accuracy.
The run selected the existing Marlin compressed-tensor linear and MoE methods.

Static tensor accounting excludes checkpoint staging/repacking, communication,
KV, workspaces and allocator overhead. Actual small-tensor PyTorch peak was
37.50 MiB/rank, not 94.65 GiB of resident model. Full loading and inference remain
unverified. Earlier RDMA container-permission and A/B registration failures
were fixed only in disposable test code; no host configuration was installed.
See [sanitized all-rank results](../benchmarks/glm53-intmix-router-capacity-2026-08-31.json).

Checkpoint preparation subsequently passed at18:13UTC: native HF verification
matched all292upstream files; exact282safetensors total405,241,870,672bytes,
index coverage and revision marker passed before atomic final-directory rename.
Independent final-directory/marker checks passed18:29UTC; the temporary verifier
Job and Pod were removed18:30UTC without deleting the requested weights.
The verification log's587extra-local-file warning remains in the private record;
this is not a claim that every local cache file is upstream content. The
11,466.13-second shared-storage checksum pass is not model loading time or a
controlled filesystem bandwidth benchmark. Actual full INT4/INT8 loading and
inference still remain unverified; NVFP4 stays the first-priority target.

## Formal GLM-5.3 INT4/INT8 first real load, 2026-08-31

The complete verified checkpoint entered real TP4 loading at21:05UTC using the
fixed FP32-router image, native compilation and Marlin sparse-attention stack.
All four NCCL Mesh ranks joined. The first fastsafetensors producer batch then
requested a4.65GiB whole-file GPU buffer after94.91GiB Torch allocation and
failed before any completed shard batch or API readiness. The97GiB test cap
cannot contain those allocations together. CUDA also reported2.84GiB free;
raising the cap alone is not a proven solution. This allocation failure does
not measure or establish a SeaweedFS bandwidth bottleneck.

Native load calls lasted5.046/11.647/11.580/11.598seconds until exception;
these are **failure timings, not full model-load times**. Parent-process
lifetimes were45.86-58.51seconds. Minimum host MemAvailable was13.80-14.66GiB.
No model request succeeded and no tokens/s measurement exists for this run.

All four exited containers and the test ConfigMap were removed, full logs were
preserved privately, and original four-rank DeepSeek inference/authentication
passed internally and publicly at21:18-21:19UTC. Nodes retained their boot IDs,
SSH/services/Leases and four storage registrations. Original-service startup
emitted driver allocation warnings21:17:24-21:18:15Z; no Xid/Linux OOM or new
warnings after readiness were observed. This remains a warning-qualified
`failed-restored` result, not formal-model success or NVFP4-weight evidence.

The next bounded hypothesis selects the same runtime's native safetensors
per-tensor CPU reader instead of whole-file GPU staging, with model, kernels,
compilation and safety limits unchanged. It is not a validated optimization
until measured. [Exact failed-run data](../benchmarks/glm53-intmix-first-load-2026-08-31.json).

## SeaweedFS ConnectX storage measurements

With rank 2 offline, the remaining three-node storage path wrote a temporary
4 GiB file through the existing CSI/FUSE mount in 3.135 seconds. Simultaneous
first reads from ranks 1 and 3 completed in 2.581 and 2.558 seconds,
or about 1.55 and 1.56 GiB/s. The mount used direct volume-server `publicUrl`
addresses on the OSPF-routed ConnectX network, `cacheCapacityMB=0`, and no host
cache drop, writeback cache, storage replica, or tuning daemon.

Both readers consumed all 4,294,967,296 bytes. Their BusyBox wrappers then
exited non-zero because BusyBox `date` did not supply the requested nanosecond
field; the `dd` measurements themselves completed successfully. The file and
all disposable Pods were removed. This is a degraded three-node storage result,
not a four-node inference pass, but it shows that raw SeaweedFS/FUSE bandwidth
was several times higher than the measured model-loader throughput.

Machine-readable input:

- [`seaweedfs-connectx-2026-08-28.json`](../benchmarks/seaweedfs-connectx-2026-08-28.json)
