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

The immutable SGLang image needed source-hash-guarded, test-local compatibility
fixes for gated-MoE TP padding and for the QSA sparse-decode architecture gate.
The latter lets SM121 use the already-present TRT-LLM sparse decode kernel rather
than the broken FlashAttention-4 CuTe fallback. No host daemon, cache-drop loop,
swap setting, route, or management-network change was added.

Machine-readable input:

- [`qwen38-flash-next-nvfp4-2026-08-27.json`](../benchmarks/qwen38-flash-next-nvfp4-2026-08-27.json)
