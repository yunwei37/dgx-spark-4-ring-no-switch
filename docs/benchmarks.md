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
