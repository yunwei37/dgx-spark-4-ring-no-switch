# Formal GLM-5.3 community experiments

Observed 2026-08-31. This is a source audit, **not a local benchmark**. Formal
GLM-5.3 and GLM-5.3-Flash are different targets. NVFP4 weights and NVFP4 KV
cache are different settings. Our formal NVFP4 and INT4/INT8 inference results
remain uncompleted; see [our measured results](benchmarks.md).

## Closest published GB10 reproduction

[Tony's formal GLM-5.3 repository](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/tree/a1806cb82493aa6f28709f77acf59c1937bdf756)
was inspected at `a1806cb82493aa6f28709f77acf59c1937bdf756` (August29).
Its target is the full model using Tech2wild INT4/INT8 weights, not Flash.
The old Hugging Face `2wild4tv` URL redirects to `Tech2wild`; this is the
checkpoint we already verified, not another required 405GB download.

The [original forum report](https://forums.developer.nvidia.com/t/glm-5-3-743b-int4-int8mix-on-4x-dgx-spark-gb10-tp4-up-to-46-tok-s-200k-ctx/381755)
reports TP4 with MTP4: 12.12 output tok/s at concurrency1 and 46.03 total
tok/s at concurrency6. The title's 46 is not a single-user decode rate.

The newer [DFlash2 measurements](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/bench/RESULTS-dflash2.md)
use HTTP request-to-complete-response time, including prefill, not just decode:

| INT4/INT8 target, TP4 | Count-to-100 | Prose C1 | Prose C6 aggregate | Configured context |
| --- | ---: | ---: | ---: | ---: |
| MTP4, FP8 KV | 26.91 | 19.62 | 51.93 | 200K |
| DFlash2 k7, FP8 KV | 53.32 | 18.70 | 49.88 | 80K |

DFlash2's counting acceptance was 95.6%, but its prose acceptance was about
23%. Counting speed does not establish coding-agent speed, and the author
explicitly has not run the full quality evaluation.

The [NVFP4-KV plus DFlash2 report](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/bench/RESULTS-nvfp4-dflash2.md)
still uses **INT4/INT8 model weights**: 51.03 counting, 17.81 prose C1 and
50.98 prose C6 aggregate tok/s, with a 293447-token KV pool and configured
270K context. These pool/configuration values do not prove a 270K-input
retrieval test. Its pinned older implementation cannot combine this DFlash2
lane with DCP; that is not a universal limit of all future runtimes.

## A second report uses a switchless ring

[ajclark's August31 field report](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/2)
uses four GB10 nodes with a 200G direct ring. It reports 53.8 tok/s counting
at C1 and 192.6 aggregate at C12, and 2.9x over its own 10GbE baseline.
The author measured no prose, ran one sample per point, and supplied no error
bars. This is another operator's report, not our result or an assurance that
every ring outperforms a switch.

That report makes rank order, both port GIDs, subnet-aware routing and lazy
connection setup important for its native-NCCL route. A standalone collective
can pass while framework initialization still attempts unreachable edges.
Our Mesh path is different: do not combine all of its environment overrides
with ours without checking exact NCCL behavior. Do not change management
networking or clamp host clocks merely to follow the report.

## Reproduction gaps, not an available one-command package

The [launcher](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/launch/launch-glm53-tp4.sh)
uses a locally named `probe-modded` image and ten paired sm12x kernel overlays.
It also assumes a particular NFS layout, interface names, NCCL library and
cache-flushing process. Its vLLM build is
`0.23.1rc1.dev190+gab6660699`, related to our previously tested GLM-5.2 stack;
that alone does not prove identical image bytes. A fixed checkpoint path is
insufficient to reproduce the complete stack.

[Issue1](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/1)
identifies missing `dsa_block.py` and an unnamed draft checkpoint. A
[follow-up contribution](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/1#issuecomment-5473400663)
provides a reconstructed Apache-2.0 candidate, explicitly not the author's
original file, plus an available donor-image tag. Neither is locally validated.
[Issue3](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/3)
still requests a ready-to-download image after unsuccessful deployment attempts.
All three issues were open when inspected; no upstream messages were posted.

The [port analysis](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/dflash2-port/README.md)
reports a Flash-tested runtime producing incorrect output on the formal model,
even without speculation. A newer version or successful Flash test is therefore
not a substitute for correct formal-model sparse-attention kernels.

## What exists for actual NVFP4 weights

- [IncoAI's formal NVFP4 card](https://huggingface.co/incoai/GLM-5.3-NVFP4)
  publishes a 433GiB experts-NVFP4 checkpoint, quality comparisons and TP8
  SGLang/vLLM examples. It is useful evidence that the format can work, but not
  evidence of four-Spark fit. Another similarly sized checkpoint alone will
  not remove our measured allocation peak.
- [IncoAI's formal DFlash2 card](https://huggingface.co/incoai/GLM-5.3-DFlash2)
  benchmarks the full target on four **GB300**, not GB10. Its drafter is not a
  standalone model; its CC BY-NC-ND4.0 terms must remain separate from runtime
  code licensing when considering reuse or redistribution.
- [Ressl's card](https://huggingface.co/ressl/GLM-5.3-NVFP4)
  explicitly says this export has not been serving-validated; its PP7 recipe
  was proven on GLM-5.2 with seven RTX PRO6000 cards. Do not count it as a
  formal5.3/Spark success.
- [NVIDIA's Dynamo recipe](https://docs.dynamo.nvidia.com/dynamo/dev/recipes/glm-5-2)
  targets B200/H200. The performance section explicitly says GLM-5.2; it does
  not establish GLM-5.3 GB10 performance or justify adding Dynamo here.

No directly reproducible full **NVFP4-weight** four-GB10 success was found in
this search. That is a search result boundary, not proof that none exists.

## Consequences for our next experiments

1. Keep the requested actual NVFP4-weight test open. Our CPU-only pre-read
   ownership/sliced-vocabulary candidate is not GPU-tested and is not a pass.
2. Use the already verified full INT4/INT8 checkpoint as the community-backed
   comparison: validate the matching image/overlays, first correct base
   inference, then MTP, then DFlash2 if its extra artifacts justify the test.
   Do not call an eager diagnostic or counting-only run a successful developer
   workload. Preserve the complete architecture and router/indexer precision.
3. Measure cold start, real input length, TTFT, decode and HTTP end-to-end
   separately. Match prompts/concurrency/thinking/cache state for A/B. No
   published source inspected here proves a one-minute full cold start.
4. INT4/INT8's approximately405GB and NVFP4's approximately465GB checkpoints
   exceed two128GB nodes before runtime overhead. Four nodes are required for
   these resident-weight paths. Offloading or lower-bit weights would be a
   different experiment, not a hidden way to claim the same two-node result.
5. Do not import periodic drop-cache, host swap tuning, clock overrides or
   sub-GiB memory margins as fleet defaults. If a bounded experiment needs a
   candidate, record its actual benefit, management health and cleanup first.

Research did not pause our original service, download duplicate weights,
install a host mechanism or publish an untested serving image.
