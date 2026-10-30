# NCCL 2.30.7 switchless-ring candidate

This is an experimental transport candidate, not a production throughput result.
The retained baseline is NCCL 2.29.7/Mesh. A bounded full-service candidate trial
was restored to that baseline after an application logprob normalization crash.

## Sources and minimal fix

- NVIDIA NCCL v2.30.7-1: 73cf112295c33aee2b895f329f592f2a9b4b0f97.
- Cumulative dual-PCI/subnet patch: FujitsuPolycom/sparkring,
  9932204d25ab857ec4a5a7c437f4dc0f5bddee2e,
  spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch.
- CTS acquire fence: NVIDIA/nccl#2393, as carried by the SGLang recipe
  knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 at
  cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa.
- Apply nccl-listener-physical-devices.patch after the cumulative routing patch.
  NIC Fusion keeps physical devices and adds merged aliases. Enumerating both
  advertised five entries for four PFs and fell back to the legacy two-GID
  format. Enumerate ncclNIbDevs exactly once instead. Real four-rank logs changed
  from legacy/2/5 to ipv4-v1/4/4; both PCI roots then carried bytes on every rank.

Build with the target CUDA SDK and SM121. The measured binary used CUDA 13.0,
Ubuntu 24.04 AArch64 ABI and the standard NCCL src.build target. The host cross
build used CXX=aarch64-linux-gnu-g++ and target CUDA_LIB; a native ARM64 build
uses the corresponding native compiler. No driver or kernel change is required.
The fence was verified as dmb ishld in ncclIbIsend.

## Measurement boundary

nccl_collective_bench.py uses the serving image's SGLang PyNcclCommunicator.
Four ranks check out-of-place BF16 AllReduce and AllGather at 1, 6, 24 and
96 rows with hidden size 5120. Each result has eight blocks of 32 operations,
after eight warmups, measured with CUDA events. Each trial checks all 32
rank/operation/shape results and records per-HCA byte counters.

Use one NCCL library. PyTorch explicitly preloads its wheel directory, so
LD_PRELOAD alone leaves two different NCCL libraries mapped. Candidate tests
bind the candidate lib directory over that wheel path in a private child mount
namespace, with private propagation, before starting Python. The script records
/proc/self/maps and verifies the selected library inode. The parent service
mount and library remain unchanged.

Rank zero uses the existing native pause_generation mode=in_place and always
continues generation in finally. This preserves active requests and KV state.
The test creates no serving replica, route, persistent daemon, or credential.
These eager collective timings do not establish end-to-end token throughput.

## Results, 2026-10-30 UTC

Maximum of four rank medians, microseconds. All four trials below passed all
32 numerical checks and resumed generation. Sources, raw timing samples and
non-secret routing evidence are retained privately; the companion aggregate
contains no host identities, network addresses, prompts or credentials.

| Operation / rows | Mesh 2.29.7 D1 | Native 2.30.7 E1 | Physical-PF fix F1 | Fix + max channels 4 G1 |
| --- | ---: | ---: | ---: | ---: |
| AllReduce / 1 | 101.18 | 38.10 | 37.86 | 38.29 |
| AllReduce / 6 | 138.86 | 49.36 | 49.31 | 53.66 |
| AllReduce / 24 | 840.00 | 116.78 | 109.60 | 79.40 |
| AllReduce / 96 | 867.92 | 1104.79 | 1095.53 | 155.91 |
| AllGather / 1 | 65.74 | 23.82 | 20.17 | 20.18 |
| AllGather / 6 | 292.55 | 32.15 | 32.19 | 31.78 |
| AllGather / 24 | 1507.25 | 55.39 | 54.41 | 48.50 |
| AllGather / 96 | 532.98 | 323.60 | 327.23 | 82.76 |

D1 used the existing Mesh profile. E1 switched to the pinned native transport
with switchless-ring, subnet-aware, extended IPv4 GIDs and PCI-domain
preservation. F1 changed only physical device enumeration. G1 changed only
NCCL_MAX_NCHANNELS=4, matching the pinned dual-PCI recipe's channel recommendation;
the native default created 64 channels. All G1 ranks used both PCI roots.
This flag is still an experimental candidate until a matched serving A/B
establishes benefit and stability. CROSS_NIC=0 was an earlier failed hypothesis
for the routing issue and is not retained.

Earlier B2/C1 mixed-library tests are preliminary and superseded by these
single-runtime trials. Full speedbench, concurrent serving, quality, stability,
and real 1M acceptance for this transport remain unmeasured.


## CUDA Graph qualification and first serving attempt

H1/I1 repeated the operations in captured 32-collective blocks on a side stream.
All 32 checks passed per trial and generation resumed after 3.59/1.97 seconds.
Max-rank median microseconds:

| Operation / rows | Mesh H1 | Physical-PF fix + max channels 4 I1 |
| --- | ---: | ---: |
| AllReduce / 1 | 106.78 | 44.45 |
| AllReduce / 6 | 132.73 | 63.14 |
| AllReduce / 24 | 843.38 | 92.32 |
| AllReduce / 96 | 874.95 | 171.54 |
| AllGather / 1 | 97.82 | 26.85 |
| AllGather / 6 | 236.94 | 39.81 |
| AllGather / 24 | 1913.88 | 58.82 |
| AllGather / 96 | 754.10 | 92.89 |

The experimental carrier is published as
ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:nccl-2.30.7-ring-physical-20261030a
with manifest sha256:8b010622708a66f0263635653c5ca26a4edada3eb74fae79541e0f9db9b55468.
The existing publisher succeeded and all four machines pulled the ARM64 image.

The first full-service attempt reached Ready and loaded exactly one 2.30.7
runtime per process. Its quality pilot then crashed Python normalization:
a token-ID logprob row was already a list, but the result processor called
tolist unconditionally. No speedbench result was produced and no NCCL error
was observed. The baseline transport was restored. Treat this as a failed
application acceptance attempt, not proof of either transport regression or
end-to-end improvement.

The native-runtime.patch now normalizes the documented list-or-Tensor union
at both prefill and decode consumers. A regression invokes the actual
token-ID producer and both consumers, including an empty requested-ID row.
All four eager/delayed consumer cases failed before the fix and pass after it;
the full ten-test CPU MMLU protocol suite passes. These tests do not establish
GPU/model quality acceptance. Rerun the real pilot before the serving A/B.

## Matched serving results and next prefetch candidate (2026-10-30)

Both transport arms used the same corrective SGLang image 20261030d,
official October checkpoint 1bae0e1, frozen 73-prompt speedbench protocol,
thinking off, temperature zero and 256 output tokens. Clients used the same
direct private head endpoint during a drained maintenance window.

| Measurement | Mesh / d | Native / d | Native + prefetch / e |
| --- | ---: | ---: | ---: |
| C1 ALL decode geometric mean | 66.92 | 76.86 | 78.16 |
| C1 code decode geometric mean | 85.74 | 95.91 | 97.18 |
| C16 whole-level block tok/s | 185.76 | 205.41 | 202.15 |
| C16 window aggregate tok/s | 191.52 | 213.98 | 209.98 |
| Cold 64K prefill tok/s | 5406.3 | 6055.3 | 6118.8 |
| 114-context frozen MMLU pilot, correct | 104 | 102 | 102 |

All 73 decode measurements, four sentinels and twelve long-context probes were
valid in both arms, with zero request failures. Correction: upstream's 211 is whole-level block throughput, including tail
drain, and cannot be compared to the window-only aggregate of 213.98. The raw
block values are 205.41 for native d and 202.15 for e; neither meets 211. The
191.28-235.30 interval belongs to the window metric, not the block metric. C1 ALL/code still miss 89.3/112.2. Four pilot choices
changed. The full frozen 14,042-context evaluation subsequently completed:
native 12,073 correct (85.9778%) versus reference 12,077 (86.0063%), with 387
changed choices. There were 157 reference-only and 153 native-only correct
answers; exact McNemar p=0.86474. This test did not find a significant accuracy
difference, and does not establish identical outputs or universal quality. Native near-1M retrieval recovered
all three needles with 1,039,058 input tokens and 13 output tokens in 196.342 s.

A subsequent 32-step CPU/GPU trace captured mixed consumer/evaluation traffic;
it is not an isolated C1 performance trial. Joining kernel correlation IDs to
CUDA graph launches and their CPU stage annotations separated eleven bs=1
verify replays per rank. Median graph windows were 43.59-43.71 ms and summed
NCCL kernels 7.25-7.98 ms. Rank 0's dominant per-verify kernel sums were routed
MoE phase1 12.27 ms, phase2 5.75 ms and two dense FP8 families 7.52/2.88 ms.
Kernel sums can overlap and are not independent critical-path fractions.

The pinned upstream recipe explicitly notes that its L2 prefetch adapter learns
no collective plan when RoCEnante is disabled
([switchless-ring.md](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa/docs/switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths)).
The source candidate extends that same read-only, learned-weight prefetch to the
native PyNcclCommunicator in-place/out-of-place AllReduce and AllGather paths.
It reuses the existing main prefetch gate and stream resolver, leaves disabled
communicators and RoCEnante unchanged, and preserves original calls/results.
No new tuning flag, cache, controller or network path is introduced. CPU
contracts pass; real CUDA-graph operation, identical outputs and matched
performance remain required before deployment or an improvement claim.

The CPU-qualified prefetch image is published as
ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:sglang-cfe3cfb-production-20261030e
(manifest 467fa04d7642504a3475143779ddf43f97afbddcdbcfc12ad3170ff7befb1615).
[Owned publisher run](https://github.com/yunwei37/dgx-spark-4-ring-no-switch/actions/runs/41075016730)
succeeded. All four serving hosts preloaded it; the additional image layer is
71,680 bytes and changes only the two adapter modules and provenance above
corrective image d. Publication and preload do not establish GPU acceptance.

The e trial completed all 73 decode samples with zero errors and all twelve
long-context probes; cold 4K prefill was 5455.0 tok/s and 64K 6118.8 tok/s.
Actual bs=1 graphs learned 125 collectives and captured 162 prefetch branches.
All 114 pilot score vectors exactly matched native d. Two longer controls
did not retain identical content; the older controls also encountered renewed
backend activity, so neither byte preservation nor their timing difference
is established by that comparison. The small separate-session C1 gain
(1.7%) and lower C16 block result do not justify retaining the extension.
The bounded trial restores qualified image d; the source/image stay available
as research artifacts, not an accepted production improvement.

## Neighbour-only Tree candidate (not yet GPU qualified)

The current switchless patch explicitly skips Tree and PAT transport setup.
The standard four-node double tree has a degree-three internal vertex, which
cannot be embedded in a degree-two physical ring without a non-neighbour path.
The source candidate builds two complementary spanning paths from NCCL's
actual ring-next map before the device communicator is copied, including
runtime-connect mode. Each root has maximum depth two; internal-node sets are
disjoint and all edges join existing ring neighbours. Existing upstream Tree
device kernels, arithmetic and IB transport are reused. No OS route, hairpin,
firewall or new tuning flag is added.

The existing switchless gate selects this topology only for four nodes with
one rank per node. Other sizes retain the previous Tree skip, and gate-off
retains upstream topology. A host-only test compiles the actual helper,
upstream double-tree code and transport entrypoint: 96 permutation/channel
combinations and 384 rank cases check neighbour-only edges, reciprocity,
connectivity, depth, complementary internal nodes, invalid-cycle rejection
and unchanged gate-off behavior. This is not GPU or performance acceptance.

For a finite comparison, upstream's existing per-function algorithm syntax
allows Ring for other collectives and Tree for AllReduce:
NCCL_ALGO=Ring;allreduce:Tree
([NVIDIA documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo)).
A global Tree-only selection would leave AllGather without a supported algorithm.
No serving configuration has been changed to this candidate.

Before any new routing is trusted, the existing collective harness now checks
four base-31 index-digit patterns with different rank coefficients. Every
individual value and partial sum is an exactly representable BF16 integer.
This detects stale rounds and within-rank permutations that constant inputs
could miss. Real eager/CUDA-graph checks and latency A/B remain required.

The ARM64 candidate linked successfully in 5.134 s (61,423,968-byte library).
A fresh extraction of pinned NCCL plus the same routing/CTS/listener/tree patch
chain reproduced the CPU proof. The upstream routing patch writes generic.cc
with CRLF; the build normalizes only that file's line endings before the local
patch. Without that necessary preparation the patch demonstrably fails.
From this directory, run the host proof against a patched source tree with:

    NCCL_SOURCE_DIR=/path/to/patched/nccl python3 test_nccl_ring_tree.py

The nonconstant-pattern oracle also passed CPU checks over every rank addition
order and reconstructs all 491,520 element indices from its four rounds. None
of these host tests claims real four-rank GPU correctness or a latency gain.
