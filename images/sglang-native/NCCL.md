# NCCL 2.30.7 switchless-ring candidate

This is an experimental transport candidate, not a production throughput result.
The serving service remains on NCCL 2.29.7 and Mesh while this candidate is tested.

## Sources and minimal fix

- NVIDIA NCCL v2.30.7-1: 73cf112295c33aee2b895f329f592f2a9b4b0f97.
- Cumulative dual-PCI/subnet patch: FujitsuPolycom/sparkring,
  9932204d25ab857ec4a5a7c437f4dc0f5bddee2e,
  spark_transport/nccl/ (see pinned DUAL_PCI_DOMAIN.md and cumulative patch).
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
