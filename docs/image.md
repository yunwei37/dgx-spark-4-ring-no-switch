# Runtime image

All images are `linux/arm64`, built from digest-pinned bases, and published
to `ghcr.io/yunwei37/dgx-spark-4-ring-no-switch` by the `workflow_dispatch`
workflows under `.github/workflows/` (the repository push token has no
`write:packages` scope; the workflows use the job's `GITHUB_TOKEN`). Every
workflow also pushes a `<name>-<source-sha>` alias next to the documented
content tag, and records the digest in its run summary. Publication status
per tag:

| Tag | Workflow | Status |
| --- | --- | --- |
| `deepseek-v41-flash-20260911` | `build-deepseek-v41-flash.yml` | published, `sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c` |
| `qwen38-flash-next-nvfp4-20260827` | `build-qwen38-nvfp4.yml` | published, `sha256:5823a60b6b5ce2cd927c1881bb9c4bf17ae8400f092047f260648430b34f4422` |
| `glm53-nvfp4-loader-20260829` | `build-glm53-nvfp4-loader.yml` | published, `sha256:f5f8d9e9dc4925bd6f4fea6b0ba4ac36f31becbbc2dedc2b71f501e1d6276331` |
| `int4-int8mix-20260824` | `build-int4-int8mix.yml` | published, `sha256:de17ff1d442fe41d461bf23c7d11213ac54afe7cc52a5f42a9489d434ec4db33` (source-`Dockerfile` CI build; the locally assembled package ID below is separate component evidence) |
| `glm53-intmix-router-<sha>` | `build-int4-int8mix.yml` (second stage) | published at `glm53-intmix-router-adb99cf6b93b1a1b2666e7a2f72a40536d875190`, `sha256:11f0b8cc4484a672115796e845af17e52b3f671f3ca56974c31739d157b5b09d` |
| `glm53-intmix-nvfp4-dcp4-<sha>` | `build-glm53-intmix-nvfp4.yml` | published |
| `glm53-intmix-nvfp4-dflash2-<sha>` | `build-glm53-intmix-nvfp4-dflash2.yml` | published |

## DeepSeek-V4.1-Flash runtime (newest)

`Dockerfile.deepseek-v41-flash` reproduces the tested serving image chain from
the pinned MIT recipe `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark@592540c`:
the vLLM day-0 image at digest
`sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58`,
then overlay3 (FlashInfer 0.7.0rc1 `07869c61` with pinned cutlass/cccl/spdlog),
overlay4 (`mxfp8_gemm_cutlass_sm120` prebuilt for sm_121a) and overlay5
(`sparse_mla_sm120` rebuilt under the launcher's exact JIT environment via the
vendored `images/runtime/deepseek41/prewarm5.py`). The recipe's overlay3
prewarm is intentionally not repeated: overlay5 rebuilds that module under the
runtime environment, which is why the recipe introduced overlay5. The seven
recipe patch files and the `images/runtime/deepseek41/` serving files stay
bind-mounted exactly as [`deepseek-v41-flash.md`](deepseek-v41-flash.md)
records; no serving patch is baked in.

Target image:
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:deepseek-v41-flash-20260911`

Published digest:
`sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c`
(revision `9a07e93`, ARM64, `ubuntu-24.04-arm` runner)

Build notes: overlay3 completed in 96 s on the ARM64 runner. The first run
lost its runner while nvcc compiled `mxfp8_gemm_cutlass_sm120` (overlay4) —
disk was fine at 109 GiB free, so the 16 GiB runner RAM is the limit, and the
recipe itself documents that compile wedging memory on real nodes. The
workflow now adds a 48 GiB build swapfile; the successful run completed in
about 16 minutes. CI has no GPU, so overlay5's load step raising without one
is expected and recorded by the script; the GPU-side verification ran on the
fleet during the trial.

This publication reproduces the image recipe. It does not add any new
inference claim beyond the trial record.

Target package:

`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:int4-int8mix-20260824`

The source-build `Dockerfile` is ARM64-oriented and pins both build and runtime
bases by digest. It builds NCCL and the Mesh plugin at the exact tested commits.

The first package uses `Dockerfile.package` with the already four-node-tested
runtime layer. Packaging verifies the recorded SHA-256 values before copying
the libraries into the image. This avoids compiling on inference nodes. Both
paths apply the loader change only when the upstream source file matches the
tested SHA-256.

Publication status: published by `build-int4-int8mix.yml` from the source
`Dockerfile` as `int4-int8mix-20260824` at digest
`sha256:de17ff1d442fe41d461bf23c7d11213ac54afe7cc52a5f42a9489d434ec4db33`
(revision `adb99cf`). The locally assembled package recorded below is separate
component evidence, not this registry digest.

Repository validation: passed

Image build: passed on ARM64 at source revision `db05308`

Local image ID: `sha256:d58701574065dc23b5bf246fc98c974ca5fbd40e0f16ff4972db4eb05e244746`

Uncompressed image size: 19,456,486,051 bytes

Static package checks: passed (vLLM and fastsafetensors versions, both runtime
library hashes, and exactly one loader cache-release insertion)

Inference smoke with this assembled image: pending

Until the last item passes, benchmark numbers in this repository describe the
tested components at their original immutable image digests. They do not prove
that this newly assembled package serves the model correctly.

## Qwen3.8 Flash Next NVFP4 package

`Dockerfile.sglang-qwen38` defaults to reproducibly building NCCL and the Mesh
plugin from the pinned source commits. A builder may instead set `MESH_IMAGE`
to an already verified image containing `/opt/nccl-mesh`; this skips redundant
package-index access and compilation without changing the copied runtime files.
The SGLang runtime base remains pinned by digest, and both compatibility patches
refuse an unknown source hash.

Target image:
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:qwen38-flash-next-nvfp4-20260827`

Source revision: `100984d5e44a181ebac3e312a5dc807c5b1606bf`

Local ARM64 build: passed

Local image ID: `sha256:c13c7345095ad010c0b11e68e36707ea9cb109427171e2ebb92d4b7cf5cd50c0`

Uncompressed image size: 30,409,004,170 bytes

Static package checks: passed (both NCCL Mesh runtime hashes, all four patched
source hashes, Python compilation, architecture, revision label)

GHCR publication: published by `build-qwen38-nvfp4.yml` as
`qwen38-flash-next-nvfp4-20260827` at digest
`sha256:5823a60b6b5ce2cd927c1881bb9c4bf17ae8400f092047f260648430b34f4422`
(revision `ce15f56`, `sbom: false`). The direct push path stays closed because
the repository push token has no `write:packages` scope; the workflow's job
`GITHUB_TOKEN` publishes instead. The first run solved the image and then
failed only at the SBOM attestation (`sbom.spdx.json exceeds 41943040
bytes`), so the re-dispatch builds the same content with `sbom: false`.

Assembled-image TP2 inference: passed with this exact config ID in the native
262K test on 2026-08-28 and the MTP file-view test on 2026-08-31. The latter
passed strict final-key retrieval and measured 40.20/102.19 tokens/s single /
four-concurrent synthetic completion. It does not prove assembled-image TP4,
production stability, or GHCR publication. See the dated benchmark records.

## Formal GLM-5.3 NVFP4 package

`Dockerfile.sglang-glm53` pins the exact SGLang runtime and Mesh build bases by
digest. It installs the already exercised `fastsafetensors 0.3.3` release and
applies two exact-source-hash guards: process-local CUDA device selection and
allocator release in `ParallelLoader` after a completed file batch closes.
Later call-path inspection found that this pinned SGLang iterator uses
`SafeTensorsFileLoader` directly, so the `ParallelLoader` cache patch is not
active on this GLM-5.3 path and is not evidence of memory protection. These are image
changes only; the recipe installs no host daemon, timer, cache-drop loop, swap,
or recovery service.

Target image:
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:glm53-nvfp4-loader-20260829`

Repository validation: passed

ARM64 image build: passed at `1642ff5ea03bd7bc51f5ad00e6cb779cb5540b24`.
Image config ID: `sha256:dd2c0dc94073644d6fe4455e1b80ebac4cd64b620b9f93da9b81da1e8e95ded9`.
Identical image imports and source/library hashes passed on all four nodes.

GHCR publication: published by `build-glm53-nvfp4-loader.yml` as
`glm53-nvfp4-loader-20260829` at digest
`sha256:f5f8d9e9dc4925bd6f4fea6b0ba4ac36f31becbbc2dedc2b71f501e1d6276331`
(revision `ce15f56`, after the same SBOM-attestation size limit as the Qwen
package). The recorded local build evidence above predates this registry
publication and stays separate from it.

Four-rank full-checkpoint inference smoke: **failed, unsafe; zero requests**.
All ranks reached Mesh initialization and weight loading, but available host
memory collapsed before any rank completed loading. The test was removed and
the nodes recovered; the preceding service was paused during diagnosis and
subsequently restored as documented at the end of this section.

The first actual ARM64 build on 2026-08-31 installed 0.3.3 successfully but
found that the base uses a Python virtual environment, not `/usr/local/lib`.
The package patch now resolves the installed distribution's file location
instead of assuming a system installation path. The exact unmodified and
patched SHA-256 checks are unchanged. This packaging failure happened before
any model load and is not an inference result. A second real build exposed a
stale expected output hash. An isolated container diff verified that the only
changes to the exact original source are `import torch`, two explanatory comment
lines and `torch.cuda.empty_cache()` after `batch.fb.close()`. The expected
patched hash is now the measured `5bed0a36...39152`; the original source guard
remains unchanged. No check was removed to make the build pass.

The 2026-08-31 assembled-image run repeated the memory failure; prebuilding and
completed-batch release alone were insufficient. This is distinct from the
2026-08-29 attempt, and neither is an inference pass.

### CUTLASS constructor scale duplication

Inspection of the pinned runtime found two unused swizzled scale placeholders
allocated for every NVFP4 MoE layer before any checkpoint tensor loads. Normal
post-load processing already recomputes those values and aliases them to source
scale storage. `apply_sglang_nvfp4_deferred_scales.py` extends the existing
TRTLLM deferral to CUTLASS only; it changes neither kernels nor post-load logic.
The exact upstream file hash must match before applying the image-local patch.

A bounded GPU A/B measured **144 MiB saved per local 64-expert layer**, hence
**10.546875 GiB per rank** across the full model's 75 MoE layers. Final scale
hashes, aliases and synthetic packed weights matched. See
[the diagnostic record](../benchmarks/glm53-nvfp4-constructor-2026-08-31.json).
The single-layer test is not a full-model or kernel-output correctness claim.
The image containing this additional patch still needs full allocation sizing,
real TP4 inference and stability validation; no corrected serving result or
GHCR publication is claimed.

The corrected ARM64 build passed locally with image config ID
`sha256:f5e0b9c3117483bdac1d81a527757fc5026d52ae8d2fa2398fd4862569a78531`.
The committed one-layer A/B also passed inside that assembled image, repeating
the recorded allocation and equivalence results. For the next sizing step,
`tests/Dockerfile.glm53-shape-audit` creates a disposable diagnostic-only image:
the real model constructor runs under PyTorch FakeTensor and then deliberately
exits before weight loading. Its 5% Torch allocator cap is test-local protection,
not a serving parameter or host policy. Any unsupported fake operation is a
diagnostic failure; it must not fall through to allocating the full model.

The first shape-audit run failed safely at the logits processor's CPU host
membership query (`FakeTensor.tolist` needs real values). No full weights were
allocated or loaded, and GPU processes exited normally. The diagnostic now
queries the real process group's membership before entering FakeTensor and
reuses that measured result for the same group. It does not assume co-location
or change the serving image. This diagnostic failure is retained separately
from the actual model-memory failure.
The first adjustment referenced the caller module instead of the function's
definition in `parallel_state`; that second diagnostic also exited before
model allocation. Source inspection confirmed the caller uses a local import,
and the wrapper now patches the defining module only within its scoped test.

All four corrected shape audits subsequently agreed on 106.032193 GiB of
unique registered model tensor storage per rank. This excludes runtime and
staging allocations. A test-local `NCCL_MAX_NCHANNELS=8` candidate reduced
logged distributed-initialization memory from about 2.23 GiB to 0.71–0.73 GiB
without changing the tensor inventory. Communication throughput is not yet
measured. EP=1 gave no communication-memory benefit and changed the shared
expert storage inventory; its apparent saving is not an established
precision-equivalent optimization and is not selected.

`tests/Dockerfile.glm53-bounded-load` is the next **experimental**, not approved
serving image. It applies a native 107.5 GiB Torch allocator limit before the
real constructor, requires at least 3 GiB outside that budget at entry, and
logs actual allocations. The cap does not cover CPU or non-Torch allocations.
The disposable Job also has a 112 GiB cgroup limit and a 30-minute deadline.
The candidate retains TP4/EP4, full checkpoint precision and native kernels;
it uses the upstream safetensors mmap loader, eight NCCL channels, an initial
8K context/token budget and 512-token chunked prefill. This is a capacity
bring-up, not a one-variable throughput A/B or the final maximum-context
profile. No host configuration or persistent monitoring process is installed.

The bounded image built successfully with config ID
`sha256:06f5229f99788c293318427968224574455230f63d2f45711ff6461da7665c85`
and was imported on all four nodes. A native allocator sanity test rejected a
2 GiB request under a 1 GiB budget without allocating any memory. The real TP4
Job then encountered a scheduling conflict: an unrelated workspace's 16 GiB
reservation left one rank Pending. The other three stopped before constructor
allocation; all test Pods were removed. This is **resource-blocked**, not a new
loader OOM, not corrected inference success. The unrelated workspace was not
stopped or resized.

A native PP4 capacity-only follow-up distributed all 78 layers as 22/19/18/19,
with BF16 shared experts retained. FakeTensor storage was 107.752329,
103.757127, 98.313978 and 105.512142 GiB by pipeline stage. These numbers did
not establish safe runtime margin alongside the other reservation. No real PP
weights were loaded. This pinned runtime also excludes speculative decoding
with PP, so the diagnostic is not a replacement for the requested TP4 result.

The attempt ended **failed-restored**: the original four-node DeepSeek service
was restored with its unchanged image, arguments and data, and passed internal
and public authenticated generation plus missing/invalid-key checks. Its GitOps
owner converged successfully. All GLM test Jobs and temporary transfer files
were removed; the requested checkpoint and reproducible experimental images
were retained. Corrected formal GLM-5.3 inference remains unmeasured, with zero
successful requests. The 16 GiB reservation belongs to separate work and was
not altered to make the experiment fit. A confirmed exclusive-memory window is
still required; do not present these experimental images as a working service.

## Formal GLM-5.3 INT4/INT8 router candidate

The old package ignores `moe_router_dtype=float32`. The four-rank synthetic GPU
preflight proved that FP32 parameter/output selection matches the reference;
the full-checkpoint correctness and memory gates remain outstanding.
`images/runtime/apply_glm53_router_fp32.py` adds exactly those two constructor
kwargs when the checkpoint requests float32, preserving unspecified behavior.
It changes neither quantization nor the model's layers/experts/IndexShare.

On2026-08-31 the corrected module was assembled into an experimental ARM64 OCI
image on a managed non-Spark container, then imported and inspected on all four
inference nodes. All25base layers stayed unchanged. No compiler, host source
tree, driver or daemon was installed. Candidate identity:

```text
local tag: ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:glm53-intmix-router-d6cf87377e53
config: sha256:71ef93e62e5ad7f76821f25632e2df9c3840f1bcb1df1ad268b8618a40031ab2
manifest: sha256:65cfb227b40f92709bed00d49bb502772ad6948ba31dbaf3b09fbbb3b6c41a3d
patched source: d6cf87377e530c878b7f4a12b1517c08ba4213f320caa337b5e9c1462db4e24e
GHCR publication: published by the second stage of build-int4-int8mix.yml as
  glm53-intmix-router-adb99cf6b93b1a1b2666e7a2f72a40536d875190 at
  sha256:11f0b8cc4484a672115796e845af17e52b3f671f3ca56974c31739d157b5b09d,
  built from Dockerfile.vllm-glm53-intmix over the published
  int4-int8mix-20260824 base; the locally assembled candidate above remains
  a separate, recipe-unverified record
full-checkpoint inference smoke: not run
```

The51,200-byte OCI increment reused already cached base layers; its0.2–0.3s
imports are **not** full-image installation timings. The14,489-byte compressed
layer contains only the corrected module. Source compile and AST kwargs checks
passed; the build-time patch script also patched a real temporary source file
to the expected hash and refused mismatched input without changing it. The
temporary directory was removed. These checks are not runtime/GPU inference.

`Dockerfile.vllm-glm53-intmix` is the conventional build recipe using the exact
locally assembled base tag from `Dockerfile.package`. This particular image
was built by OCI assembly, not by executing that Dockerfile, so recipe execution
is not yet verified. The existing base and candidate tags are not presently
claimed pullable from GHCR. There is no startup-time patch/install step in the
assembled candidate. Do not publish it as a working GLM service before the
complete checkpoint, capacity, correctness and throughput gates pass.
