# Runtime image

Target package:

`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:int4-int8mix-20260824`

The source-build `Dockerfile` is ARM64-oriented and pins both build and runtime
bases by digest. It builds NCCL and the Mesh plugin at the exact tested commits.

The first package uses `Dockerfile.package` with the already four-node-tested
runtime layer. Packaging verifies the recorded SHA-256 values before copying
the libraries into the image. This avoids compiling on inference nodes. Both
paths apply the loader change only when the upstream source file matches the
tested SHA-256.

Publication status: pending (`write:packages` authorization required)

Published digest: pending

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

GHCR publication: blocked. Authentication succeeded, but the current GitHub CLI
token lacks the registry's required `write:packages` scope. The temporary Docker
credential directory was deleted after the refused push.

Same-hardware full-image inference smoke: pending. The exact base image, patches,
Mesh runtime and selected profile have each passed in the active four-rank run;
this does not substitute for launching all four ranks from the assembled image.

## Formal GLM-5.3 NVFP4 package

`Dockerfile.sglang-glm53` pins the exact SGLang runtime and Mesh build bases by
digest. It installs the already exercised `fastsafetensors 0.3.3` release and
applies two exact-source-hash guards: process-local CUDA device selection and
allocator release only after a completed file batch closes. These are image
changes only; the recipe installs no host daemon, timer, cache-drop loop, swap,
or recovery service.

Target image:
`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:glm53-nvfp4-loader-20260829`

Repository validation: passed

ARM64 image build: passed at `1642ff5ea03bd7bc51f5ad00e6cb779cb5540b24`.
Image config ID: `sha256:dd2c0dc94073644d6fe4455e1b80ebac4cd64b620b9f93da9b81da1e8e95ded9`.
Identical image imports and source/library hashes passed on all four nodes.

GHCR publication and digest: pending

Four-rank full-checkpoint inference smoke: **failed, unsafe; zero requests**.
All ranks reached Mesh initialization and weight loading, but available host
memory collapsed before any rank completed loading. The test was removed and
the nodes recovered; the preceding service remains paused for active diagnosis.

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
