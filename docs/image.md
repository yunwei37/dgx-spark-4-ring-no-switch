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

ARM64 image build: pending

GHCR publication and digest: pending

Four-rank full-checkpoint inference smoke: pending

The first actual ARM64 build on 2026-08-31 installed 0.3.3 successfully but
found that the base uses a Python virtual environment, not `/usr/local/lib`.
The package patch now resolves the installed distribution's file location
instead of assuming a system installation path. The exact unmodified and
patched SHA-256 checks are unchanged. This packaging failure happened before
any model load and is not an inference result.

The earlier four-rank run proved NCCL Mesh startup and reached ModelOpt FP4
loading, but exhausted unified memory before serving a request. It does not
validate this assembled image. Publication and serving results must remain
pending until the full image is built and a request completes without a node,
SSH, kubelet, or storage regression.
