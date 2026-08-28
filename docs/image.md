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

Publication and same-hardware smoke status: pending.
