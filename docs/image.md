# Runtime image

Target package:

`ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:int4-int8mix-20260824`

The Dockerfile is ARM64-oriented and pins both build and runtime bases by
digest. It builds NCCL and the Mesh plugin at the exact tested commits, copies
only their runtime libraries, and applies the loader change only when the
upstream source file matches the tested SHA-256.

Publication status: pending

Published digest: pending

Repository validation: pending

Image build: pending

Inference smoke with this assembled image: pending

Until the last item passes, benchmark numbers in this repository describe the
tested components at their original immutable image digests. They do not prove
that this newly assembled package serves the model correctly.
