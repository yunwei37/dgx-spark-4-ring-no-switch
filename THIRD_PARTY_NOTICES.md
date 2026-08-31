# Third-party notices

This repository does not redistribute model weights.

The INT4 runtime image is built from:

- `ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x` at digest
  `sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481`.
  The associated public recipe repository did not declare a repository-level
  license when this artifact was recorded. This project does not infer one;
  downstream users must review the image contents and upstream terms.
- NVIDIA NCCL source at commit
  `b91894bd5b190c874d98a017f93f5daa515b65d0`, under its upstream license.
- `autoscriptlabs/nccl-mesh-plugin` at commit
  `19924dcc7c571d6e260953724d394ae50bad82cf`, under the MIT license declared
  by that source repository.

The tested NVFP4 result used
`ghcr.io/aeon-7/aeon-vllm-ultimate@sha256:2fb855ffd6fbf4330cf9f4653c09d3e6584d197acba8e9e93a032da36bb4559f`.
It is referenced for reproducibility but is not copied into this repository's
first published image.

The formal GLM-5.3 Int4/Int8Mix NVFP4-KV image incorporates the `port/`
sources from
`tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s` at commit
`34e81562984bda993e0c9ed01ed6900c17e4857b`, under that repository's
Apache-2.0 license. The source archive is checksum-pinned in the Dockerfile.
The resulting image also contains the separately distributed base image named
above, whose repository-level license was not declared when recorded; the OCI
license label therefore remains `NOASSERTION` rather than implying a license
for the complete image.

The optional formal GLM-5.3 DFlash2 image copies the three DFlash2 runtime
modules from the immutable donor image
`ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
That donor reports vLLM commit `487ecf187d3dfe74d2cf6119a92881dba403c219`;
the copied vLLM modules retain their Apache-2.0 notices. The image also uses the
community port at
`tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark@a1806cb82493aa6f28709f77acf59c1937bdf756`,
and an Apache-2.0 reconstruction published in that repository's issue #1.
The DFlash2 draft checkpoint is not copied into the image. Its upstream model
card currently declares CC BY-NC-ND 4.0; operators must review that license
before downloading or using the separate weights.

The Qwen3.8 Flash Next NVFP4 package uses
`lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
as its immutable runtime base and applies only the source-hash-guarded
compatibility changes published in this repository. Review the base image and
SGLang upstream license notices before redistribution.
