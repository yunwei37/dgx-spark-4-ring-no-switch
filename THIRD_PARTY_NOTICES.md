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

The Qwen3.8 Flash Next NVFP4 package uses
`lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
as its immutable runtime base and applies only the source-hash-guarded
compatibility changes published in this repository. Review the base image and
SGLang upstream license notices before redistribution.
