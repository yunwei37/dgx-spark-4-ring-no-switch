# NCCL Mesh runtime for the four-Spark DeepSeek V4.1 run

The published ARM64 runtime image is the containerd image used by the four-Spark
service alongside `vllm-dsv41:overlay5`. Its image config ID is
`sha256:2bcc8d15e3d85016a323f2fba5bec9add5ea51c4be384ea0f192c95596eaa1b3`.

```bash
docker pull ghcr.io/yunwei37/dgx-spark-4-ring-no-switch@sha256:6cc6fd4f42c23940dae2888ba90e9f0f158c08eb970c9b7972edf9f2c3be315e
```

`/opt/nccl-mesh/lib` contains NCCL 2.29.7 and the unmodified Mesh network
plugin. The source build is pinned to NVIDIA NCCL commit
`b91894bd5b190c874d98a017f93f5daa515b65d0` and
`autoscriptlabs/nccl-mesh-plugin` commit
`19924dcc7c571d6e260953724d394ae50bad82cf`. Run
`./build-runtime.sh ./runtime` and then `docker build -t nccl-mesh-runtime .`
on an ARM64 Docker host to rebuild. The image is a library carrier; it does
not contain the model weights, serving patches, or the vLLM overlay image.

The successful September 10-11 run mounted the runtime's `lib` directory over
the wheel-bundled NCCL directory and included `/opt/nccl-mesh/lib` on
`LD_LIBRARY_PATH`. It used `NCCL_NET=Mesh`, `NCCL_ALGO=Ring`,
`NCCL_RUNTIME_CONNECT=1`, and `NCCL_NVLS_ENABLE=0`. No Tree/PAT patch was used.
The matching seven-file serving patch set is in
[`../runtime/deepseek41/`](../runtime/deepseek41/); use its `mounts.txt` for
container paths. See [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
for source licenses.
