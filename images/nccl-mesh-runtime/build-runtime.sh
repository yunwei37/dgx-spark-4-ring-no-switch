#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${script_dir}/runtime}"
base_image='ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8'

mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

docker run --rm \
  --entrypoint /bin/bash \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "${script_dir}:/workspace:ro" \
  -v "${output_dir}:/out" \
  "$base_image" -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      build-essential ca-certificates git libibverbs-dev pkg-config
    rm -rf /var/lib/apt/lists/*

    build_root="$(mktemp -d)"
    trap '\''rm -rf "$build_root"'\'' EXIT

    git clone --filter=blob:none https://github.com/NVIDIA/nccl.git "$build_root/nccl"
    git -C "$build_root/nccl" checkout b91894bd5b190c874d98a017f93f5daa515b65d0
    make -C "$build_root/nccl" -j"$(nproc)" src.build CUDA_HOME=/usr/local/cuda

    git clone --filter=blob:none https://github.com/autoscriptlabs/nccl-mesh-plugin.git \
      "$build_root/nccl-mesh-plugin"
    git -C "$build_root/nccl-mesh-plugin" checkout \
      19924dcc7c571d6e260953724d394ae50bad82cf
    make -C "$build_root/nccl-mesh-plugin" -j"$(nproc)"

    rm -rf /out/*
    install -d -m 0755 /out/lib
    install -m 0755 "$build_root/nccl/build/lib/libnccl.so.2.29.7" /out/lib/
    install -m 0755 "$build_root/nccl-mesh-plugin/libnccl-net.so" /out/lib/
    ln -s libnccl.so.2.29.7 /out/lib/libnccl.so.2
    ln -s libnccl.so.2 /out/lib/libnccl.so
    ln -s libnccl-net.so /out/lib/libnccl-net-mesh.so
    chown -R "$HOST_UID:$HOST_GID" /out
  '
