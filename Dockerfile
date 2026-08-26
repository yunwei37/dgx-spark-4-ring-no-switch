ARG BUILD_IMAGE=ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
FROM ${BUILD_IMAGE} AS mesh-builder

ARG NCCL_COMMIT=b91894bd5b190c874d98a017f93f5daa515b65d0
ARG MESH_COMMIT=19924dcc7c571d6e260953724d394ae50bad82cf

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git libibverbs-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/NVIDIA/nccl.git /src/nccl \
    && git -C /src/nccl checkout "${NCCL_COMMIT}" \
    && make -C /src/nccl -j"$(nproc)" src.build CUDA_HOME=/usr/local/cuda \
    && git clone https://github.com/autoscriptlabs/nccl-mesh-plugin.git /src/mesh \
    && git -C /src/mesh checkout "${MESH_COMMIT}" \
    && make -C /src/mesh -j"$(nproc)" \
    && mkdir -p /opt/nccl-mesh/lib \
    && install -m 0755 /src/nccl/build/lib/libnccl.so.2.29.7 /opt/nccl-mesh/lib/ \
    && install -m 0755 /src/mesh/libnccl-net.so /opt/nccl-mesh/lib/ \
    && ln -s libnccl.so.2.29.7 /opt/nccl-mesh/lib/libnccl.so.2 \
    && ln -s libnccl.so.2 /opt/nccl-mesh/lib/libnccl.so \
    && ln -s libnccl-net.so /opt/nccl-mesh/lib/libnccl-net-mesh.so

ARG RUNTIME_IMAGE=ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481
FROM ${RUNTIME_IMAGE}

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="DGX Spark four-ring GLM-5.2 INT4 runtime" \
      org.opencontainers.image.description="Pinned tested vLLM/B12X runtime with NCCL Mesh and bounded loader memory release" \
      org.opencontainers.image.source="https://github.com/yunwei37/dgx-spark-4-ring-no-switch" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

COPY --from=mesh-builder /opt/nccl-mesh /opt/nccl-mesh
COPY images/runtime/apply_fastsafetensors_cache_release.py /tmp/
RUN python3 /tmp/apply_fastsafetensors_cache_release.py && rm /tmp/apply_fastsafetensors_cache_release.py

ENV LD_LIBRARY_PATH=/opt/nccl-mesh/lib:${LD_LIBRARY_PATH}
