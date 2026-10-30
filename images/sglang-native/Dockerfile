# Build context: python docker/dgx-spark-native/prepare.py
# ARM64 sibling of the recipe's dev-dsv41 base, resolved 2026-10-29.
ARG BASE_IMAGE=lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9
FROM ${BASE_IMAGE}
# One layer also permits a native snapshotter build in a Workspace where overlay
# whiteout device creation is unavailable. The bind mount is build input only.
RUN --mount=type=bind,target=/context \
    pip install --no-deps --no-cache-dir sglang-kernel==0.4.7 sgl-deep-gemm==0.2.0 \
 && rm -rf /sgl-workspace/sglang/python \
 && cp -a /context/python /sgl-workspace/sglang/python \
 && mkdir -p /opt/dsv41 /opt/b12x_next \
 && cp -a /context/adapter /opt/dsv41/adapter \
 && cp -a /context/b12x_next /opt/b12x_next/b12x_next \
 && cp /context/scripts/pack_engram.py /opt/dsv41/pack_engram.py \
 && cp /context/boot.py /context/production.env /context/provenance.json /context/LICENSE.recipe /context/LICENSE.b12x_next /context/NOTICE.recipe /opt/dsv41/ \
 && cp -a /context/runtime /opt/dsv41/runtime \
 && g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    /opt/dsv41/adapter/row_store.cpp -o /opt/dsv41/adapter/librow_store.so \
 && python3 /opt/dsv41/runtime/flashinfer_pce/build_pce.py
ENV PYTHONPATH=/opt/dsv41/adapter:/opt/b12x_next \
    SGLANG_BUILD_COMMIT=f80c91a4b97e52e17fbcff088aa6b47b6270f6b9 \
    SGLANG_BUILD_URL=https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa \
    SGLANG_IMAGE_TAG=docker.io/library/sglang-dsv41-native:cfe3cfb-production-20261030c \
    OFFLOAD_MODE=nvme NNODES=4 TP_SIZE=4 \
    DSV41_CACHE_GIB=4 DSV41_INDEXER_CHUNKED=1 DSV41_FOLDED_FENCE=1 \
    DSV41_MAX_NEW_TOKENS=0 DSV41_LOOP_ABORT=0 \
    SGLANG_DEFAULT_THINKING=true SGLANG_ROCE_ALLREDUCE=0 \
    SGLANG_RUST_BUILD_MODE=never
WORKDIR /opt/dsv41
ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]
CMD ["run"]
