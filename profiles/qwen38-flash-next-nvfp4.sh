#!/usr/bin/env bash

PROFILE_NAME=qwen38-flash-next-nvfp4
CONTAINER_NAME=qwen38-flash-next-nvfp4
IMAGE="${IMAGE:-ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:qwen38-flash-next-nvfp4-20260827}"
MODEL_REPOSITORY=RadixArk/Qwen3.8-Flash-Next-NVFP4
MODEL_REVISION=7b719225
MODEL_CONTAINER_PATH=/model
SERVED_MODEL_NAME=qwen38-flash-next-nvfp4-tp4-mtp-262k
PORT="${PORT:-8414}"
DIST_PORT="${DIST_PORT:-29661}"

SGLANG_PROFILE_ARGS=(
  --model-path "$MODEL_CONTAINER_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --random-seed 42
  --quantization modelopt_fp4
  --fp4-gemm-backend flashinfer_cutlass
  --attention-backend fa4
  --page-size 64
  --mamba-radix-cache-strategy extra_buffer
  --mamba-track-interval 64
  --chunked-prefill-size 4096
  --max-running-requests 36
  --context-length 262144
  --mem-fraction-static 0.80
  --allow-auto-truncate
  --speculative-algorithm EAGLE
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
  --host 0.0.0.0
  --port "$PORT"
)

CONTAINER_ENV=(
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  TORCH_CUDA_ARCH_LIST=12.1a
  FLASHINFER_CUDA_ARCH_LIST=12.1a
  CUTE_DSL_ARCH=sm_121a
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  NCCL_NET_PLUGIN=mesh
  NCCL_NET=Mesh
  NCCL_ALGO=Ring
  NCCL_RUNTIME_CONNECT=1
  NCCL_CUMEM_ENABLE=1
  NCCL_NVLS_ENABLE=0
  NCCL_IGNORE_CPU_AFFINITY=1
  NCCL_DEBUG=WARN
)
