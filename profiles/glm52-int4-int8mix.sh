#!/usr/bin/env bash

PROFILE_NAME=glm52-int4-int8mix
CONTAINER_NAME=glm52-int4-int8mix
IMAGE="${IMAGE:-ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:int4-int8mix-20260824}"
MODEL_REPOSITORY=QuantTrio/GLM-5.2-Int4-Int8Mix
MODEL_REVISION=1d3bcfe5ec549ecd000fd80b37f191183842e983
MODEL_CONTAINER_PATH=/models/GLM-5.2-Int4-Int8Mix
SERVED_MODEL_NAME=glm-5.2-int4-int8mix
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
MTP_TOKENS="${MTP_TOKENS:-0}"

VLLM_PROFILE_ARGS=(
  serve "$MODEL_CONTAINER_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 4
  --pipeline-parallel-size 1
  --distributed-executor-backend mp
  --load-format fastsafetensors
  --kv-cache-dtype fp8_ds_mla
  --kv-cache-memory-bytes 2147483648
  --gpu-memory-utilization 0.85
  --max-model-len 8192
  --max-num-seqs 1
  --max-num-batched-tokens 8192
  --compilation-config '{"cudagraph_mode":"FULL"}'
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
)

if [[ "$MTP_TOKENS" != 0 ]]; then
  VLLM_PROFILE_ARGS+=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"draft_tensor_parallel_size\":1,\"attention_backend\":\"FLASHMLA_SPARSE\"}"
  )
fi

CONTAINER_ENV=(
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  NCCL_NET_PLUGIN=none
  NCCL_IB_SUBNET_AWARE_ROUTING=1
  NCCL_IB_MERGE_NICS=0
  NCCL_ALGO=Ring
  NCCL_DEBUG=WARN
)
