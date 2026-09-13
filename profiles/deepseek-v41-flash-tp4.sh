#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, TP=4 across the four-node ring, one rank per node.
# See docs/deepseek-v41-flash.md for the image, checkpoint pin, measurements
# and the per-node NVMe prefix-cache tier.
#
# Required environment (never committed):
#   NODE_RANK     0..3, in physical ring order so every NCCL ring hop is a
#                 direct link carried by the Mesh plugin
#   MASTER_ADDR   rank 0's address on the socket interface
# Optional:
#   KV_OFFLOAD_BYTES  per-node NVMe tier file size (default 64 GiB)
#
# Mount into the serving container:
#   /models/DeepSeek-V4.1-Flash   the checkpoint (read-only)
#   /kv-offload                   a local NVMe directory for the KV tier
#   /opt/spark-manage/py/dsv41_kv_nvme.py and the patched
#   vllm/tokenizers/deepseek_v41.py from images/runtime/deepseek41/
set -euo pipefail
: "${NODE_RANK:?}" "${MASTER_ADDR:?}"
export LD_LIBRARY_PATH="/opt/nccl-mesh/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/spark-manage/py${PYTHONPATH:+:$PYTHONPATH}"
KV_OFFLOAD_BYTES="${KV_OFFLOAD_BYTES:-68719476736}"
ulimit -l unlimited || true
args=(
  serve /models/DeepSeek-V4.1-Flash
  --served-model-name deepseek-v4.1-flash
  --host 0.0.0.0 --port 8000
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size 4
  # 0.83 (validated on this hardware with CUDA graphs at 1M): the vision
  # tower plus multimodal profiling needs 3.66 GiB KV per rank and 0.80
  # left 3.44 GiB, failing the one-request 1M check.
  --gpu-memory-utilization 0.83
  --max-model-len 1048576
  --max-num-seqs 8
  --max-num-batched-tokens 8192
  --block-size 128
  --engram-config '{"cpu_offload": false}'
  # Vision: the checkpoint ships a vision tower (config.vision_config) and
  # this runtime implements the full multimodal path. Bound the per-prompt
  # image count so profiling stays close to the text-only profile; every
  # other flag is unchanged.
  --limit-mm-per-prompt '{"image": 5}'
  --default-chat-template-kwargs '{"thinking": false}'
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}'
  --enable-cumem-allocator
  --kv-transfer-config "{\"kv_connector\":\"CacheableGroupsOffloadingConnector\",\"kv_connector_module_path\":\"dsv41_kv_nvme\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"NVMeOffloadingSpec\",\"spec_module_path\":\"dsv41_kv_nvme\",\"nvme_dir\":\"/kv-offload\",\"nvme_bytes\":${KV_OFFLOAD_BYTES},\"offload_prompt_only\":false}}"
  --distributed-executor-backend mp
  --nnodes 4 --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29541
)
if [ "$NODE_RANK" != 0 ]; then args+=(--headless); fi
exec vllm "${args[@]}"
