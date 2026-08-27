#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch-sglang-ring.sh [--dry-run|--stop] <profile.sh>

Required environment:
  RING_NODES       Four comma-separated SSH hosts in physical ring order.
  MASTER_ADDR      Address reachable by every rank for process rendezvous.
  SOCKET_IFNAME    Interface used for rendezvous/control traffic.
  MODEL_HOST_PATH  Identical existing model directory on every node.
  CACHE_HOST_PATH  Existing writable SGLang cache directory on every node.

Optional environment:
  SSH_USER         SSH user. Default: current local user.
  IMAGE            Immutable tag or digest override.
EOF
}

mode=launch
case "${1:-}" in
  --dry-run|--stop)
    mode=${1#--}
    shift
    ;;
esac

profile=${1:-}
if [[ -z "$profile" || ! -f "$profile" ]]; then
  usage >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$profile"

: "${RING_NODES:?RING_NODES is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${SOCKET_IFNAME:?SOCKET_IFNAME is required}"
: "${MODEL_HOST_PATH:?MODEL_HOST_PATH is required}"
: "${CACHE_HOST_PATH:?CACHE_HOST_PATH is required}"
: "${CONTAINER_NAME:?profile must define CONTAINER_NAME}"
: "${MODEL_CONTAINER_PATH:?profile must define MODEL_CONTAINER_PATH}"
: "${IMAGE:?profile must define IMAGE}"

SSH_USER=${SSH_USER:-${USER}}
IFS=',' read -r -a nodes <<<"$RING_NODES"
if [[ ${#nodes[@]} -ne 4 ]]; then
  echo "RING_NODES must contain exactly four hosts" >&2
  exit 2
fi

quote_command() {
  local quoted=() value
  for value in "$@"; do
    printf -v value '%q' "$value"
    quoted+=("$value")
  done
  local IFS=' '
  printf '%s' "${quoted[*]}"
}

ssh_host() {
  local host=$1
  shift
  ssh -o BatchMode=yes -- "${SSH_USER}@${host}" "$@"
}

if [[ "$mode" == stop ]]; then
  for host in "${nodes[@]}"; do
    ssh_host "$host" docker rm -f "$CONTAINER_NAME"
  done
  exit 0
fi

for host in "${nodes[@]}"; do
  [[ "$mode" == dry-run ]] && continue
  ssh_host "$host" "command -v docker >/dev/null && test -d $(printf '%q' "$MODEL_HOST_PATH") && test -w $(printf '%q' "$CACHE_HOST_PATH") && test -d /dev/infiniband"
done

launch_rank() {
  local rank=$1 host=${nodes[$1]}
  local command=(
    docker run -d --pull never
    --name "$CONTAINER_NAME"
    --network host
    --ipc host
    --gpus all
    --ulimit memlock=-1
    --ulimit stack=67108864
    --cap-add IPC_LOCK
    -v /dev/infiniband:/dev/infiniband
    -e "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME"
    -e "TP_SOCKET_IFNAME=$SOCKET_IFNAME"
    -e "GLOO_SOCKET_IFNAME=$SOCKET_IFNAME"
    -v "$MODEL_HOST_PATH:$MODEL_CONTAINER_PATH:ro"
    -v "$CACHE_HOST_PATH:/root/.cache/sglang"
  )
  local item
  for item in "${CONTAINER_ENV[@]}"; do
    command+=(-e "$item")
  done
  command+=(
    --entrypoint /bin/bash
    "$IMAGE" -lc
    "exec python -m sglang.launch_server $(quote_command "${SGLANG_PROFILE_ARGS[@]}") --tp-size 4 --nnodes 4 --node-rank $rank --dist-init-addr $MASTER_ADDR:$DIST_PORT"
  )

  local rendered
  rendered=$(quote_command "${command[@]}")
  if [[ "$mode" == dry-run ]]; then
    printf '# rank %d on %s\nssh %q %s\n' "$rank" "$host" "${SSH_USER}@${host}" "$rendered"
  else
    ssh_host "$host" "$rendered"
  fi
}

for rank in 1 2 3; do launch_rank "$rank"; done
launch_rank 0

printf 'API head: http://%s:%s/v1\n' "$MASTER_ADDR" "$PORT"
printf 'Profile: %s; image: %s\n' "$PROFILE_NAME" "$IMAGE"
