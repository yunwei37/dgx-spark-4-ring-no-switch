# Native DeepSeek 1017 SGLang runtime

This directory preserves the build inputs for the measured
[2026-10-30 qualification](../../docs/deepseek41-1017-sglang.md).
It uses the official checkpoint, not an uncensored derivative.

## Build

Run in the project's development environment, from the repository root:

```sh
python3 images/sglang-native/prepare.py
docker build --platform linux/arm64 \
  -f artifacts/sglang-native/v262/context/Dockerfile \
  -t ghcr.io/yunwei37/dgx-spark-4-ring-no-switch:sglang-rebuild \
  artifacts/sglang-native/v262/context
```

The preparation script fetches the exact credited upstream recipe and its
ordered SGLang/b12x backports, applies the recorded patches, generates the
Production profile, and builds the native extra-page module in the image.
No extra source/download SHA acceptance gate is used. Git and image digests
remain provenance. Rebuilding is reproducible source provenance; it is not
a claim that a new compiler invocation is byte-identical to the measured image.

The measured image is reconstructed without compilation by combining the
release's `sglang-native-production-overlay.oci.tar` with the pinned public
base specified in the qualification report. The existing DeepSeek workflow's
`qualified_release` input publishes that exact OCI manifest. The companion
NCCL/Mesh OCI image is a second release asset. No weights or credentials are
included.

## Run contract

One process per Spark, in physical ring order; checkpoint and writable cache
are node-local. Each rank supplies `NODE_RANK`, `DIST_INIT_ADDR`, the existing
`API_KEY`, and appropriate NCCL/Mesh interface selection for its own topology.
Use the companion runtime at `/opt/nccl-mesh` and over the image's pip NCCL
library directory, as required by the upstream single-runtime guidance.

The fleet launcher sources the image's generated profile before executing
upstream boot. A portable equivalent inside the serving image is:

```sh
set -a
. /opt/dsv41/production.env
set +a
export SGLANG_NCCL_SO_PATH=/opt/nccl-mesh/lib/libnccl.so.2
export LD_LIBRARY_PATH="/opt/nccl-mesh/lib:${LD_LIBRARY_PATH:-}"
exec python3 -u /opt/dsv41/boot.py run
```

Mount the checkpoint read-only at `/models/DeepSeek-V4.1-Flash`, a writable
node-local cache at `/cache`, and the existing ConnectX devices. Inject the
API key through the normal runtime mechanism, never into this repository.
The profile itself contains no credential.

Before the packed profile is used, prepare both local rank shards with the
upstream utility included in the image:

```sh
python3 /opt/dsv41/pack_engram.py --rank "$NODE_RANK" --tp 4 \
  --out /cache/engram-tp4
```

The source checkpoint must be the exact official 1017 revision. Packing
rewrites disk layout only and consumes about 47.2 GiB per rank. Retain the
checkpoint for normal tensor loading. A changed derivative may reuse packed
Engram only after its actual tensor changes are checked.

The standard profile uses the documented NCCL fallback, not RoCEnante
opposite-node hardware forwarding. It does not configure host networking,
GPU clocks, swap, compaction or power. Backend authentication remains required.

## Verification and limitations

The included probe and MMLU client are reusable harness source, not raw user
traffic. See the report for the frozen throughput protocol, actual 1M result,
pilot quality boundary, OMP acceptance and unresolved decode-performance gap.
The earlier generic launch.py is retired; there is one upstream boot command.

Upstream attribution and license text are retained in `NOTICE.recipe`,
`LICENSE.recipe` and the prepared image's b12x license. Dependencies and
backports are recorded in the generated `provenance.json`.
