#!/usr/bin/env bash
set -euo pipefail
out=${1:?Usage: build-native-nccl.sh OUTPUT_RUNTIME_DIRECTORY}
mkdir -p "$out"
out=$(cd "$out" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -rf -- "$build"' EXIT
curl -fL https://github.com/NVIDIA/nccl/archive/73cf112295c33aee2b895f329f592f2a9b4b0f97.tar.gz -o "$build/nccl.tar.gz"
tar -xf "$build/nccl.tar.gz" --strip-components=1 -C "$build"
curl -fL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/9932204d25ab857ec4a5a7c437f4dc0f5bddee2e/spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch -o "$build/routing.patch"
cd "$build"
patch -p1 < routing.patch
patch -p1 < "$here/nccl-cts-acquire.patch"
patch -p1 < "$here/nccl-listener-physical-devices.patch"
# The upstream routing patch writes this source with CRLF.
sed -i 's/\r$//' src/transport/generic.cc
patch -p1 < "$here/nccl-switchless-tree.patch"
make -j"$(nproc)" src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" NCCL_GIT_BRANCH=v2.30.7-1 NCCL_GIT_COMMIT_HASH=73cf112295c33aee2b895f329f592f2a9b4b0f97
mkdir -p "$out/lib"
install -m 755 build/lib/libnccl.so.2.30.7 "$out/lib/"
ln -s libnccl.so.2.30.7 "$out/lib/libnccl.so.2"
ln -s libnccl.so.2.30.7 "$out/lib/libnccl.so"
install -m 644 LICENSE.txt "$out/LICENSE.nccl"
