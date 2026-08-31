# Qwen built-in MTP: selecting the files before reading them

Status: complete checkpoint and draft-view metadata preflight passed. The new
draft-path GPU inference trial has not passed yet; this is not a startup-speedup
or model-correctness claim.

Our historical Qwen NVFP4 TP2 native-262K run spent 1,221–1,236 seconds loading
the target and another 839–842 seconds loading its built-in MTP draft, reaching
the API after 2,140.59 seconds. That second scan is worth investigating before
changing the shared filesystem or adding cache-control machinery.

## What the actual pinned runtime does

In SGLang `d91c3682b`, the Qwen4 MTP implementation inherits the Qwen3.5 MTP
loader. It discards non-MTP tensors **after** receiving them from the weight
iterator. The target worker supplies the draft's embedding and output head.
The native `--speculative-draft-model-path` can select a separate directory;
whether this particular reduced view works must still be demonstrated by the
real model. The upstream interface is documented in
[SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding).

For `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`:

| Quantity | Measured value |
| --- | ---: |
| Complete checkpoint files | 206 |
| Complete checkpoint bytes | 135,195,303,851 |
| MTP tensors | 31 |
| MTP tensor bytes | 5,214,301,696 |
| Files containing all MTP tensors | 3 |
| Bytes in those three files, including other tensors | 14,734,642,001 |
| Entries in their accurate reduced file index | 1,049 |
| Copied weight bytes / changed source files | 0 / 0 |

The selected files are `model-bf16-00010.safetensors` through
`model-bf16-00012.safetensors`. A symlink to the complete original index would
be wrong: the native file selector verifies that indexed files exist. The
temporary draft directory therefore needs a truthful index of every tensor
in the three selected files. It is an incomplete *draft view*, never the target
model or a smaller substitute for the full checkpoint.

## What has actually passed

All 206 complete shards passed SHA256 against pinned Hub metadata in 223.73
seconds. This measures reading **plus hashing**, with the observed cache state;
it is not a cold raw-filesystem bandwidth benchmark. Header, tensor-index and
file-end checks passed for the temporary view. Original files were read-only.

Both intended GPU nodes imported the same existing assembled runtime, config
ID `sha256:c13c7345095ad010c0b11e68e36707ea9cb109427171e2ebb92d4b7cf5cd50c0`.
The 30,692,449,792-byte archive transferred over an existing direct ConnectX
link in 35.81 seconds; imports took 233.5 and 215.2 seconds. Transfer, unpacking,
hashing and model loading are distinct measurements. Node-cache import is not
GHCR publication. Both temporary archives and the completed reader Job were
removed; the bounded transfer listener expired.

One isolated native-module import stopped because its default compilation-cache
directory was read-only. It did not load a model and does not establish a model
compatibility failure. The GPU trial will use the baseline's writable cache.

## Remaining experiment and restoration boundary

Keep the exact full target, runtime, TP2/EP2 layout, seed, kernels, MTP one-step /
two-draft configuration and 262,144 context. Compare target and draft loading
separately, then run correctness, full-window retrieval and single/concurrent
decode. The historical figures are not a matched-cache A/B control.

The bounded GPU trial adds a loopback-only API, ordinary resource-accounted
scheduling, 110GiB Pod memory limits and a one-hour deadline. Those test boundaries
are recorded separately from the sole performance candidate: file selection.
It creates no host daemon, filesystem tuning, cache-drop loop or duplicated
checkpoint. The previous inference service is restored after the trial.

Machine-readable preflight evidence:
[`benchmarks/qwen38-mtp-view-preflight-2026-08-31.json`](../../benchmarks/qwen38-mtp-view-preflight-2026-08-31.json).
