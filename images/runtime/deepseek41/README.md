# DeepSeek-V4.1-Flash runtime files

Mounted over the vLLM day-0 image described in
[`docs/deepseek-v41-flash.md`](../../../docs/deepseek-v41-flash.md); they are
not baked into a published image. See `THIRD_PARTY_NOTICES.md` for licenses.

| File | Container path | Replaces | Why |
| --- | --- | --- | --- |
| `engram.py` | `vllm/models/deepseek_v4_1/common/engram.py` | the recipe's disk-backed `engram.py` | Adds `gather_engram_hashes`, which the day-0 `nvidia/model.py` imports; identity at Engram data-parallel size 1, raises above it |
| `deepseek_v41_tokenizer.py` | `vllm/tokenizers/deepseek_v41.py` | image file, MD5 `1ffa7369593c525262f000641e956cef` | Accepts the fleet's effort levels: V4.1 names low (25), high (50), xhigh (75) and max (100); `medium` maps to budget 37 and `minimal` to low instead of HTTP 400 |
| `dsv41_kv_nvme.py` | on `PYTHONPATH` (the profile uses `/opt/spark-manage/py`) | new | Per-node NVMe prefix-cache tier, loaded through `--kv-transfer-config` |

The seven other patch files come unchanged from the recipe commit named in
`THIRD_PARTY_NOTICES.md`; mount them as the recipe's `mounts` list describes.

## `dsv41_kv_nvme.py`

- `CacheableGroupsOffloadingConnector` gives vLLM's `OffloadingConnector` only
  the KV cache groups that are prefix-cacheable and transferable. V4.1's
  compressor ring (`CircularBufferSpec`) is per-request scratch that the GPU
  prefix cache already skips; hits resume at scheduler-block boundaries where
  the ring is empty. Per-group block IDs and the scheduler output are narrowed
  with copies, never mutated.
- `NVMeOffloadingSpec` keeps vLLM's CPU LRU manager on the scheduler as a slot
  allocator; each worker copies its own KV slice between device memory and a
  preallocated local file (`kv-slots-r<rank>.bin`) through a pinned staging
  buffer, with `O_DIRECT` when the filesystem supports it. No KV bytes cross
  nodes. vLLM's own filesystem tier is unsuitable for one-rank-per-node TP:
  it performs I/O in the scheduler process through that node's `/dev/shm`
  region, which holds only rank 0's slice.
- vLLM's connector worker asserts that every transfer succeeds, so an I/O
  error is raised instead of reported; the slot file is preallocated so a
  store cannot run out of space.
