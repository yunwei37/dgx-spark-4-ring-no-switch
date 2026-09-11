# SPDX-License-Identifier: Apache-2.0
"""Per-node NVMe prefix-cache tier for DeepSeek-V4.1-Flash under multi-node TP.

Loaded through ``--kv-transfer-config``::

    {"kv_connector": "CacheableGroupsOffloadingConnector",
     "kv_connector_module_path": "dsv41_kv_nvme",
     "kv_role": "kv_both",
     "kv_connector_extra_config": {
         "spec_name": "NVMeOffloadingSpec", "spec_module_path": "dsv41_kv_nvme",
         "nvme_dir": "/kv-offload", "nvme_bytes": 137438953472,
         "offload_prompt_only": false}}

``CacheableGroupsOffloadingConnector`` hands vLLM's ``OffloadingConnector``
only the KV cache groups that take part in prefix caching. DeepSeek-V4.1's
compressor ring (``CircularBufferSpec``) is one scratch block per request that
the GPU prefix cache already excludes: hits resume at scheduler-block
boundaries, where the ring holds no open compression group.

``NVMeOffloadingSpec`` keeps every worker's KV slice in a preallocated file on
that worker's own node. The scheduler tracks slot ownership with vLLM's CPU
LRU manager and never touches KV bytes, so nothing crosses nodes. vLLM's
filesystem tier is unusable here: it moves bytes through the scheduler node's
``/dev/shm`` region, which holds only rank 0's slice when TP ranks sit on
different nodes.
"""

import dataclasses
import os
import queue
import threading
import time

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    scheduler as _offload_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.fs.io import probe_o_direct

# Under "vllm." so vLLM's logging configuration applies to this module.
logger = init_logger(f"vllm.{__name__}")

_ALIGN = 4096

_window_in_chunks = _offload_scheduler.get_sliding_window_size_in_chunks


def _uniform_aware_window_in_chunks(kv_cache_spec, tokens_per_chunk):
    # Every layer of a UniformTypeKVCacheSpecs group has the same attention
    # type (and window), so its first layer describes the group.
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = kv_cache_spec.first_spec
    return _window_in_chunks(kv_cache_spec, tokens_per_chunk)


_offload_scheduler.get_sliding_window_size_in_chunks = _uniform_aware_window_in_chunks


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def offloaded_group_ids(kv_cache_config) -> tuple[int, ...]:
    return tuple(
        idx
        for idx, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.enable_kv_transfer and group.kv_cache_spec.prefix_cacheable
    )


class CacheableGroupsOffloadingConnector(OffloadingConnector):
    """OffloadingConnector restricted to the prefix-cacheable KV groups.

    The wrapped connector sees a KV cache config with only those groups; every
    per-group value crossing the connector boundary is narrowed to match.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        groups = kv_cache_config.kv_cache_groups
        self._group_ids = offloaded_group_ids(kv_cache_config)
        self._num_groups = len(groups)
        if not self._group_ids:
            raise ValueError("no prefix-cacheable KV cache group to offload")
        for idx, group in enumerate(groups):
            spec = group.kv_cache_spec
            logger.info(
                "KV offload group %d: %s%s block_size=%d layers=%d offloaded=%s",
                idx,
                type(spec).__name__,
                f"[{type(spec.first_spec).__name__}]"
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else "",
                spec.block_size,
                len(group.layer_names),
                idx in self._group_ids,
            )
        super().__init__(
            vllm_config,
            role,
            dataclasses.replace(
                kv_cache_config,
                kv_cache_groups=[groups[idx] for idx in self._group_ids],
            ),
        )

    def _select(self, per_group):
        if per_group is None:
            return None
        assert len(per_group) == self._num_groups, (len(per_group), self._num_groups)
        return tuple(per_group[idx] for idx in self._group_ids)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return super().update_state_after_alloc(
            request, KVCacheBlocks(self._select(blocks.blocks)), num_external_tokens
        )

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        narrowed = dataclasses.replace(
            scheduler_output,
            scheduled_new_reqs=[
                dataclasses.replace(req, block_ids=self._select(req.block_ids))
                for req in scheduler_output.scheduled_new_reqs
            ],
            scheduled_cached_reqs=dataclasses.replace(
                cached,
                new_block_ids=[self._select(ids) for ids in cached.new_block_ids],
            ),
        )
        return super().build_connector_meta(narrowed)


class NVMeOffloadingSpec(OffloadingSpec):
    """Offloaded blocks live in one preallocated file per worker.

    ``nvme_bytes`` is the per-worker file size; every worker's file holds the
    same slot numbers, each slot one offloaded block of that worker's slice.
    """

    @classmethod
    def build_metric_definitions(cls, extra_config):
        # The scheduler side is vLLM's CPU LRU manager, which reports the CPU
        # spec's metrics; the Prometheus bridge asserts every reported key.
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    def __init__(self, config):
        super().__init__(config)
        if self.blocks_per_chunk != 1:
            raise ValueError("NVMeOffloadingSpec supports blocks_per_chunk=1 only")
        self.nvme_dir = str(self.extra_config["nvme_dir"])
        nvme_bytes = int(self.extra_config["nvme_bytes"])
        self.batch_bytes = int(self.extra_config.get("batch_bytes", 64 << 20))
        self.slot_bytes = _round_up(config.worker_kv_bytes_per_block, _ALIGN)
        self.num_slots = nvme_bytes // self.slot_bytes if self.slot_bytes else 0
        self._manager: OffloadingManager | None = None
        self._worker: OffloadingWorker | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            logger.info(
                "NVMe KV tier: %d slots x %d bytes per worker (%.1f GiB)",
                self.num_slots,
                self.slot_bytes,
                self.num_slots * self.slot_bytes / 2**30,
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_slots,
                cache_policy=self.extra_config.get("eviction_policy", "lru"),
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=0,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = NVMeOffloadingWorker(
                kv_caches,
                path=os.path.join(
                    self.nvme_dir, f"kv-slots-r{self.config.parallel.rank}.bin"
                ),
                num_slots=self.num_slots,
                slot_bytes=self.slot_bytes,
                batch_bytes=self.batch_bytes,
            )
        return self._worker


@dataclasses.dataclass
class _Job:
    job_id: int
    # (group index, device block ids, slot ids), slot-sorted within a group
    groups: list[tuple[int, np.ndarray, np.ndarray]]
    # Compute-stream point the transfer must follow; None for CPU tensors.
    after: "torch.cuda.Event | None"


class NVMeOffloadingWorker(OffloadingWorker):
    """Moves blocks between device KV and a local slot file.

    One thread per direction runs jobs in submission order through a pinned
    staging buffer. vLLM's connector worker asserts every transfer succeeds, so
    an I/O error is raised from get_finished() rather than reported per job;
    the file is preallocated so a store cannot fail for lack of space.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        path: str,
        num_slots: int,
        slot_bytes: int,
        batch_bytes: int,
    ):
        self._tensors = [
            t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
            for t in kv_caches.tensors
        ]
        self._device = self._tensors[0].device
        self._on_gpu = self._device.type == "cuda"
        self._slot_bytes = slot_bytes
        # Per group: (tensor index, byte offset in the slot, bytes) per ref.
        self._layouts: list[list[tuple[int, int, int]]] = []
        for refs in kv_caches.group_data_refs:
            layout, offset = [], 0
            for ref in refs:
                layout.append((ref.tensor_idx, offset, ref.page_size_bytes))
                offset += ref.page_size_bytes
            if offset > slot_bytes:
                raise ValueError(
                    f"KV group needs {offset} bytes per block, slot is {slot_bytes}"
                )
            self._layouts.append(layout)

        rows = max(1, batch_bytes // slot_bytes)
        self._buffers = {
            is_store: torch.empty(
                (rows, slot_bytes), dtype=torch.int8, pin_memory=self._on_gpu
            )
            for is_store in (True, False)
        }
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        direct = probe_o_direct(directory) and all(
            buf.data_ptr() % _ALIGN == 0 for buf in self._buffers.values()
        )
        flags = os.O_RDWR | os.O_CREAT | (os.O_DIRECT if direct else 0)
        self._fd = os.open(path, flags, 0o600)
        size = num_slots * slot_bytes
        if os.fstat(self._fd).st_size != size:
            os.ftruncate(self._fd, size)
        os.posix_fallocate(self._fd, 0, size)
        logger.info(
            "NVMe KV tier file %s: %.1f GiB, %d-row staging, O_DIRECT=%s",
            path,
            size / 2**30,
            rows,
            direct,
        )

        self._lock = threading.Condition()
        self._pending: set[int] = set()
        self._finished: list[TransferResult] = []
        self._error: BaseException | None = None
        self._queues: dict[bool, queue.Queue] = {True: queue.Queue(), False: queue.Queue()}
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(is_store,),
                name=f"nvme-kv-{'store' if is_store else 'load'}",
                daemon=True,
            )
            for is_store in (True, False)
        ]
        for thread in self._threads:
            thread.start()

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id, src_spec, dst_spec, is_store=True)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(job_id, dst_spec, src_spec, is_store=False)

    def _submit(self, job_id, device_spec, slot_spec, is_store) -> bool:
        assert isinstance(device_spec, GPULoadStoreSpec)
        assert isinstance(slot_spec, BlockIDsLoadStoreSpec)
        device_ids = device_spec.block_ids
        slot_ids = slot_spec.block_ids
        assert len(device_ids) == len(slot_ids), (len(device_ids), len(slot_ids))
        assert len(device_spec.group_sizes) == len(self._layouts)
        groups, start = [], 0
        for group_idx, size in enumerate(device_spec.group_sizes):
            if size:
                ids = device_ids[start : start + size]
                slots = slot_ids[start : start + size]
                order = np.argsort(slots, kind="stable")
                groups.append((group_idx, ids[order], slots[order]))
            start += size
        after = None
        if self._on_gpu:
            # Stores read KV the model is still writing; loads write blocks
            # whose earlier zeroing is already queued on the compute stream.
            after = torch.cuda.Event()
            after.record(torch.cuda.current_stream(self._device))
        with self._lock:
            self._pending.add(job_id)
        self._queues[is_store].put(_Job(job_id, groups, after))
        return True

    def _run(self, is_store: bool) -> None:
        stream = None
        if self._on_gpu:
            torch.cuda.set_device(self._device)
            stream = torch.cuda.Stream(self._device)
        jobs = self._queues[is_store]
        while True:
            job = jobs.get()
            if job is None:
                return
            start = time.monotonic()
            num_bytes = 0
            try:
                if stream is not None and job.after is not None:
                    stream.wait_event(job.after)
                for group_idx, ids, slots in job.groups:
                    num_bytes += self._transfer(group_idx, ids, slots, is_store, stream)
            except BaseException as exc:  # noqa: BLE001 - surfaced in get_finished
                logger.exception("NVMe KV %s job %d failed", "store" if is_store else "load", job.job_id)
                with self._lock:
                    self._error = self._error or exc
                    self._lock.notify_all()
                continue
            result = TransferResult(
                job_id=job.job_id,
                success=True,
                transfer_size=num_bytes,
                transfer_time=time.monotonic() - start,
            )
            with self._lock:
                self._finished.append(result)
                self._pending.discard(job.job_id)
                self._lock.notify_all()

    def _transfer(self, group_idx, ids, slots, is_store, stream) -> int:
        layout = self._layouts[group_idx]
        buf = self._buffers[is_store]
        rows = buf.shape[0]
        num_bytes = 0
        for begin in range(0, len(slots), rows):
            batch_ids = ids[begin : begin + rows]
            batch_slots = slots[begin : begin + rows]
            n = len(batch_slots)
            if not is_store:
                self._io(buf, batch_slots, write=False)
            if stream is not None:
                with torch.cuda.stream(stream):
                    self._copy(layout, buf, n, batch_ids, is_store)
                    done = torch.cuda.Event()
                    done.record(stream)
                done.synchronize()
            else:
                self._copy(layout, buf, n, batch_ids, is_store)
            if is_store:
                self._io(buf, batch_slots, write=True)
            num_bytes += n * sum(nbytes for _, _, nbytes in layout)
        return num_bytes

    def _copy(self, layout, buf, n, block_ids, is_store) -> None:
        index = torch.as_tensor(block_ids.astype(np.int64), device=self._device)
        for tensor_idx, offset, nbytes in layout:
            kv = self._tensors[tensor_idx]
            staged = buf[:n, offset : offset + nbytes]
            if is_store:
                staged.copy_(kv.index_select(0, index)[:, :nbytes], non_blocking=True)
            else:
                kv[:, :nbytes].index_copy_(
                    0, index, staged.to(self._device, non_blocking=True)
                )

    def _io(self, buf, slots, write: bool) -> None:
        view = memoryview(buf.numpy()).cast("B")
        size = self._slot_bytes
        i, n = 0, len(slots)
        while i < n:
            j = i + 1
            while j < n and slots[j] == slots[j - 1] + 1:
                j += 1
            chunk = view[i * size : j * size]
            offset = int(slots[i]) * size
            done = 0
            while done < len(chunk):
                if write:
                    count = os.pwrite(self._fd, chunk[done:], offset + done)
                else:
                    count = os.preadv(self._fd, [chunk[done:]], offset + done)
                if count <= 0:
                    raise OSError(f"short {'write' if write else 'read'} at {offset + done}")
                done += count
            i = j

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("NVMe KV tier transfer failed") from self._error
            finished, self._finished = self._finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            while self._error is None and not self._pending.isdisjoint(job_ids):
                self._lock.wait()
            if self._error is not None:
                raise RuntimeError("NVMe KV tier transfer failed") from self._error

    def shutdown(self) -> None:
        for jobs in self._queues.values():
            jobs.put(None)
        for thread in self._threads:
            thread.join()
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
