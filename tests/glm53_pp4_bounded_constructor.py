# Appended only to the pinned loader in a disposable, bounded load-test image.
# Native Torch allocation limit, not a host policy or a resident monitor.
_bounded_original_initialize_model = _initialize_model


def _initialize_model(*args, **kwargs):
    import json
    import time
    import torch

    from sglang.srt.distributed.parallel_state import get_pp_group
    pp = get_pp_group()
    assert pp.world_size == 4, "This bounded experiment requires native PP4"
    budget = int((105, 107.5, 107.5, 107.5)[pp.rank] * 1024**3)
    reserve = 3 * 1024**3
    device = torch.cuda.current_device()
    cuda_free, total = torch.cuda.mem_get_info(device)
    from sglang.srt.utils.common import get_available_gpu_memory
    free = int(get_available_gpu_memory("cuda", device, empty_cache=False) * (1 << 30))
    allocated = torch.cuda.memory_allocated(device)
    from sglang.srt.distributed.parallel_state import get_world_group
    # Reuse the existing CPU group: no GPU allocation before all ranks agree.
    admitted = torch.tensor(
        [int(free >= budget - allocated + reserve)], dtype=torch.int32, device="cpu"
    )
    torch.distributed.all_reduce(
        admitted, op=torch.distributed.ReduceOp.MIN, group=get_world_group().cpu_group
    )
    if not admitted.item():
        raise RuntimeError(
            f"World admission failed; requires 3 GiB outside each Torch budget: "
            f"free={free}, allocated={allocated}, budget={budget}"
        )
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    print("GLM53_TORCH_BUDGET=" + json.dumps({
        "budget_bytes": budget, "free_before_bytes": free,
        "cuda_free_before_bytes": cuda_free,
        "total_bytes": total, "allocated_before_bytes": allocated,
        "reserve_bytes": reserve,
        "scope": "Torch caching allocator only; excludes CPU and non-Torch allocations",
    }, sort_keys=True), flush=True)
    started = time.monotonic()
    model = None
    try:
        model = _bounded_original_initialize_model(*args, **kwargs)
    finally:
        print("GLM53_CONSTRUCTOR_EXIT=" + json.dumps({
            "pp_rank": pp.rank,
            "constructor_returned": model is not None,
            "elapsed_seconds": time.monotonic() - started,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_bytes": torch.cuda.max_memory_allocated(device),
            "host_available_bytes": int(get_available_gpu_memory("cuda", device, empty_cache=False) * (1 << 30)),
            "checkpoint_loaded": False,
        }, sort_keys=True), flush=True)
    print("GLM53_REAL_CONSTRUCTOR=" + json.dumps({
        "elapsed_seconds": time.monotonic() - started,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_loaded": False,
    }, sort_keys=True), flush=True)
    return model
