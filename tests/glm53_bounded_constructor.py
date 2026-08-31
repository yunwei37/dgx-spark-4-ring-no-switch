# Appended only to the pinned loader in a disposable, bounded load-test image.
# Native Torch allocation limit, not a host policy or a resident monitor.
_bounded_original_initialize_model = _initialize_model


def _initialize_model(*args, **kwargs):
    import json
    import time
    import torch

    budget = int(107.5 * 1024**3)
    reserve = 3 * 1024**3
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    if free < budget - allocated + reserve:
        raise RuntimeError(
            f"Bounded load needs 3 GiB free outside its Torch budget: "
            f"free={free}, allocated={allocated}, budget={budget}"
        )
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    print("GLM53_TORCH_BUDGET=" + json.dumps({
        "budget_bytes": budget, "free_before_bytes": free,
        "total_bytes": total, "allocated_before_bytes": allocated,
        "reserve_bytes": reserve,
        "scope": "Torch caching allocator only; excludes CPU and non-Torch allocations",
    }, sort_keys=True), flush=True)
    started = time.monotonic()
    model = _bounded_original_initialize_model(*args, **kwargs)
    print("GLM53_REAL_CONSTRUCTOR=" + json.dumps({
        "elapsed_seconds": time.monotonic() - started,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_loaded": False,
    }, sort_keys=True), flush=True)
    return model
