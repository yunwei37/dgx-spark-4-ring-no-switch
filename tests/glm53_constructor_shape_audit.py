# Appended only to the pinned loader in the disposable shape-audit image.
# It always terminates before checkpoint loading; it cannot serve a model.
_shape_audit_original_initialize_model = _initialize_model


def _initialize_model(*args, **kwargs):
    import collections
    import json
    import torch
    from unittest.mock import patch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    # Any unexpected real Torch allocation is bounded independently of SSH.
    # NCCL setup already happened; FakeTensor handles model tensors below.
    torch.cuda.set_per_process_memory_fraction(0.05)
    # This host-membership collective needs real CPU data, not fake .tolist().
    # Query the actual group first, then reuse that exact result only here.
    group = parallel_state.get_tp_group().cpu_group
    membership = triton_symm_mem_ag.in_the_same_node_as(group, source_rank=0)

    def measured_membership(process_group, source_rank=0):
        assert process_group is group and source_rank == 0
        return membership

    with patch.object(triton_symm_mem_ag, "in_the_same_node_as", measured_membership), FakeTensorMode(allow_non_fake_inputs=True):
        model = _shape_audit_original_initialize_model(*args, **kwargs)
        grouped = collections.Counter()
        storages = {}
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if not isinstance(tensor, FakeTensor):
                raise RuntimeError("shape audit encountered real model tensor: " + name)
            storage = tensor.untyped_storage()
            identity = storage._cdata
            if identity in storages:
                continue
            size = storage.nbytes()
            storages[identity] = size
            category = "other"
            if ".experts." in name:
                category = "routed_experts"
            elif ".shared_experts." in name:
                category = "shared_experts"
            elif ".self_attn." in name:
                category = "attention"
            elif "embed_tokens" in name or "lm_head" in name:
                category = "embedding_head"
            grouped[category + ":" + str(tensor.dtype)] += size
        report = {"diagnostic": "constructor-shapes", "full_inference": False,
                  "unique_storage_bytes": sum(storages.values()),
                  "grouped_storage_bytes": dict(grouped),
                  "actual_torch_allocated_bytes": torch.cuda.memory_allocated(),
                  "actual_torch_reserved_bytes": torch.cuda.memory_reserved()}
        print("GLM53_SHAPE_AUDIT=" + json.dumps(report, sort_keys=True), flush=True)
    raise RuntimeError("CONSTRUCTOR_SHAPE_AUDIT_COMPLETE; checkpoint not loaded")
