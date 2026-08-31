# Appended only to the pinned loader in the disposable shape-audit image.
# It always terminates before checkpoint loading; it cannot serve a model.
_shape_audit_original_initialize_model = _initialize_model


def _initialize_model(*args, **kwargs):
    import collections
    import json
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    # Any unexpected real Torch allocation is bounded independently of SSH.
    # NCCL setup already happened; FakeTensor handles model tensors below.
    torch.cuda.set_per_process_memory_fraction(0.05)
    with FakeTensorMode(allow_non_fake_inputs=True):
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
