#!/usr/bin/env python3
"""One-layer GPU diagnostic, not inference. Run in the pinned GLM image.

Use one idle GPU, an 8 GiB container limit, no network and a 180-second external
deadline. Only the two unused placeholders differ; normal post-load code runs.
"""
import gc
import hashlib
import inspect
import json
import textwrap
import types
from unittest.mock import patch

import torch

torch.cuda.set_per_process_memory_fraction(0.05)
from sglang.srt.layers.quantization import modelopt_quant as modelopt
import sglang.srt.layers.moe as moe
from sglang.srt.layers.moe import MoeRunnerBackend

backend = MoeRunnerBackend.FLASHINFER_CUTLASS
source = textwrap.dedent(inspect.getsource(modelopt.ModelOptNvFp4FusedMoEMethod.create_weights))
old_condition = "    if self.enable_flashinfer_trtllm_moe:\n"
new_condition = "    if self.enable_flashinfer_trtllm_moe or self.enable_flashinfer_cutlass_moe:\n"
if source.count(new_condition) == 2:
    source = source.replace(new_condition, old_condition)
assert source.count(old_condition) == 2


def method_from_source(text):
    namespace = dict(modelopt.__dict__)
    exec(compile(text, "<scale-constructor-ab>", "exec"), namespace)
    return namespace["create_weights"]


def digest(tensor):
    return hashlib.sha256(tensor.detach().view(torch.uint8).cpu().contiguous().numpy().tobytes()).hexdigest()


rows = []
with (
    patch.object(modelopt, "get_moe_runner_backend", return_value=backend),
    patch.object(moe, "get_moe_runner_backend", return_value=backend),
    # No collective dispatcher or distributed group in this one-layer test.
    patch.object(modelopt, "should_use_flashinfer_cutlass_moe_fp4_allgather", return_value=False),
):
    for mode in ("baseline", "deferred"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        layer = torch.nn.Module()
        layer.num_local_experts = 64
        layer.num_experts = 256
        layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
        layer.dispatcher = types.SimpleNamespace(set_quant_config=lambda config: None)
        quant = modelopt.ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16, use_per_token_activation=False)
        method = modelopt.ModelOptNvFp4FusedMoEMethod(quant)
        variant = source if mode == "baseline" else source.replace(old_condition, new_condition)
        method.create_weights = types.MethodType(method_from_source(variant), method)
        with torch.device("cuda"):
            method.create_weights(layer, num_experts=256, hidden_size=6144, intermediate_size_per_partition=2048, params_dtype=torch.bfloat16, weight_loader=lambda *args, **kwargs: None)
            torch.cuda.synchronize()
            row = {"mode": mode, "constructor_allocated_bytes": torch.cuda.memory_allocated(), "constructor_reserved_bytes": torch.cuda.memory_reserved(), "constructor_peak_bytes": torch.cuda.max_memory_allocated()}
            with torch.no_grad():
                for name, param in layer.named_parameters():
                    if "blockscale_swizzled" in name:
                        continue
                    if name in ("w13_weight_scale", "w2_weight_scale"):
                        values = (torch.arange(param.shape[1], device="cuda", dtype=torch.float32) % 8 + 1).reshape(1, -1, 1)
                        param.copy_(values.expand_as(param))
                    else:
                        param.fill_(17 if param.dtype == torch.uint8 else 1)
            method.process_weights_after_loading(layer)
            torch.cuda.synchronize()
            row.update(final_allocated_bytes=torch.cuda.memory_allocated(), peak_bytes=torch.cuda.max_memory_allocated())
            row["scale_aliases"] = {name: getattr(layer, name + "_weight_scale").data_ptr() == getattr(layer, name + "_blockscale_swizzled").data_ptr() for name in ("w13", "w2")}
            row["derived_sha256"] = {name: digest(getattr(layer, name + "_blockscale_swizzled")) for name in ("w13", "w2")}
            row["weight_constant_preserved"] = all(bool(torch.all(getattr(layer, name + "_weight") == 17)) for name in ("w13", "w2"))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del layer, method, quant, param, values

assert rows[0]["derived_sha256"] == rows[1]["derived_sha256"]
assert all(all(row["scale_aliases"].values()) and row["weight_constant_preserved"] for row in rows)
saved = rows[0]["constructor_allocated_bytes"] - rows[1]["constructor_allocated_bytes"]
assert saved == 150994944, saved
print(json.dumps({"diagnostic": "PASS", "full_inference": False, "per_layer_saved_bytes": saved, "75_layer_saved_GiB": saved * 75 / 2**30}), flush=True)
