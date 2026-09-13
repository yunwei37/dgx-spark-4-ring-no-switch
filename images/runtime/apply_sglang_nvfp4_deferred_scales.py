#!/usr/bin/env python3
"""Defer unused CUTLASS blockscale placeholders until normal post-load swizzling."""

from pathlib import Path

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py")


def patched_source(raw: bytes) -> str:
    text = raw.decode("utf-8")
    start = text.index("    def create_weights(", text.index("class ModelOptNvFp4FusedMoEMethod"))
    end = text.index("    def process_weights_after_loading(", start)
    constructor = text[start:end]
    old = "        if self.enable_flashinfer_trtllm_moe:\n"
    new = "        if self.enable_flashinfer_trtllm_moe or self.enable_flashinfer_cutlass_moe:\n"
    if constructor.count(old) != 2:
        raise ValueError("expected exactly two constructor blockscale placeholders")
    constructor = constructor.replace(old, new)
    old_comment = (
        "        # TRTLLM replaces blockscale_swizzled with an alias to weight_scale\n"
        "        # during process_weights_after_loading, so skip the expensive\n"
        "        # swizzle+allocate here to avoid GPU memory fragmentation\n"
    )
    if constructor.count(old_comment) != 1:
        raise ValueError("expected constructor rationale anchor")
    constructor = constructor.replace(old_comment, (
        "        # TRTLLM and CUTLASS derive these scales after loading weights.\n"
        "        # Do not swizzle uninitialized buffers or retain duplicate\n"
        "        # placeholders while the entire model is being constructed.\n"
    ))
    return text[:start] + constructor + text[end:]


if __name__ == "__main__":
    updated = patched_source(PATH.read_bytes())
    compile(updated, str(PATH), "exec")
    PATH.write_text(updated, encoding="utf-8")
    print("deferred scales patch applied")
