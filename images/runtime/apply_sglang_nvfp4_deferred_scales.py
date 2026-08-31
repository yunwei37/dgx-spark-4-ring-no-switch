#!/usr/bin/env python3
"""Defer unused CUTLASS blockscale placeholders until normal post-load swizzling."""

import hashlib
from pathlib import Path

EXPECTED_SHA256 = "6ff8f28fbf1d567792375a333d936ca994962e27d9bbc9dcd3489016fce426de"
PATH = Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py")


def patched_source(raw: bytes) -> str:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f"unexpected modelopt_quant.py sha256: {actual}")
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
    print("deferred_scales_sha256=" + hashlib.sha256(PATH.read_bytes()).hexdigest())
