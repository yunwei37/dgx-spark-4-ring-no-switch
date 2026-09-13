#!/usr/bin/env python3
"""Apply the Qwen QSA compatibility fixes required on GB10 SM121."""

from __future__ import annotations

from pathlib import Path


PATCHES = (
    (
        Path(
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
            "qwen_sparse_attn_backend.py"
        ),
        """    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None
""",
        """    from sglang.srt.utils import is_sm100_supported, is_sm120_supported

    if not (is_sm100_supported() or is_sm120_supported()):
        return None
""",
    ),
    (
        Path(
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
            "attention_registry.py"
        ),
        '                    allowed = {"triton", "trtllm_mha", "flashinfer"}\n',
        '                    allowed = {"triton", "trtllm_mha", "flashinfer", "fa4"}\n',
    ),
    (
        Path(
            "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/"
            "rotary_triton.py"
        ),
        """            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
""",
        """            valid_rotary = cos_offsets < half_rd
            h_mask = (
                ((cos_offsets % 3) == 1)
                & (cos_offsets <= 3 * mrope_section_h)
                & valid_rotary
            )
            w_mask = (
                ((cos_offsets % 3) == 2)
                & (cos_offsets <= 3 * mrope_section_w)
                & valid_rotary
            )
            t_mask = ~(h_mask | w_mask) & valid_rotary
""",
    ),
)


def patch(path: Path, old: str, new: str) -> None:
    original = path.read_bytes()
    text = original.decode()
    if new in text:
        print(f"already patched {path}")
        return
    if text.count(old) != 1:
        raise SystemExit(f"refusing ambiguous {path}: anchor count={text.count(old)}")
    updated = text.replace(old, new, 1).encode()
    path.write_bytes(updated)
    print(f"patched {path}")


def main() -> None:
    for args in PATCHES:
        patch(*args)


if __name__ == "__main__":
    main()

