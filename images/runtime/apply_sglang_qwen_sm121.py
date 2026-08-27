#!/usr/bin/env python3
"""Apply the Qwen QSA compatibility fixes required on GB10 SM121."""

from __future__ import annotations

import hashlib
from pathlib import Path


PATCHES = (
    (
        Path(
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
            "qwen_sparse_attn_backend.py"
        ),
        "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2",
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
        "b2700564bbbe536d81ee76775449bd12e835e7b02e94a23d9545aeedc52b095e",
        '                    allowed = {"triton", "trtllm_mha", "flashinfer"}\n',
        '                    allowed = {"triton", "trtllm_mha", "flashinfer", "fa4"}\n',
    ),
    (
        Path(
            "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/"
            "rotary_triton.py"
        ),
        "49b3a2ca2784b4ffb0773947d2e26055f55a520c9cd847b86bcc3f6c2bf94d05",
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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch(path: Path, expected: str, old: str, new: str) -> None:
    original = path.read_bytes()
    source_hash = digest(original)
    text = original.decode()
    if new in text:
        print(f"already patched {path}: {source_hash}")
        return
    if source_hash != expected:
        raise SystemExit(
            f"refusing unknown {path}: expected {expected}, got {source_hash}"
        )
    if text.count(old) != 1:
        raise SystemExit(f"refusing ambiguous {path}: anchor count={text.count(old)}")
    updated = text.replace(old, new, 1).encode()
    path.write_bytes(updated)
    print(f"patched {path}: {source_hash} -> {digest(updated)}")


def main() -> None:
    for args in PATCHES:
        patch(*args)


if __name__ == "__main__":
    main()

