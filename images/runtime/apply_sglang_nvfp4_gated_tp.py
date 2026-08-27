#!/usr/bin/env python3
"""Patch SGLang's ModelOpt NVFP4 gated-MoE TP padding in-place.

This is an experiment-scoped compatibility patch for the immutable SGLang
image recorded in ../README.md. It refuses unknown source and is idempotent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/"
    "modelopt_quant.py"
)
EXPECTED_SHA256 = "66749a8bd32307c5d4f59aa7f59c41dc8e3bfddd2da8a3d987b426ed3af52539"

ANCHOR = """        else:
            # CUTLASS processing - handle w13 and w2 separately

            if self._is_cutedsl_v2_standard and layer.moe_runner_config.is_gated:
"""

REPLACEMENT = """        else:
            # CUTLASS processing - handle w13 and w2 separately

            # FlashInfer's NVFP4 block-scale layout aligns each logical gated
            # projection independently to 64 rows. With TP, a fused W13 half
            # can be unaligned (for example 160 rows at TP=4). Padding only the
            # fused tensor tail moves the gate/up boundary, so split, pad, and
            # rejoin the two halves before block-scale swizzling or interleave.
            if layer.moe_runner_config.is_gated:
                w13_weight = layer.w13_weight
                w13_weight_scale = layer.w13_weight_scale
                assert w13_weight.size(1) % 2 == 0
                assert w13_weight_scale.size(1) % 2 == 0

                half_rows = w13_weight.size(1) // 2
                half_scale_rows = w13_weight_scale.size(1) // 2
                aligned_half_rows = ((half_rows + 63) // 64) * 64
                half_pad = aligned_half_rows - half_rows

                if half_pad:
                    assert half_pad % 16 == 0
                    scale_pad = half_pad // 16
                    gate, up = torch.split(w13_weight, half_rows, dim=1)
                    gate_scale, up_scale = torch.split(
                        w13_weight_scale, half_scale_rows, dim=1
                    )
                    copy_or_rebind_param(
                        layer,
                        "w13_weight",
                        torch.cat(
                            (
                                torch.nn.functional.pad(gate, (0, 0, 0, half_pad)),
                                torch.nn.functional.pad(up, (0, 0, 0, half_pad)),
                            ),
                            dim=1,
                        ).contiguous(),
                    )
                    copy_or_rebind_param(
                        layer,
                        "w13_weight_scale",
                        torch.cat(
                            (
                                torch.nn.functional.pad(
                                    gate_scale, (0, 0, 0, scale_pad)
                                ),
                                torch.nn.functional.pad(
                                    up_scale, (0, 0, 0, scale_pad)
                                ),
                            ),
                            dim=1,
                        ).contiguous(),
                    )
                    copy_or_rebind_param(
                        layer,
                        "w2_weight",
                        torch.nn.functional.pad(
                            layer.w2_weight, (0, half_pad // 2, 0, 0)
                        ),
                    )
                    copy_or_rebind_param(
                        layer,
                        "w2_weight_scale",
                        torch.nn.functional.pad(
                            layer.w2_weight_scale, (0, scale_pad)
                        ),
                    )

            if self._is_cutedsl_v2_standard and layer.moe_runner_config.is_gated:
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    original = TARGET.read_bytes()
    source_hash = sha256(original)
    text = original.decode()

    if REPLACEMENT in text:
        print(f"already patched: {source_hash}")
        return
    if source_hash != EXPECTED_SHA256:
        raise SystemExit(
            f"refusing unknown {TARGET}: expected {EXPECTED_SHA256}, got {source_hash}"
        )
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"refusing ambiguous source: anchor count={text.count(ANCHOR)}")

    patched = text.replace(ANCHOR, REPLACEMENT, 1).encode()
    TARGET.write_bytes(patched)
    print(f"patched: {source_hash} -> {sha256(patched)}")


if __name__ == "__main__":
    main()

