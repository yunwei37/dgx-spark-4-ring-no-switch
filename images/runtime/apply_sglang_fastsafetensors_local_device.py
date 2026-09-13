#!/usr/bin/env python3
"""Select the process-local CUDA device in SGLang's distributed fast loader."""

from pathlib import Path

OLD = '    device = torch.device(f"cuda:{rank}")\n'
NEW = (
    "    # Distributed rank is global; select the GPU visible to this process.\n"
    '    device = torch.device("cuda", torch.cuda.current_device())\n'
)
PATH = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py"
)

raw = PATH.read_bytes()
text = raw.decode("utf-8")
if text.count(OLD) != 1:
    raise SystemExit("tested SGLang local-device patch anchor is not unique")
PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")

