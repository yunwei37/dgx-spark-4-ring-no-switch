#!/usr/bin/env python3
"""Select the process-local CUDA device in SGLang's distributed fast loader."""

import hashlib
from pathlib import Path

EXPECTED_SHA256 = "d82dc59e8d4a2fafac2e61c468da485e9f7a85042cf9044b6b37c3a3b6b86041"
OLD = '    device = torch.device(f"cuda:{rank}")\n'
NEW = (
    "    # Distributed rank is global; select the GPU visible to this process.\n"
    '    device = torch.device("cuda", torch.cuda.current_device())\n'
)
PATH = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py"
)

raw = PATH.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"unexpected {PATH} sha256: {actual}")

text = raw.decode("utf-8")
if text.count(OLD) != 1:
    raise SystemExit("tested SGLang local-device patch anchor is not unique")
PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")

