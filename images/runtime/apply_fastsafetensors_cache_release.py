#!/usr/bin/env python3
"""Apply the exact tested fastsafetensors CUDA-cache lifetime change."""

import hashlib
from pathlib import Path

EXPECTED_SHA256 = "3d22ddb0afa51e4f3a56257911aadee99832bcb14bd44a4835d1d63806877709"
NEEDLE = "batch.fb.close()\n"
REPLACEMENT = "batch.fb.close()\n            torch.cuda.empty_cache()\n"

candidates = []
for package_dir in ("site-packages", "dist-packages"):
    candidates.extend(Path("/usr/local/lib").glob(
        f"python*/{package_dir}/fastsafetensors/parallel_loader.py"
    ))
if len(candidates) != 1:
    raise SystemExit(f"expected one parallel_loader.py, found {len(candidates)}")

path = candidates[0]
raw = path.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"unexpected {path} sha256: {actual}")

text = raw.decode("utf-8")
if text.count(NEEDLE) != 1:
    raise SystemExit("tested insertion point was not unique")
path.write_text(text.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
