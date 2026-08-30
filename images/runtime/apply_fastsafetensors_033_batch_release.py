#!/usr/bin/env python3
"""Apply the measured GB10 completed-batch cache release to fastsafetensors 0.3.3."""

import hashlib
from pathlib import Path

EXPECTED_SHA256 = "19bdaed6fd26261eeab00ab0cf374f0ac6adbe5bc25a3d15941e694885b8804e"
EXPECTED_PATCHED_SHA256 = "36499d7ca463d39d8c010ce57820ece1b3b5672d7f8db718f2ce6c0bb8949fbc"
IMPORT_ANCHOR = "import os\n"
CLOSE_ANCHOR = (
    '            with TimingContext("fb.close", self._log_message, batch.batch_id) as timer:\n'
    "                batch.fb.close()\n"
    "            close_time = timer.elapsed_ms\n"
)
CLOSE_REPLACEMENT = (
    '            with TimingContext("fb.close", self._log_message, batch.batch_id) as timer:\n'
    "                batch.fb.close()\n"
    "                # GB10 uses one unified CPU/GPU memory pool. Release allocator\n"
    "                # blocks only after the completed file batch has closed.\n"
    "                torch.cuda.empty_cache()\n"
    "            close_time = timer.elapsed_ms\n"
)

candidates = []
for package_dir in ("site-packages", "dist-packages"):
    candidates.extend(
        Path("/usr/local/lib").glob(
            f"python*/{package_dir}/fastsafetensors/parallel_loader.py"
        )
    )
if len(candidates) != 1:
    raise SystemExit(f"expected one parallel_loader.py, found {len(candidates)}")

path = candidates[0]
raw = path.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"unexpected {path} sha256: {actual}")

text = raw.decode("utf-8")
if text.count(IMPORT_ANCHOR) != 1 or text.count(CLOSE_ANCHOR) != 1:
    raise SystemExit("tested fastsafetensors patch anchors are not unique")
text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "import torch\n", 1)
text = text.replace(CLOSE_ANCHOR, CLOSE_REPLACEMENT, 1)
path.write_text(text, encoding="utf-8")

patched = hashlib.sha256(path.read_bytes()).hexdigest()
if patched != EXPECTED_PATCHED_SHA256:
    raise SystemExit(f"unexpected patched {path} sha256: {patched}")

