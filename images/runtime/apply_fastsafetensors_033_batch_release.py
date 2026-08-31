#!/usr/bin/env python3
"""Apply the measured GB10 completed-batch cache release to fastsafetensors 0.3.3."""

import hashlib
from importlib.metadata import distribution
from pathlib import Path

EXPECTED_SHA256 = "19bdaed6fd26261eeab00ab0cf374f0ac6adbe5bc25a3d15941e694885b8804e"
EXPECTED_PATCHED_SHA256 = "5bed0a369aeb511f14ffc4cb2b7b8c0b18782a3887b544f6916f99b73ac39152"
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

path = Path(distribution("fastsafetensors").locate_file("fastsafetensors/parallel_loader.py"))
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
