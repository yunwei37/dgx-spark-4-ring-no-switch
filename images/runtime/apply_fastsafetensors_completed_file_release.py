#!/usr/bin/env python3
"""Release only the completed local shard from the Linux file cache."""

import hashlib
from importlib.metadata import distribution
from pathlib import Path


EXPECTED_SHA256 = "02121afee8262772fc718c729ffd25bb6c716820deff7eb2d5ee04d26b7f6983"
EXPECTED_PATCHED_SHA256 = "c277174047c608794959c38443f2fd9f324295f27efcb34152f83744dbd89b6e"
RANK_ANCHOR = '        self.log_prefix = f"PG{pg.rank() if pg is not None else 0}"\n'
RANK_REPLACEMENT = (
    RANK_ANCHOR
    + "        self.process_group_rank = pg.rank() if pg is not None else 0\n"
)
BATCH_ANCHOR = "            batch = FileBatch(fb, keys, batch_id)\n"
BATCH_REPLACEMENT = (
    BATCH_ANCHOR
    + "            batch.completed_filename = (\n"
    + "                file_list[self.process_group_rank]\n"
    + "                if self.process_group_rank < len(file_list)\n"
    + "                else None\n"
    + "            )\n"
)
CLOSE_ANCHOR = (
    "                batch.fb.close()\n"
    "            torch.cuda.empty_cache()\n"
)
CLOSE_REPLACEMENT = (
    CLOSE_ANCHOR
    + "            # GB10 CPU and GPU share physical memory. Release only the\n"
    + "            # local source shard after its device buffer is closed.\n"
    + "            if batch.completed_filename is not None:\n"
    + "                fd = os.open(batch.completed_filename, os.O_RDONLY)\n"
    + "                try:\n"
    + "                    os.posix_fadvise(\n"
    + "                        fd, 0, 0, os.POSIX_FADV_DONTNEED\n"
    + "                    )\n"
    + "                finally:\n"
    + "                    os.close(fd)\n"
)


path = Path(distribution("fastsafetensors").locate_file(
    "fastsafetensors/parallel_loader.py"
))
raw = path.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"unexpected {path} sha256: {actual}")

text = raw.decode("utf-8")
for name, anchor in (
    ("process-group rank", RANK_ANCHOR),
    ("completed filename", BATCH_ANCHOR),
    ("completed-batch close", CLOSE_ANCHOR),
):
    if text.count(anchor) != 1:
        raise SystemExit(f"tested {name} patch anchor is not unique")

text = text.replace(RANK_ANCHOR, RANK_REPLACEMENT, 1)
text = text.replace(BATCH_ANCHOR, BATCH_REPLACEMENT, 1)
text = text.replace(CLOSE_ANCHOR, CLOSE_REPLACEMENT, 1)
path.write_text(text, encoding="utf-8")

patched = hashlib.sha256(path.read_bytes()).hexdigest()
if patched != EXPECTED_PATCHED_SHA256:
    raise SystemExit(f"unexpected patched {path} sha256: {patched}")
