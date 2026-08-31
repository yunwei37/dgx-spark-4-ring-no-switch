#!/usr/bin/env python3
"""Make an ephemeral, weight-copy-free view for Qwen4's built-in MTP loader.

Use only as the draft path, never as the target model. The target supplies the
embedding and output head. This script changes no original checkpoint file.
"""
import argparse
import json
import struct
from pathlib import Path


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        tensors = json.loads(stream.read(size))
        stream.seek(-1, 2)
        if len(stream.read(1)) != 1:
            raise ValueError(f"Unreadable tail: {path.name}")
    return size, {k: v for k, v in tensors.items() if k != "__metadata__"}


def prepare(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Draft view must be outside the original checkpoint")
    if destination.exists():
        raise FileExistsError("Use a fresh Pod-owned temporary draft directory")
    index_name = "model.safetensors.index.json"
    original = json.loads((source / index_name).read_text())
    weight_map = original["weight_map"]
    mtp = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}
    if not mtp:
        raise ValueError("No Qwen MTP tensors in this checkpoint index")
    files = sorted(set(mtp.values()))
    selected = {k: v for k, v in weight_map.items() if v in files}
    selected_bytes = 0
    mtp_bytes = 0
    payload_bytes = 0
    file_sizes = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError("Expected a checkpoint-local shard filename")
        path = source / name
        size, tensors = header(path)
        file_size = path.stat().st_size
        payload_end = max(t["data_offsets"][1] for t in tensors.values())
        if size + 8 + payload_end != file_size:
            raise ValueError(f"Truncated or inconsistent shard: {name}")
        for tensor, shard in selected.items():
            if shard == name and tensor not in tensors:
                raise ValueError(f"Index/header mismatch: {tensor}")
        for tensor, spec in tensors.items():
            begin, end = spec["data_offsets"]
            payload_bytes += end - begin
            if tensor in mtp:
                mtp_bytes += end - begin
        file_sizes[name] = file_size
        selected_bytes += file_size
    destination.mkdir(parents=False)
    for path in source.iterdir():
        if path.is_file() and path.name != index_name and (
            path.name in files or path.suffix in {".json", ".jinja", ".txt", ".model"}
        ):
            (destination / path.name).symlink_to(path)
    reduced = {
        "metadata": {**original.get("metadata", {}), "total_size": payload_bytes},
        "weight_map": selected,
    }
    (destination / index_name).write_text(json.dumps(reduced, indent=2) + "\n")
    return {
        "target_files": len(set(weight_map.values())),
        "draft_files": files,
        "draft_file_sizes": file_sizes,
        "draft_file_bytes": selected_bytes,
        "mtp_tensor_count": len(mtp),
        "mtp_tensor_bytes": mtp_bytes,
        "draft_index_tensor_count": len(selected),
        "copied_weight_bytes": 0,
        "source_modified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), sort_keys=True))
