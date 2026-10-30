#!/usr/bin/env python3
"""Measure the serving image's PyNccl collectives during a reversible in-place pause."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--master-port", type=int, required=True)
parser.add_argument("--library", required=True)
parser.add_argument("--label", choices=["mesh", "native", "native-pci", "native-pci-min4"], required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
os.environ["SGLANG_NCCL_SO_PATH"] = args.library
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,GRAPH"
if args.label != "mesh":
    os.environ.pop("NCCL_NET_PLUGIN", None)
    os.environ.update(NCCL_NET="IB", NCCL_SWITCHLESS_RING_ONLY="1",
                      NCCL_IB_SUBNET_AWARE_ROUTING="1", NCCL_IB_EXTENDED_IPV4_GIDS="1",
                      NCCL_IB_ROUTE_DIAGNOSTICS="1")
    if args.label in ("native-pci", "native-pci-min4"):
        os.environ["NCCL_IB_PRESERVE_PCI_DOMAIN"] = "1"
    if args.label == "native-pci-min4":
        os.environ["NCCL_MIN_NCHANNELS"] = "4"

import torch
import torch.distributed as dist
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

def api(path, payload=None):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())

def fabric_counters():
    values = {}
    for device in Path("/sys/class/infiniband").glob("*"):
        for name in ("port_xmit_data", "port_rcv_data"):
            path = device / "ports/1/counters" / name
            if path.exists():
                values[device.name + "/" + name] = int(path.read_text()) * 4
    return values

def event(phase, **fields):
    print(json.dumps({"phase": phase, "rank": args.rank, "pid": os.getpid(),
                      "time": time.time(), **fields}), flush=True)

mapped_nccl = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                      if "/libnccl.so" in line})
selected_identity = (Path(args.library).stat().st_dev, Path(args.library).stat().st_ino)
mapped_identities = {(Path(path).stat().st_dev, Path(path).stat().st_ino) for path in mapped_nccl}
if mapped_identities != {selected_identity}:
    raise RuntimeError(f"Benchmark requires one NCCL runtime: {mapped_nccl}")
event("cpu_ready", library=args.library, label=args.label, mapped_nccl=mapped_nccl)
dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{args.master_port}",
                        rank=args.rank, world_size=4, timeout=timedelta(seconds=90))
paused = False
pause_started = None
try:
    dist.barrier()
    if args.rank == 0:
        event("entry_load", load=api("get_load"))
        pause_started = time.monotonic()
        paused = True
        api("pause_generation", {"mode": "in_place"})
        event("generation_paused")
    dist.barrier()
    deadline = time.monotonic() + 45
    while True:
        utilization = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip()
        if utilization == "0":
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU remained active after the native in-place pause")
        time.sleep(0.2)
    dist.barrier()
    fabric_before = fabric_counters()
    torch.cuda.set_device(0)
    comm = PyNcclCommunicator(dist.group.WORLD, 0, library_path=args.library)
    if not comm.available:
        raise RuntimeError("SGLang PyNccl could not load the selected library")
    config = json.loads(Path("/models/DeepSeek-V4.1-Flash/config.json").read_text())
    hidden = config.get("text_config", config)["hidden_size"]
    event("communicator_ready", version=comm.nccl_version, hidden=hidden)
    rows = []
    with comm.change_state(enable=True):
        stream = comm._resolve_stream()
        for operation in ("all_reduce", "all_gather"):
            for tokens in (1, 6, 24, 96):
                count = hidden * tokens // (4 if operation == "all_gather" else 1)
                source = torch.full((count,), args.rank + 1, dtype=torch.bfloat16, device="cuda")
                output = torch.empty(count * (4 if operation == "all_gather" else 1),
                                     dtype=torch.bfloat16, device="cuda")
                stream.wait_stream(torch.cuda.current_stream())
                def collective():
                    if operation == "all_reduce":
                        comm.outplace_all_reduce(source, output)
                    else:
                        comm.all_gather(output, source)
                for _ in range(8):
                    collective()
                stream.synchronize()
                samples = []
                for _ in range(8):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    for _ in range(32):
                        collective()
                    end.record(stream)
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000 / 32)
                expected = (torch.full_like(output, 10) if operation == "all_reduce" else
                            torch.arange(1, 5, dtype=output.dtype, device="cuda")
                            .repeat_interleave(count))
                if not torch.equal(output, expected):
                    raise RuntimeError(f"{operation} numerical mismatch at {tokens} token rows")
                rows.append({"operation": operation, "token_rows": tokens,
                             "input_bytes": source.numel() * source.element_size(),
                             "blocks": 8, "iterations_per_block": 32,
                             "us_per_collective": samples, "median_us": statistics.median(samples),
                             "correct": True})
    fabric_after = fabric_counters()
    result = {"mapped_nccl": mapped_nccl, "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
              "fabric_delta_bytes": {k: v - fabric_before[k] for k, v in fabric_after.items()},
              "rank": args.rank, "label": args.label, "library": args.library,
              "nccl_version": comm.nccl_version, "rows": rows}
    gathered = [None] * 4 if args.rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    if args.rank == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ranks": gathered}, indent=2))
        event("measurement_saved", output=str(path))
    dist.barrier()
finally:
    if args.rank == 0 and paused:
        api("continue_generation", {})
        event("generation_resumed", pause_seconds=time.monotonic() - pause_started,
              load=api("get_load"))
    dist.destroy_process_group()
event("exited")
