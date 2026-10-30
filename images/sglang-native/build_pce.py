"""Compile the pinned native-page module without importing Torch or requiring a GPU.

Uses the same CUDA/C++ options as FlashInfer 0.6.18's JIT builder. Keeping package
discovery in stdlib also permits an ARM64 image build under userspace emulation.
"""

import ctypes
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent
    fi = Path(metadata.distribution("flashinfer-python").locate_file("flashinfer"))
    ffi = Path(metadata.distribution("apache-tvm-ffi").locate_file("tvm_ffi"))
    csrc, include = fi / "data/csrc", fi / "data/include"
    spec = importlib.util.spec_from_file_location(
        "pce_generator", root / "gen_pce_sources.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = root / "gen"
    manifest = gen.generate(csrc, include, out)
    name = f"dsv41_pce_sparse_mla_sm120_{manifest['module_tag']}"
    nvcc = Path(shutil.which("nvcc")).resolve()
    cuda = nvcc.parent.parent
    flags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-DPy_LIMITED_API=0x03090000",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
        "-static-global-template-stub=false",
        "-use_fast_math",
        "-Xfatbin=-compress-all",
        "--compress-mode=size",
        "-gencode=arch=compute_121a,code=sm_121a",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    includes = [csrc, include, ffi / "include", ffi / "3rdparty/dlpack/include"]
    includes += [p for p in (fi / "data/cccl").glob("*/include") if p.is_dir()]
    objects = []
    commands = []
    for source in ("pce_paged_attention.cu", "pce_prefill.cu", "pce_binding.cu"):
        obj = out / (source + ".o")
        cmd = [
            str(nvcc),
            *flags,
            *(f"-I{p}" for p in includes),
            "-c",
            str(out / source),
            "-o",
            str(obj),
        ]
        print(json.dumps(cmd), flush=True)
        subprocess.run(cmd, check=True)
        objects.append(str(obj))
        commands.append(cmd)
    so = out / (name + ".so")
    cmd = [
        os.environ.get("CXX", "c++"),
        *objects,
        "-shared",
        f"-L{cuda}/lib64",
        f"-L{cuda}/lib64/stubs",
        "-lcudart",
        "-lcuda",
        "-o",
        str(so),
    ]
    subprocess.run(cmd, check=True)
    commands.append(cmd)
    # Link/load the target-architecture ABI without opening a GPU. The CUDA
    # driver stub is confined to this build process; runtime uses the real driver.
    ctypes.CDLL(str(ffi / "lib/libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(cuda / "lib64/stubs/libcuda.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(so))
    record = {
        "name": name,
        "module_tag": manifest["module_tag"],
        "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
        "commands": commands,
        "abi_load": {
            "machine": platform.machine(),
            "driver": "build-only stub",
            "passed": True,
        },
        "scope": "Image-build provenance; no digest acceptance gate or GPU execution",
    }
    (out / "module.json").write_text(json.dumps(record, indent=2) + "\n")
    for obj in objects:
        Path(obj).unlink()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
