#!/usr/bin/env python3
"""Run CPU behavior tests against the prepared engine, without loading weights."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=Path, default=Path("artifacts/sglang-native/v262/context")
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("artifacts/sglang-native/recipe-cfe3cfb")
    )
    args = parser.parse_args()
    context = args.context.resolve()
    results = []
    # #45804 extends and supersedes the older #45762 AST harness.
    tests = [
        p
        for p in sorted((args.recipe / "scripts").glob("test_sglang_*.py"))
        if p.name != "test_sglang_dsv41_default_thinking.py"
    ]
    log_path = context.parent / "validation.log"
    with log_path.open("w") as log:
        for test in tests:
            start = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(test), str(context / "python")],
                env=dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            results.append(
                {
                    "test": test.name,
                    "returncode": result.returncode,
                    "seconds": time.monotonic() - start,
                }
            )
        # Exercise the actual native storage implementation: byte parity,
        # eviction, unaligned reads and simultaneous callers, using tiny fixtures.
        with tempfile.TemporaryDirectory(prefix="sglang-native-row-store-") as tmp:
            root = Path(tmp)
            (root / "adapter").mkdir()
            (root / "tests").mkdir()
            test = root / "tests/test_row_store.py"
            shutil.copy2(args.recipe / "tests/test_row_store.py", test)
            compiled = subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-pthread",
                    str(context / "adapter/row_store.cpp"),
                    "-o",
                    str(root / "adapter/librow_store.so"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result = compiled.returncode
            if result == 0:
                result = subprocess.run(
                    [sys.executable, str(test)], stdout=log, stderr=subprocess.STDOUT
                ).returncode
            results.append(
                {
                    "test": "native row-store byte parity and concurrency",
                    "returncode": result,
                }
            )
    spec = importlib.util.spec_from_file_location(
        "native_launch", context / "launch.py"
    )
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    cmd = launch.command(
        {"NODE_RANK": "3", "DIST_INIT_ADDR": "head:20000"},
        ["--context-length", "8192", "--chunked-prefill-size", "4096"],
    )
    assert cmd[-2:] == ["--context-length", "1048576"]
    assert cmd[cmd.index("--node-rank") + 1] == "3"
    assert cmd[cmd.index("--served-model-name") + 1] == "deepseek-v4.1-flash"
    assert "--max-running-requests" not in cmd and "--max-queued-requests" not in cmd
    results.append(
        {"test": "native launch context/rank/backend identity", "returncode": 0}
    )
    provenance = json.loads((context / "provenance.json").read_text())
    report = {
        "provenance": provenance,
        "results": results,
        "scope": "CPU source behavior and launcher; no ARM64/TP4 inference claim",
    }
    (context.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(result["test"], "PASS" if result["returncode"] == 0 else "FAIL")
    return int(any(result["returncode"] for result in results))


if __name__ == "__main__":
    sys.exit(main())
