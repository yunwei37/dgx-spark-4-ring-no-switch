#!/usr/bin/env python3
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"validation_error={message}", file=sys.stderr)
    raise SystemExit(1)


required = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "Dockerfile",
    "profiles/glm52-int4-int8mix.sh",
    "scripts/launch-ring.sh",
    "benchmarks/glm52-int4-int8mix-2026-08-24.json",
    "benchmarks/glm52-nvfp4-2026-08-25.json",
    "docs/architecture.md",
    "docs/benchmarks.md",
    "docs/configuration-decisions.md",
    "docs/image.md",
    "docs/loader-memory-results.md",
    "docs/blog/2026-08-25-glm52-on-four-dgx-sparks.md",
    "docs/assets/glm52-decode-throughput.svg",
)
for relative in required:
    if not (ROOT / relative).is_file():
        fail(f"missing {relative}")

benchmark_files = sorted((ROOT / "benchmarks").glob("*.json"))
if len(benchmark_files) != 2:
    fail("first release must contain exactly the two tested benchmark records")
for path in benchmark_files:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        fail(f"{path.name}: schema_version must be 1")
    if data.get("outcome") not in {"passed-restored", "failed-restored"}:
        fail(f"{path.name}: unsupported outcome")

private_patterns = {
    "tailscale IPv4": re.compile(r"100\.(?:\d{1,3}\.){2}\d{1,3}"),
    "private lab subnet": re.compile(r"128\.114\.59\.\d{1,3}"),
    "private PVC id": re.compile(r"pvc-[0-9a-f-]{20,}"),
    "fleet hostname": re.compile(r"spark-(?:9773|d1d8|44d7|d5fe)"),
    "authorized password": re.compile("159806" + "94763"),
}
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for label, pattern in private_patterns.items():
        if pattern.search(text):
            fail(f"{label} in {path.relative_to(ROOT)}")

dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
if re.search(r"(?:^|[:=])latest(?:\s|$)", dockerfile, re.MULTILINE):
    fail("Dockerfile must not use latest")
if dockerfile.count("@sha256:") < 2:
    fail("both Dockerfile bases must be digest-pinned")
if "EXPECTED_SHA256" not in (ROOT / "images/runtime/apply_fastsafetensors_cache_release.py").read_text(encoding="utf-8"):
    fail("loader patch must retain its source hash guard")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "60,469" not in readme or "unsafe" not in readme:
    fail("README must disclose the NVFP4 unsafe boundary")

print("repository_validation=ok")
