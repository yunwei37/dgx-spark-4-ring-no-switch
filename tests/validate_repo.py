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
    "Dockerfile.package",
    "Dockerfile.sglang-qwen38",
    "Dockerfile.sglang-glm53",
    "Dockerfile.vllm-glm53-intmix-nvfp4",
    "Dockerfile.vllm-glm53-intmix-nvfp4-dflash2",
    "images/runtime/apply_glm53_router_fp32.py",
    "images/runtime/apply_dflash2_swa_under_mla.py",
    "images/runtime/dsa_block.py",
    "profiles/glm52-int4-int8mix.sh",
    "profiles/qwen38-flash-next-nvfp4.sh",
    "scripts/launch-ring.sh",
    "scripts/launch-sglang-ring.sh",
    "scripts/max_context_probe.py",
    "images/runtime/apply_sglang_nvfp4_gated_tp.py",
    "images/runtime/apply_sglang_qwen_sm121.py",
    "images/runtime/apply_fastsafetensors_033_batch_release.py",
    "images/runtime/apply_sglang_fastsafetensors_local_device.py",
    "benchmarks/glm52-int4-int8mix-2026-08-24.json",
    "benchmarks/glm52-nvfp4-2026-08-25.json",
    "benchmarks/qwen38-flash-next-nvfp4-2026-08-27.json",
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
if len(benchmark_files) < 3:
    fail("repository must retain all published benchmark records")
for path in benchmark_files:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        fail(f"{path.name}: schema_version must be 1")
    if data.get("outcome") not in {
        "passed-restored",
        "failed-restored",
        "passed-active-experiment",
        "passed-bounded-experiment",
    }:
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
package_dockerfile = (ROOT / "Dockerfile.package").read_text(encoding="utf-8")
if "@sha256:" not in package_dockerfile or "sha256sum -c SHA256SUMS" not in package_dockerfile:
    fail("package Dockerfile must pin its base and verify the tested runtime")
qwen_dockerfile = (ROOT / "Dockerfile.sglang-qwen38").read_text(encoding="utf-8")
if qwen_dockerfile.count("@sha256:") < 2:
    fail("Qwen Dockerfile must pin both build and runtime bases")
for patch_name in (
    "apply_sglang_nvfp4_gated_tp.py",
    "apply_sglang_qwen_sm121.py",
):
    if patch_name not in qwen_dockerfile:
        fail(f"Qwen Dockerfile must apply {patch_name}")
if "EXPECTED_SHA256" not in (ROOT / "images/runtime/apply_fastsafetensors_cache_release.py").read_text(encoding="utf-8"):
    fail("loader patch must retain its source hash guard")
for patch_name in (
    "apply_sglang_nvfp4_gated_tp.py",
    "apply_sglang_qwen_sm121.py",
):
    if "hashlib.sha256" not in (ROOT / "images/runtime" / patch_name).read_text(encoding="utf-8"):
        fail(f"{patch_name} must retain its source hash guard")

glm53_dockerfile = (ROOT / "Dockerfile.sglang-glm53").read_text(encoding="utf-8")
if glm53_dockerfile.count("@sha256:") < 2:
    fail("GLM-5.3 Dockerfile must pin both build and runtime bases")
if "fastsafetensors==0.3.3" not in glm53_dockerfile:
    fail("GLM-5.3 Dockerfile must pin fastsafetensors 0.3.3")
for patch_name in (
    "apply_fastsafetensors_033_batch_release.py",
    "apply_sglang_fastsafetensors_local_device.py",
    "apply_sglang_nvfp4_deferred_scales.py",
):
    if patch_name not in glm53_dockerfile:
        fail(f"GLM-5.3 Dockerfile must apply {patch_name}")
    patch_text = (ROOT / "images/runtime" / patch_name).read_text(encoding="utf-8")
    if "hashlib.sha256" not in patch_text or "EXPECTED_SHA256" not in patch_text:
        fail(f"{patch_name} must retain its source hash guard")

intmix_nvfp4_dockerfile = (
    ROOT / "Dockerfile.vllm-glm53-intmix-nvfp4"
).read_text(encoding="utf-8")
for required_text in (
    "@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481",
    "34e81562984bda993e0c9ed01ed6900c17e4857b",
    "--checksum=sha256:7c8d22715693cfa7ddb428d761b6fac71935adcdf3a77c58c80768061d876a72",
    "apply_glm53_router_fp32.py",
    "nvfp4_ds_mla",
):
    if required_text not in intmix_nvfp4_dockerfile:
        fail(f"GLM-5.3 IntMix NVFP4 Dockerfile missing {required_text}")

dflash2_dockerfile = (
    ROOT / "Dockerfile.vllm-glm53-intmix-nvfp4-dflash2"
).read_text(encoding="utf-8")
for required_text in (
    "@sha256:52b289baf653bcb550194822f1c1381275601731d4882fefe7c0413513a99cae",
    "a1806cb82493aa6f28709f77acf59c1937bdf756",
    "@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6",
    "--checksum=sha256:c2fd1bc93957c8a534b03cc83d9d6c28cf7238a9a8ab7113b919f7e013e37c42",
    "COPY --from=dflash2-source",
    "COPY --from=mesh-builder /opt/nccl-mesh /opt/nccl-mesh",
    "NCCL_COMMIT=b91894bd5b190c874d98a017f93f5daa515b65d0",
    "MESH_COMMIT=19924dcc7c571d6e260953724d394ae50bad82cf",
    "LD_LIBRARY_PATH=/opt/nccl-mesh/lib:",
    "apply_dflash2_swa_under_mla.py",
    "apply_fastsafetensors_cache_release.py",
    "verify_dflash2.py",
):
    if required_text not in dflash2_dockerfile:
        fail(f"GLM-5.3 DFlash2 Dockerfile missing {required_text}")
if "EXPECTED_SHA256" not in (
    ROOT / "images/runtime/apply_dflash2_swa_under_mla.py"
).read_text(encoding="utf-8"):
    fail("DFlash2 SWA patch must retain its source hash guard")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "60,469" not in readme or "unsafe" not in readme:
    fail("README must disclose the NVFP4 unsafe boundary")

print("repository_validation=ok")
