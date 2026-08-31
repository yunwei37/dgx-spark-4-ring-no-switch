#!/usr/bin/env python3
# Apply the config-requested router precision correction to the pinned image.
import hashlib
from pathlib import Path

EXPECTED = "bf44d3417b6c1441345900c7fac0399bc530165d0367c997b6194555c6bf2130"
RESULT = "d6cf87377e530c878b7f4a12b1517c08ba4213f320caa337b5e9c1462db4e24e"
NEEDLE = '        self.gate = GateLinear(\n            config.hidden_size,\n            config.n_routed_experts,\n            prefix=f"{prefix}.gate",\n        )'
REPLACEMENT = '        self.gate = GateLinear(\n            config.hidden_size,\n            config.n_routed_experts,\n            params_dtype=torch.float32 if getattr(config, "moe_router_dtype", None) == "float32" else None,\n            out_dtype=torch.float32 if getattr(config, "moe_router_dtype", None) == "float32" else None,\n            prefix=f"{prefix}.gate",\n        )'
paths = [p for kind in ("site-packages", "dist-packages")
         for p in Path("/usr/local/lib").glob(
             f"python*/{kind}/vllm/model_executor/models/deepseek_v2.py")]
if len(paths) != 1:
    raise SystemExit(f"expected one packaged model module, got {len(paths)}")
path = paths[0]
raw = path.read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED or raw.decode().count(NEEDLE) != 1:
    raise SystemExit("pinned model source does not match")
patched = raw.decode().replace(NEEDLE, REPLACEMENT).encode()
compile(patched, str(path), "exec")
assert hashlib.sha256(patched).hexdigest() == RESULT
path.write_bytes(patched)
print("glm53_router_source_sha256=" + RESULT)
