#!/usr/bin/env python3
# Apply the config-requested router precision correction to the pinned image.
from pathlib import Path

NEEDLE = '        self.gate = GateLinear(\n            config.hidden_size,\n            config.n_routed_experts,\n            prefix=f"{prefix}.gate",\n        )'
REPLACEMENT = '        self.gate = GateLinear(\n            config.hidden_size,\n            config.n_routed_experts,\n            params_dtype=torch.float32 if getattr(config, "moe_router_dtype", None) == "float32" else None,\n            out_dtype=torch.float32 if getattr(config, "moe_router_dtype", None) == "float32" else None,\n            prefix=f"{prefix}.gate",\n        )'
paths = [p for kind in ("site-packages", "dist-packages")
         for p in Path("/usr/local/lib").glob(
             f"python*/{kind}/vllm/model_executor/models/deepseek_v2.py")]
if len(paths) != 1:
    raise SystemExit(f"expected one packaged model module, got {len(paths)}")
path = paths[0]
raw = path.read_bytes()
if raw.decode().count(NEEDLE) != 1:
    raise SystemExit("model patch anchor is not unique")
patched = raw.decode().replace(NEEDLE, REPLACEMENT).encode()
compile(patched, str(path), "exec")
path.write_bytes(patched)
print("glm53 router patch applied")
