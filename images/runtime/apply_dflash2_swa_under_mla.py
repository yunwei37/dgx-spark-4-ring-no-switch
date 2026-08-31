#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Allow a non-MLA DFlash2 SWA layer under an MLA target model."""

import ast
import hashlib
import pathlib
import sys

EXPECTED_SHA256 = "014f289e4957619b43f4981c02481344a4b56e86d75e85677dec855a05f129a3"
MARKER = "YUNWEI37_DFLASH2_SWA_UNDER_MLA"
RELATIVE_PATH = pathlib.Path("model_executor/layers/attention/attention.py")


def main() -> int:
    root = pathlib.Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/usr/local/lib/python3.12/dist-packages/vllm"
    )
    path = root / RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        ast.parse(source, filename=str(path))
        print(f"already_patched={path}")
        return 0

    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"refusing unexpected {RELATIVE_PATH}: {actual} != {EXPECTED_SHA256}"
        )

    old = (
        "        if self.sliding_window is not None:\n"
        "            assert not vllm_config.model_config.use_mla, (\n"
        '                "MLA is not supported for slidingwindow"\n'
        "            )\n"
    )
    if source.count(old) != 1:
        raise SystemExit(f"expected one SWA-under-MLA anchor, found {source.count(old)}")
    new = (
        "        if self.sliding_window is not None:\n"
        f"            # {MARKER}: model_config.use_mla describes the target model,\n"
        "            # while this Attention instance is the non-MLA DFlash2 drafter.\n"
    )
    result = source.replace(old, new, 1)
    ast.parse(result, filename=str(path))
    path.write_text(result, encoding="utf-8")
    print(f"patched={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
