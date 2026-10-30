"""Exercise the staged native-page generator/loader policy without a GPU."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2] / "artifacts/sglang-native"
CONTEXT = ROOT / "v262/context"
BASE = ROOT / "recipe-cfe3cfb/runtime/flashinfer_pce/base"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativePages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(BASE, self.root / "base")
        self.gen = load(
            "native_generator", CONTEXT / "runtime/flashinfer_pce/gen_pce_sources.py"
        )
        self.pce = load(
            "native_pce", CONTEXT / "python/sglang/kernels/ops/attention/dsv41_pce.py"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self):
        return self.gen.generate(
            self.root / "base/csrc", self.root / "base/include", self.root / "gen"
        )

    def test_native_page_dispatch_and_functional_anchor_checks(self):
        self.generate()
        pre = (self.root / "gen/pce_prefill.cu").read_text()
        self.assertIn("DISPATCH_FULLTILE_BY_NH_PBSX(128)", pre)
        self.assertIn("DISPATCH_FULLTILE_BY_NH_PBSX(256)", pre)
        path = self.root / "base/csrc/sparse_mla_sm120_prefill.cu"
        path.write_text("invalid source")
        with self.assertRaises(self.gen.AnchorError):
            self.generate()

    def test_hashes_record_provenance_without_build_or_runtime_gates(self):
        first = self.generate()
        header = (
            self.root
            / "base/include/flashinfer/attention/sparse_mla_sm120/arch/common.cuh"
        )
        header.write_text(header.read_text() + "\n// provenance fixture\n")
        second = self.generate()
        self.assertNotEqual(first["closure"], second["closure"])
        self.assertNotEqual(first["module_tag"], second["module_tag"])
        # The loader consumes its image-built manifest and file; no runtime SHA scan.
        second["outputs"] = {"pce_prefill.cu": "provenance-only"}
        (self.root / "gen/manifest.json").write_text(json.dumps(second))
        self.assertEqual(self.pce.check_sources(self.root / "gen", None, None), second)
        (self.root / "gen/module.json").write_text(
            json.dumps(
                {
                    "name": "fixture",
                    "module_tag": "provenance-only",
                    "so_sha256": "provenance-only",
                }
            )
        )
        so = self.root / "gen/fixture.so"
        so.touch()
        self.assertEqual(self.pce.check_module(self.root / "gen", second), so)
        so.unlink()
        with self.assertRaisesRegex(RuntimeError, "prebuilt module missing"):
            self.pce.check_module(self.root / "gen", second)

    def test_only_native_extra_pages_route_when_enabled(self):
        old = os.environ.get(self.pce.ENV)
        try:
            for flag in ("0", "1"):
                os.environ[self.pce.ENV] = flag
                self.pce.enabled.cache_clear()
                for page in (2, 64, 128, 256):
                    tensor = SimpleNamespace(ndim=4, shape=(4, page, 1, 584))
                    self.assertEqual(
                        self.pce.routes(tensor), flag == "1" and page in (128, 256)
                    )
                self.assertFalse(self.pce.routes(None))
        finally:
            if old is None:
                os.environ.pop(self.pce.ENV, None)
            else:
                os.environ[self.pce.ENV] = old
            self.pce.enabled.cache_clear()


if __name__ == "__main__":
    unittest.main()
