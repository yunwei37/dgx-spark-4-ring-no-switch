#!/usr/bin/env python3
"""Regression for the test-only bounded loader's native UMA accounting."""
import contextlib
import io
import pathlib
import types
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).with_name("glm53_bounded_constructor.py").read_text()

class BoundedMemoryTest(unittest.TestCase):
    def test_native_availability_and_unchanged_reserve(self):
        for available, admitted in ((112, True), (109, False), (98, False)):
            with self.subTest(available_gib=available):
                calls = []
                cuda = types.SimpleNamespace(
                    current_device=lambda: 0,
                    mem_get_info=lambda _: (98 * 2**30, 120 * 2**30),
                    memory_allocated=lambda _: 72 * 2**20,
                    memory_reserved=lambda _: 72 * 2**20,
                    max_memory_allocated=lambda _: 72 * 2**20,
                    set_per_process_memory_fraction=lambda f, d: calls.append((f, d)),
                )
                def native(device, gpu, *, empty_cache):
                    self.assertEqual((device, gpu, empty_cache), ("cuda", 0, False))
                    return available
                modules = {"torch": types.SimpleNamespace(cuda=cuda),
                    "sglang.srt.utils.common": types.SimpleNamespace(get_available_gpu_memory=native)}
                scope = {"_initialize_model": lambda: calls.append("constructed")}
                with patch.dict("sys.modules", modules), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(SOURCE, "bounded.py", "exec"), scope)
                    if admitted:
                        scope["_initialize_model"]()
                        self.assertEqual(calls, [(107.5 / 120, 0), "constructed"])
                    else:
                        with self.assertRaises(RuntimeError):
                            scope["_initialize_model"]()
                        self.assertEqual(calls, [])

if __name__ == "__main__":
    unittest.main()
