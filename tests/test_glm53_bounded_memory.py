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
        for available, peer_ok, communication_error, admitted in ((112, True, False, True), (109, True, False, False), (98, True, False, False), (112, False, False, False), (112, True, True, False)):
            with self.subTest(available_gib=available, peer_ok=peer_ok, communication_error=communication_error):
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
                cpu_group = object()
                def tensor(values, *, dtype, device):
                    self.assertEqual((dtype, device), ("int32", "cpu"))
                    self.assertEqual(values, [int(available >= 107.5 - 72 / 1024 + 3)])
                    return types.SimpleNamespace(data=values, item=lambda: values[0])
                def all_reduce(value, *, op, group):
                    self.assertEqual(op, "MIN")
                    self.assertIs(group, cpu_group)
                    if communication_error:
                        raise RuntimeError("peer disconnected")
                    value.data[0] = min(value.data[0], int(peer_ok))
                distributed = types.SimpleNamespace(all_reduce=all_reduce, ReduceOp=types.SimpleNamespace(MIN="MIN"))
                modules = {"torch": types.SimpleNamespace(cuda=cuda, tensor=tensor, int32="int32", distributed=distributed),
                    "sglang.srt.distributed.parallel_state": types.SimpleNamespace(get_world_group=lambda: types.SimpleNamespace(cpu_group=cpu_group)),
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
