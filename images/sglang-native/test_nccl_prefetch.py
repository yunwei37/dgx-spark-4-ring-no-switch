"""CPU contract tests for the actual patched NCCL prefetch adapter.

Real CUDA graph, byte-equality and performance acceptance still run on the fleet.
"""
import ast
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


class NcclPrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="sglang-nccl-prefetch-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.context = Path(cls.temp.name)
        here = Path(__file__).resolve().parent
        recipe = here.parents[1] / "artifacts/sglang-native/recipe-cfe3cfb"
        patch_text = (here / "production-profile.patch").read_text()
        for name in re.findall(r"^\+\+\+ b/(.+)$", patch_text, re.M):
            destination = cls.context / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recipe / name, destination)
        subprocess.run(["patch", "--batch", "-p1"], cwd=cls.context,
                       input=patch_text, text=True, check=True, capture_output=True)

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "prefetch_under_test", self.context / "adapter/l2_prefetch.py")
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.env = patch.dict(os.environ, {
            "DSV41_L2_PREFETCH": "1", "SGLANG_ROCE_ALLREDUCE": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.lib = patch.object(self.adapter, "_lib")
        self.lib_mock = self.lib.start()
        self.addCleanup(self.lib.stop)
        self.events = []

        events = self.events
        class Communicator:
            disabled = False
            def _resolve_stream(self):
                return "current-communication-stream"
            def all_reduce(self, tensor, op="sum"):
                events.append(("original_ar", tensor, op))
                return tensor
            def outplace_all_reduce(self, tensor, output=None, op="sum"):
                events.append(("original_outplace", tensor, output, op))
                return output
            def all_gather(self, output, tensor, sizes=None):
                events.append(("original_ag", output, tensor, sizes))
                return output
        self.module = types.SimpleNamespace(PyNcclCommunicator=Communicator)
        self.comm = Communicator()

    def test_all_three_entrypoints_preserve_arguments_and_return_values(self):
        tensor, output = object(), object()
        with patch.object(self.adapter, "_before_collective",
                          side_effect=lambda kind, stream: self.events.append((kind, stream))):
            self.adapter.install_nccl(self.module)
            self.assertIs(self.comm.all_reduce(tensor, op="max"), tensor)
            self.assertIs(self.comm.outplace_all_reduce(tensor, output, "sum"), output)
            self.assertIs(self.comm.all_gather(output, tensor, sizes=[2, 3]), output)
        self.assertEqual(self.events, [
            ("ar", "current-communication-stream"), ("original_ar", tensor, "max"),
            ("ar", "current-communication-stream"), ("original_outplace", tensor, output, "sum"),
            ("ag", "current-communication-stream"), ("original_ag", output, tensor, [2, 3])])
        self.lib_mock.assert_called_once_with()

    def test_disabled_communicator_never_advances_or_forks_prefetch(self):
        self.adapter.install_nccl(self.module)
        self.comm.disabled = True
        with patch.object(self.adapter, "_before_collective") as before:
            self.comm.all_reduce(None)
            self.comm.outplace_all_reduce(None)
            self.comm.all_gather(None, None)
        before.assert_not_called()

    def test_install_is_idempotent(self):
        self.adapter.install_nccl(self.module)
        method = self.module.PyNcclCommunicator.all_reduce
        self.adapter.install_nccl(self.module)
        self.assertIs(method, self.module.PyNcclCommunicator.all_reduce)
        self.lib_mock.assert_called_once_with()

    def test_main_gate_off_leaves_class_untouched(self):
        original = self.module.PyNcclCommunicator.all_reduce
        with patch.dict(os.environ, {"DSV41_L2_PREFETCH": "0"}):
            self.adapter.install_nccl(self.module)
        self.assertIs(original, self.module.PyNcclCommunicator.all_reduce)
        self.lib_mock.assert_not_called()

    def test_rocenante_path_is_untouched(self):
        original = self.module.PyNcclCommunicator.all_reduce
        with patch.dict(os.environ, {"SGLANG_ROCE_ALLREDUCE": "1"}):
            self.adapter.install_nccl(self.module)
        self.assertIs(original, self.module.PyNcclCommunicator.all_reduce)
        self.lib_mock.assert_not_called()

    def test_unset_transport_uses_native_nccl(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ROCE_ALLREDUCE", None)
            self.adapter.install_nccl(self.module)
        self.lib_mock.assert_called_once_with()

    def test_original_exceptions_propagate(self):
        def failing(*args, **kwargs):
            raise ValueError("original collective error")
        self.module.PyNcclCommunicator.all_reduce = failing
        self.adapter.install_nccl(self.module)
        with patch.object(self.adapter, "_before_collective"):
            with self.assertRaisesRegex(ValueError, "original collective error"):
                self.comm.all_reduce(None)

    def test_real_planner_records_collective_and_weight_order(self):
        self.adapter.install_nccl(self.module)
        state = self.adapter._state
        signature = ("fake-model", 6, "target-verify")
        state["sig"] = signature
        state["learning"][signature] = []
        class Weight:
            is_cuda = True
            shape = (64, 64)
            def dim(self): return 2
            def data_ptr(self): return 1234
            def numel(self): return 4096
            def element_size(self): return 1
        self.comm.all_reduce(None)
        self.adapter._record_linear(Weight(), None)
        self.comm.all_gather(None, None)
        self.assertEqual(state["learning"][signature], [
            ("c", 1), ("r", 1234, 4096, 64), ("c", 2)])
        self.assertEqual(state["idx"], 2)

    def test_outside_forward_has_no_cuda_work(self):
        self.adapter.install_nccl(self.module)
        with patch.object(self.adapter, "_fork") as fork:
            self.comm.all_reduce(None)
        fork.assert_not_called()
        self.assertIsNone(self.adapter._state["sig"])

    def test_pinned_real_nccl_method_bodies_preserve_pointer_calls(self):
        here = Path(__file__).resolve().parent
        source = here.parents[1] / (
            "artifacts/sglang-native/recipe-cfe3cfb/runtime/sglang-canary/python/"
            "sglang/srt/distributed/device_communicators/pynccl.py")
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "PyNcclCommunicator")
        names = {"all_reduce", "outplace_all_reduce", "all_gather", "_resolve_stream"}
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in cls.body}, names)
        future = ast.ImportFrom(module="__future__",
                                names=[ast.alias(name="annotations")], level=0)
        code = compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                       str(source), "exec")
        stream = types.SimpleNamespace(cuda_stream=42)
        enum = types.SimpleNamespace(from_torch=lambda value: value)
        namespace = {"ReduceOp": types.SimpleNamespace(SUM="sum"),
                     "get_current_device_stream_fast": lambda: stream,
                     "buffer_type": lambda value: value, "cudaStream_t": lambda value: value,
                     "ncclDataTypeEnum": enum, "ncclRedOpTypeEnum": enum}
        exec(code, namespace)
        module = types.SimpleNamespace(PyNcclCommunicator=namespace["PyNcclCommunicator"])
        self.adapter.install_nccl(module)
        comm = module.PyNcclCommunicator()
        comm.disabled = False
        comm.device = self.adapter.torch.device("cpu")
        comm.comm = "communicator"
        comm.nccl = types.SimpleNamespace(
            ncclAllReduce=lambda *args: self.events.append(("ar", args)),
            ncclAllGather=lambda *args: self.events.append(("ag", args)))
        tensor = self.adapter.torch.ones(3)
        output = self.adapter.torch.zeros(3)
        with patch.object(self.adapter, "_before_collective") as before:
            comm.all_reduce(tensor)
            self.assertIs(comm.outplace_all_reduce(tensor, output), output)
            comm.all_gather(output, tensor)
        self.assertEqual([call.args for call in before.call_args_list],
                         [("ar", stream), ("ar", stream), ("ag", stream)])
        self.assertEqual([item[0] for item in self.events], ["ar", "ar", "ag"])
        self.assertEqual(self.events[0][1][:3], (tensor.data_ptr(), tensor.data_ptr(), 3))
        self.assertEqual(self.events[1][1][:3], (tensor.data_ptr(), output.data_ptr(), 3))
        self.assertEqual(self.events[2][1][:3], (tensor.data_ptr(), output.data_ptr(), 3))
        self.assertTrue(all(item[1][-1] == 42 for item in self.events))

    def test_import_hook_installs_before_optional_roce_gather(self):
        source = (self.context / "adapter/sitecustomize.py").read_text()
        block = source.split("elif module.__name__ == 'sglang.srt.distributed.device_communicators.pynccl':", 1)[1]
        block = block.split("        elif ", 1)[0]
        self.assertLess(block.index("install_l2_prefetch_nccl(module)"),
                        block.index("install_roce_gather(module)"))


if __name__ == "__main__":
    unittest.main()
