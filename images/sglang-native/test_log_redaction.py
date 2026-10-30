"""Execute the staged diagnostic renderers without importing model code."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/sglang-native/v262/context/python/sglang/srt"
)


def renderers():
    tree = ast.parse((ROOT / "server_args.py").read_text())
    nodes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("resolved_dict", "launch_command")
    ]
    for node in nodes:
        node.returns = None
        node.decorator_list = []
    scope = {
        "_plain": lambda x: x,
        "record_fields": lambda _: [
            SimpleNamespace(name=n)
            for n in ("api_key", "admin_api_key", "model_path", "context_length")
        ],
        "resolution_result": lambda obj, name: getattr(obj, name),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "staged-diagnostics", "exec"),
        scope,
    )
    return scope


class LogRedaction(unittest.TestCase):
    def test_resolved_diagnostics_hide_only_auth_fields_without_mutation(self):
        methods = renderers()
        for key, admin in [(None, None), ("", ""), ("fixture-key", "管理-fixture")]:
            values = {
                "api_key": key,
                "admin_api_key": admin,
                "model_path": "/models/api_key-fixture",
                "context_length": 1048576,
            }
            before = copy.deepcopy(values)
            rendered = methods["resolved_dict"](SimpleNamespace(**values))
            self.assertEqual(values, before)
            self.assertEqual(rendered["model_path"], values["model_path"])
            self.assertEqual(rendered["context_length"], 1048576)
            for field in ("api_key", "admin_api_key"):
                self.assertEqual(
                    rendered[field], "<redacted>" if values[field] else values[field]
                )

    def test_launch_diagnostics_redact_cli_and_engine_auth_arguments(self):
        method = renderers()["launch_command"]
        commands = [
            "python -m sglang.launch_server --api-key fixture-secret --admin-api-key='管理 secret' --context-length 1048576",
            "sglang --api-key=fixture-secret --model-path '/models/model with spaces' --admin-api-key '管理 secret'",
            "Engine(api_key='fixture-secret', model_path='/models/api_key-fixture', admin_api_key='管理 secret', context_length=1048576)",
        ]
        for command in commands:
            obj = SimpleNamespace(_launch_command=command)
            rendered = method(obj)
            self.assertNotIn("fixture-secret", rendered)
            self.assertNotIn("管理 secret", rendered)
            self.assertIn("<redacted>", rendered)
            self.assertEqual(obj._launch_command, command)
        self.assertIsNone(method(SimpleNamespace(_launch_command=None)))
        self.assertEqual(
            method(SimpleNamespace(_launch_command="sglang --context-length 1048576")),
            "sglang --context-length 1048576",
        )

    def test_engine_logs_and_http_diagnostics_share_the_safe_renderers(self):
        tree = ast.parse((ROOT / "entrypoints/engine.py").read_text())
        logs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.JoinedStr)
            and any(
                isinstance(v, ast.Constant) and v.value == "server_args="
                for v in n.values
            )
        ]
        self.assertEqual(len(logs), 2)
        for log in logs:
            calls = [n for n in ast.walk(log) if isinstance(n, ast.Call)]
            self.assertEqual(calls[0].func.attr, "resolved_dict")
        http = (ROOT / "entrypoints/http_server.py").read_text()
        self.assertIn("**server_args.resolved_dict()", http)
        self.assertIn('"launch_command": server_args.launch_command', http)


if __name__ == "__main__":
    unittest.main()
