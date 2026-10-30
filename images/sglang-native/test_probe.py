"""Protocol-only streaming checks; no model or GPU requests."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probe import main, read_stream


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class StreamTest(unittest.TestCase):
    def test_full_probe_wire_shapes_and_tool_history_snapshot(self):
        wire = []
        messages = [
            {"content": "42", "reasoning_content": "Synthetic reasoning."},
            {"content": "42"},
            {"content": "42"},
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture-call",
                        "type": "function",
                        "function": {
                            "name": "lookup_fixture",
                            "arguments": '{"key":"alpha"}',
                        },
                    }
                ],
            },
            {"content": "The value is 42."},
            {"content": '{"answer":42}'},
            None,
            {"content": "A red circle on the left and a blue square on the right."},
        ]

        def respond(req):
            body = json.loads(req.data)
            wire.append(body)
            if body.get("stream"):
                chunks = event(
                    {
                        "choices": [
                            {"delta": {"reasoning_content": "Synthetic reasoning."}}
                        ]
                    }
                )
                chunks += event(
                    {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]}
                )
                chunks += event(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                )
                return io.BytesIO(chunks + b"data: [DONE]\n\n")
            return io.BytesIO(
                json.dumps({"choices": [{"message": messages[len(wire) - 1]}]}).encode()
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "synthetic-only.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "probe.py",
                        "--base-url",
                        "http://unused.invalid/v1",
                        "--output",
                        str(output),
                    ],
                ),
                patch("urllib.request.urlopen", respond),
            ):
                main()
            evidence = json.loads(output.read_text())
        self.assertTrue(evidence["passed"])
        self.assertEqual(len(wire), 8)
        self.assertEqual(len(wire[3]["messages"]), 1)
        self.assertEqual(
            evidence["tests"][3]["request"]["messages"], wire[3]["messages"]
        )
        self.assertEqual(len(wire[4]["messages"]), 3)
        self.assertEqual(wire[5]["response_format"]["type"], "json_schema")
        self.assertEqual(wire[7]["messages"][0]["content"][1]["type"], "image_url")
        self.assertTrue(all(body["model"] == "local-large" for body in wire))

    def test_reasoning_content_finish_and_separate_usage(self):
        body = b": heartbeat\n\n" + event(
            {"choices": [{"delta": {"role": "assistant"}}]}
        )
        for delta in [{"reasoning_content": "19+23=42."}, {"content": "42"}]:
            body += event({"choices": [{"delta": delta}]})
        body += event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        body += event(
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 8}}
        )
        result = read_stream(io.BytesIO(body + b"data: [DONE]\n\n"))
        self.assertTrue(result["done"])
        self.assertEqual(result["content"], "42")
        self.assertEqual(result["reasoning_content"], "19+23=42.")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 8)
        self.assertIsNotNone(result["first_token_seconds_after_headers"])

    def test_error_and_unfinished_stream_are_recorded(self):
        result = read_stream(
            io.BytesIO(event({"error": {"message": "synthetic failure"}}))
        )
        self.assertFalse(result["done"])
        self.assertEqual(result["error"]["message"], "synthetic failure")
        self.assertEqual(len(result["chunks"]), 1)

    def test_multiline_sse_json(self):
        result = read_stream(
            io.BytesIO(
                b'data: {"choices": [],\ndata: "usage": {"prompt_tokens": 9}}\n\ndata: [DONE]\n\n'
            )
        )
        self.assertTrue(result["done"])
        self.assertEqual(result["usage"], {"prompt_tokens": 9})


if __name__ == "__main__":
    unittest.main(verbosity=2)
