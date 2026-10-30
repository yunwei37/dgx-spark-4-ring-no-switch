#!/usr/bin/env python3
"""Record business API evidence after the owner hands over the GPU window."""

import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import time
import urllib.request


def read_stream(response):
    """Keep raw chunks and distinguish reasoning/content from role-only events."""
    chunks, content, reasoning = [], [], []
    finish = usage = error = None
    started = time.monotonic()
    ttft = None
    data = []
    done = False
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            continue
        if line or not data:
            continue
        event = "\n".join(data)
        data = []
        if event == "[DONE]":
            done = True
            break
        chunk = json.loads(event)
        chunks.append(chunk)
        error = chunk.get("error", error)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if ttft is None and (
                delta.get("content") or delta.get("reasoning_content")
            ):
                ttft = time.monotonic() - started
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = choice.get("finish_reason") or finish
    return {
        "chunks": chunks,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "finish_reason": finish,
        "usage": usage,
        "done": done,
        "error": error,
        "first_token_seconds_after_headers": ttft,
        "body_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="OpenAI base URL, including /v1"
    )
    parser.add_argument("--model", default="local-large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    secret = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("SGLANG_API_KEY")
        or os.environ.get("VLLM_API_KEY")
    )
    if secret:
        headers["Authorization"] = "Bearer " + secret
    evidence = {
        "base_url": args.base_url,
        "model": args.model,
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
        "tests": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def chat(label, **fields):
        # Callers extend tool histories after this call; preserve the wire input.
        payload = copy.deepcopy(
            dict(model=args.model, temperature=0, max_tokens=2048, **fields)
        )
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        started = time.monotonic()
        with urllib.request.urlopen(req) as response:
            result = (
                read_stream(response) if fields.get("stream") else json.load(response)
            )
        evidence["tests"].append(
            {
                "name": label,
                "request": payload,
                "response": result,
                "seconds": time.monotonic() - started,
            }
        )
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
        )
        return result if fields.get("stream") else result["choices"][0]["message"]

    arithmetic = [dict(role="user", content="What is 19 + 23? Reply with the number.")]
    default = chat("default thinking", messages=arithmetic)
    assert default.get("reasoning_content"), (
        "Omitted thinking must return a separate reasoning field"
    )
    assert "42" in (default.get("content") or ""), default
    off = chat(
        "thinking explicitly off",
        messages=arithmetic,
        chat_template_kwargs={"thinking": False},
    )
    assert not off.get("reasoning_content") and "42" in (off.get("content") or ""), off
    alias = chat(
        "vLLM thinking alias off",
        messages=arithmetic,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert not alias.get("reasoning_content") and "42" in (
        alias.get("content") or ""
    ), alias
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_fixture",
            "description": "Look up a fixture value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }
    messages = [
        dict(
            role="user",
            content="Use lookup_fixture to get the value for key alpha. Do not guess.",
        )
    ]
    called = chat(
        "tool call",
        messages=messages,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    calls = called.get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_fixture", called
    assert json.loads(calls[0]["function"]["arguments"]) == {"key": "alpha"}, called
    messages += [
        called,
        dict(role="tool", tool_call_id=calls[0]["id"], content='{"value":42}'),
    ]
    answer = chat(
        "tool continuation",
        messages=messages,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    assert "42" in (answer.get("content") or ""), answer
    structured = chat(
        "structured JSON schema",
        messages=[
            dict(
                role="user", content="Return an object whose answer is the integer 42."
            )
        ],
        chat_template_kwargs={"thinking": False},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    )
    assert json.loads(structured["content"]) == {"answer": 42}, structured
    streamed = chat(
        "stream default thinking and usage",
        messages=arithmetic,
        stream=True,
        stream_options={"include_usage": True},
    )
    assert streamed["done"] and not streamed["error"] and streamed["finish_reason"], (
        streamed
    )
    assert streamed["reasoning_content"] and "42" in streamed["content"], streamed
    assert streamed["usage"], "Missing requested streaming usage"
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (3024, 588), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 100, 588, 488), fill="red")
    draw.rectangle((2400, 100, 2788, 488), fill="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    visual = chat(
        "native vision shapes",
        chat_template_kwargs={"thinking": False},
        messages=[
            dict(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Describe the two colored shapes and their left-to-right order. Be concise.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(buffer.getvalue()).decode()
                        },
                    },
                ],
            )
        ],
    )
    text = (visual.get("content") or "").lower()
    assert all(word in text for word in ["red", "circle", "blue", "square"]), visual
    assert text.index("red") < text.index("blue"), visual
    evidence["passed"] = True
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(
        f"PASS: thinking, alias, tools, structured output, streaming and vision; {args.output}"
    )


if __name__ == "__main__":
    main()
