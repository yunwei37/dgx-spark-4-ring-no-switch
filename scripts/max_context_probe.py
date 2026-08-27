#!/usr/bin/env python3
"""Run an exact-token long-context retrieval probe against an SGLang server."""

from __future__ import annotations

import argparse
import json
import time

import requests
from transformers import AutoTokenizer


def repeated(tokens: list[int], length: int) -> list[int]:
    if not tokens:
        raise ValueError("filler must tokenize to at least one token")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def build_prompt(
    tokenizer, target_tokens: int, marker: str, long_response: bool
) -> tuple[list[int], int]:
    prefix = tokenizer.encode(
        "Read the complete archival record. Retain the exact retrieval key and "
        "answer the final question with only that key.\n",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        "This archival line contains ordinary background information and does not "
        "change the retrieval key.\n",
        add_special_tokens=False,
    )
    needle = tokenizer.encode(
        f"\nIMPORTANT RECORD: The exact retrieval key is {marker}.\n",
        add_special_tokens=False,
    )
    question_text = (
        "\nQuestion: Begin with the exact retrieval key, then write a detailed "
        "technical explanation of at least 220 tokens about how to preserve and "
        "verify that key in an archival system.\nAnswer:"
        if long_response
        else "\nQuestion: What is the exact retrieval key? Answer only the key.\nAnswer:"
    )
    question = tokenizer.encode(question_text, add_special_tokens=False)
    fixed = len(prefix) + len(needle) + len(question)
    if fixed >= target_tokens:
        raise ValueError(f"target_tokens={target_tokens} is too small for the probe")
    filler_total = target_tokens - fixed
    before = filler_total // 2
    prompt = prefix + repeated(filler, before) + needle
    marker_offset = len(prompt) - len(needle)
    prompt += repeated(filler, filler_total - before) + question
    assert len(prompt) == target_tokens
    return prompt, marker_offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--long-response", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids, marker_offset = build_prompt(
        tokenizer, args.prompt_tokens, args.marker, args.long_response
    )
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": args.ignore_eos,
        },
        "stream": True,
    }

    started = time.perf_counter()
    first_event = None
    final = None
    with requests.post(
        f"{args.endpoint.rstrip('/')}/generate",
        json=payload,
        stream=True,
        timeout=args.timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            if first_event is None:
                first_event = time.perf_counter()
            final = json.loads(data)

    finished = time.perf_counter()
    if first_event is None or final is None:
        raise RuntimeError("server returned no streaming generation events")
    output = final.get("text", "")
    meta = final.get("meta_info", {})
    ttft = first_event - started
    decode_seconds = finished - first_event
    completion_tokens = meta.get("completion_tokens")
    result = {
        "prompt_tokens_requested": args.prompt_tokens,
        "prompt_tokens_reported": meta.get("prompt_tokens"),
        "max_new_tokens": args.max_new_tokens,
        "native_window_exercised": args.prompt_tokens + args.max_new_tokens,
        "marker": args.marker,
        "marker_token_offset": marker_offset,
        "marker_recovered": args.marker in output,
        "output": output,
        "ttft_seconds": ttft,
        "prefill_tokens_per_second": args.prompt_tokens / ttft,
        "e2e_seconds": finished - started,
        "completion_tokens": completion_tokens,
        "decode_seconds_after_first_event": decode_seconds,
        "decode_tokens_per_second": (
            completion_tokens / decode_seconds
            if completion_tokens is not None and decode_seconds > 0
            else None
        ),
        "cached_tokens": meta.get("cached_tokens"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "finish_reason": meta.get("finish_reason"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["prompt_tokens_reported"] != args.prompt_tokens:
        raise SystemExit("server-reported prompt token count does not match request")
    if not result["marker_recovered"]:
        raise SystemExit("retrieval marker was not recovered")


if __name__ == "__main__":
    main()
