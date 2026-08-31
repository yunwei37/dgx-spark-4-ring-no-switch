#!/usr/bin/env python3
"""Bounded synthetic SGLang decode measurement; emits no generated prose.

This client measures an existing private GPU server. It does not deploy models,
modify hosts, act as a development agent, or establish code-review quality.
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request
import uuid


def post(url, payload, timeout):
    return urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout)


def generation(endpoint, subject, output_tokens, timeout):
    payload = {
        "text": f"Record {uuid.uuid4().hex}. Explain {subject} in a detailed technical article.\nArticle:",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": output_tokens, "ignore_eos": True},
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    final = None
    with post(endpoint + "/generate", payload, timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            final = json.loads(data)
            if first is None and final.get("meta_info", {}).get("completion_tokens", 0) > 0:
                first = time.perf_counter()
    end = time.perf_counter()
    if final is None or first is None:
        raise RuntimeError("No generated streaming event")
    meta = final["meta_info"]
    tokens = meta["completion_tokens"]
    if tokens != output_tokens:
        raise RuntimeError(f"Expected {output_tokens} output tokens, got {tokens}")
    text = final.get("text", "")
    return {
        "prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": tokens,
        "cached_tokens": meta.get("cached_tokens"),
        "ttft_seconds": first - start, "e2e_seconds": end - start,
        "e2e_tokens_per_second": tokens / (end - start),
        "decode_tokens_per_second_after_first_event": (tokens - 1) / (end - first),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "finish_reason": meta.get("finish_reason"),
        "output_characters": len(text),
        "alphabetic_fraction": sum(c.isalpha() for c in text) / max(1, len(text)),
        "output_omitted": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    subjects = ["safe C++ ownership", "distributed log recovery", "Linux memory profiling", "reproducible performance experiments"]
    single = generation(endpoint, subjects[0], args.output_tokens, args.timeout)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda s: generation(endpoint, s, args.output_tokens, args.timeout), subjects))
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "workload": "synthetic completion; forced fixed output length; not an agent benchmark",
        "single": single, "concurrent_requests": results,
        "four_request_seconds": elapsed,
        "aggregate_e2e_tokens_per_second": sum(x["completion_tokens"] for x in results) / elapsed,
        "nonce_per_request": True,
        "quality_note": "Character counts are diagnostics only; correctness is checked separately.",
    }, indent=2), flush=True)
