#!/usr/bin/env python3
"""Exact-window retrieval with a consistent instruction and real chat template."""
import argparse
import hashlib
import json
import time
import urllib.request
import uuid

from transformers import AutoTokenizer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--endpoint", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--prompt-tokens", type=int, default=261888)
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--timeout", type=int, default=600)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
template = tokenizer.apply_chat_template(
    [{"role": "user", "content": "<<<RECORD_SLOT>>>"}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
before, after = template.split("<<<RECORD_SLOT>>>")
marker = "KEY_" + uuid.uuid4().hex[:16].upper()
prefix = encode(before + "Request nonce " + uuid.uuid4().hex + ". Read the entire record. Return only the exact retrieval key asked for at the end.\n")
needle = encode("\nIMPORTANT RECORD: The exact retrieval key is " + marker + ".\n")
suffix = encode("\nWhat is the exact retrieval key? Return only that key, with no explanation.\n" + after)
filler = encode("This ordinary archival line is background information and does not change the retrieval key.\n")
remaining = args.prompt_tokens - len(prefix) - len(needle) - len(suffix)
if remaining <= 0:
    raise ValueError("Prompt budget too short")
repeat = lambda n: (filler * ((n + len(filler) - 1) // len(filler)))[:n]
mid = remaining // 2
input_ids = prefix + repeat(mid) + needle + repeat(remaining - mid) + suffix
assert len(input_ids) == args.prompt_tokens
payload = {"input_ids": input_ids, "stream": True,
           "sampling_params": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}}
request = urllib.request.Request(args.endpoint.rstrip("/") + "/generate",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
start = time.perf_counter()
first = None
final = None
with urllib.request.urlopen(request, timeout=args.timeout) as response:
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
    raise RuntimeError("No generation")
text = final.get("text", "").strip()
meta = final["meta_info"]
passed = text == marker and meta.get("prompt_tokens") == args.prompt_tokens
print(json.dumps({
    "passed": passed, "exact_final_answer": text == marker,
    "prompt_tokens_requested": args.prompt_tokens,
    "prompt_tokens_reported": meta.get("prompt_tokens"),
    "native_window_budget": args.prompt_tokens + args.max_new_tokens,
    "marker_token_offset": len(prefix) + mid,
    "completion_tokens": meta.get("completion_tokens"),
    "cached_tokens": meta.get("cached_tokens"),
    "ttft_seconds": first - start, "e2e_seconds": end - start,
    "prefill_tokens_per_second": args.prompt_tokens / (first - start),
    "spec_accept_rate": meta.get("spec_accept_rate"),
    "finish_reason": meta.get("finish_reason"),
    "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    "output_omitted": True,
    "chat_template": "checkpoint tokenizer; enable_thinking=false",
    "ignore_eos": False,
}, indent=2), flush=True)
if not passed:
    raise SystemExit("Strict retrieval failed; marker-in-reasoning is insufficient")
