"""Score frozen single-token MMLU choices through SGLang's existing native API.

No server, chat template, answer mask, prompt logprobs, or serving change.
``attach`` replaces only an existing lm_eval 0.4.13 client's token-scoring method.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MethodType
import urllib.request

CANDIDATES = (334, 406, 345, 420)


def payload(input_ids, new_tokens=1):
    if not input_ids or any(type(x) is not int or x < 0 for x in input_ids):
        raise ValueError("Expected nonempty, already-tokenized context IDs")
    return {
        "input_ids": list(input_ids),
        "sampling_params": {"temperature": 0, "max_new_tokens": new_tokens},
        "return_logprob": True,
        "logprob_start_len": len(input_ids),
        "token_ids_logprob": list(CANDIDATES),
        "top_logprobs_num": 1,
        "return_text_in_logprobs": False,
        "stream": False,
    }


def post_json(url, body, headers=None):
    headers = {"Content-Type": "application/json", **(headers or {})}
    key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("SGLANG_API_KEY")
        or os.environ.get("VLLM_API_KEY")
    )
    if key and "Authorization" not in headers:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def parse_response(response, prompt_length):
    meta = response["meta_info"]
    if meta.get("prompt_tokens") != prompt_length:
        raise ValueError("Native prompt token count differs from frozen input_ids")
    if meta.get("input_token_logprobs") or meta.get("input_token_ids_logprobs"):
        raise ValueError("This scoring protocol must not request prompt logprobs")
    rows = meta["output_token_ids_logprobs"]
    if len(rows) != 1:
        raise ValueError(f"Expected one next-token distribution, got {len(rows)}")
    values = {}
    for item in rows[0]:
        value, token_id = item[:2]
        if type(token_id) is not int or token_id in values or not math.isfinite(value):
            raise ValueError("Invalid, duplicate or non-finite candidate probability")
        values[token_id] = float(value)
    if set(values) != set(CANDIDATES):
        raise ValueError("The server did not return every requested candidate ID")
    top = meta["output_top_logprobs"][0]
    if len(top) != 1 or type(top[0][1]) is not int or not math.isfinite(top[0][0]):
        raise ValueError("Expected global top-1 for lm_eval's is_greedy result")
    return [values[t] for t in CANDIDATES], top[0][1]


def score_context(input_ids, native_url, new_tokens=1, post=None):
    body = payload(input_ids, new_tokens)
    start = time.monotonic()
    response = (post or post_json)(native_url, body)
    scores, greedy_id = parse_response(response, len(input_ids))
    return {
        "protocol": "sglang-native-next-token-v1",
        "input_ids_sha256": hashlib.sha256(
            json.dumps(list(input_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "request": body,
        "response": response,
        "scores": scores,
        "greedy_token_id": greedy_id,
        "seconds": time.monotonic() - start,
    }


def loglikelihood_tokens(lm, requests, native_url, evidence_path=None, post=None):
    groups = {}
    for index, (cache_key, context, continuation) in enumerate(requests):
        if len(continuation) != 1 or continuation[0] not in CANDIDATES:
            raise ValueError(
                "Native MMLU scoring requires one of the four single-token answers; no prompt-logprob fallback"
            )
        groups.setdefault(tuple(context), []).append(
            (index, cache_key, continuation[0])
        )
    results = [None] * len(requests)
    # Reuse the existing evaluation client's concurrency. No new serving limit.
    concurrency = getattr(lm, "_concurrent", 1)

    def run(context):
        return score_context(context, native_url, post=post)

    contexts = list(groups)
    evidence = None
    if evidence_path:
        evidence_path = Path(evidence_path)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = evidence_path.open("a")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for context, record in zip(
                contexts, executor.map(run, contexts), strict=True
            ):
                scores = dict(zip(CANDIDATES, record["scores"], strict=True))
                record["lm_eval_cache_keys"] = [entry[1] for entry in groups[context]]
                if evidence:
                    evidence.write(json.dumps(record, ensure_ascii=False) + "\n")
                    evidence.flush()
                for index, cache_key, token in groups[context]:
                    result = (scores[token], record["greedy_token_id"] == token)
                    results[index] = result
                    if cache_key is not None:
                        lm.cache_hook.add_partial("loglikelihood", cache_key, result)
    finally:
        if evidence:
            evidence.close()
    return results


def attach(lm, native_url, evidence_path):
    """Keep lm_eval's tokenizer, _encode_pair, tasks, seeds and request order."""

    def method(self, requests, **kwargs):
        def post(url, body):
            return post_json(url, body, headers=self.header)

        return loglikelihood_tokens(self, requests, native_url, evidence_path, post)

    lm._loglikelihood_tokens = MethodType(method, lm)
    return lm


def freeze_samples(paths, tokenizer_path, output, reference_method, add_bos=False):
    # Use the exact harness algorithm, including moving trailing context spaces.
    from lm_eval.api.model import TemplateLM
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    class Encoder:
        backend = "causal"

        def tok_encode(self, text):
            return tokenizer.encode(text, add_special_tokens=add_bos)

    encoder = Encoder()
    with Path(output).open("w") as out:
        for path in paths:
            with Path(path).open() as source:
                for line in source:
                    sample = json.loads(line)
                    pairs = [
                        TemplateLM._encode_pair(encoder, *arg)
                        for arg in sample["arguments"]
                    ]
                    context = pairs[0][0]
                    if any(ctx != context for ctx, _ in pairs) or [
                        cont for _, cont in pairs
                    ] != [[t] for t in CANDIDATES]:
                        raise ValueError(
                            "Sample is not four choices sharing the same tokenized context"
                        )
                    row = {
                        "sample_id": f"{Path(path).name}:{sample['doc_id']}:{sample.get('filter', '')}",
                        "input_ids": context,
                        "candidate_token_ids": list(CANDIDATES),
                        "continuations": [arg[1] for arg in sample["arguments"]],
                        "prompt_hash": sample.get("prompt_hash"),
                        "reference_method": reference_method,
                        "reference_scores": [
                            float(x[0]) for x in sample["filtered_resps"]
                        ],
                        "answer_index": sample.get("doc", {}).get("answer"),
                        "tokenizer_path": str(tokenizer_path),
                        "add_bos_token": add_bos,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")


def replay_frozen(prompts, native_url, output, new_tokens=1, post=None):
    total = compared = disagreements = scored = correct = 0
    with Path(prompts).open() as source, Path(output).open("w") as out:
        for line in source:
            sample = json.loads(line)
            if sample["candidate_token_ids"] != list(CANDIDATES):
                raise ValueError("Frozen candidate IDs differ from the MMLU protocol")
            record = score_context(sample["input_ids"], native_url, new_tokens, post)
            winner = max(range(4), key=record["scores"].__getitem__)
            record["sample"] = sample
            record["winner_index"] = winner
            ordered = sorted(record["scores"], reverse=True)
            record["choice_margin"] = ordered[0] - ordered[1]
            reference = sample.get("reference_scores")
            if reference is not None:
                ref_winner = max(range(4), key=reference.__getitem__)
                record["reference_winner_index"] = ref_winner
                record["score_delta"] = [
                    a - b for a, b in zip(record["scores"], reference, strict=True)
                ]
                record["winner_changed"] = ref_winner != winner
                ref_ordered = sorted(reference, reverse=True)
                record["reference_choice_margin"] = ref_ordered[0] - ref_ordered[1]
                compared += 1
                disagreements += int(record["winner_changed"])
            target = sample.get("answer_index")
            if type(target) is int:
                scored += 1
                correct += int(winner == target)
            total += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return {
        "contexts": total,
        "compared": compared,
        "reference_disagreements": disagreements,
        "scored": scored,
        "correct": correct,
        "new_tokens": new_tokens,
        "scope": "Native next-token probabilities; no teacher-forced numerical-equivalence claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="Freeze saved lm_eval 0.4.13 log_samples without GPU calls"
    )
    freeze.add_argument("--samples", nargs="+", required=True)
    freeze.add_argument("--tokenizer", required=True)
    freeze.add_argument("--reference-method", required=True)
    freeze.add_argument("--add-bos-token", action="store_true")
    freeze.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="Run only after the GPU owner releases the window"
    )
    replay.add_argument("--prompts", type=Path, required=True)
    replay.add_argument(
        "--native-url", required=True, help="Existing backend's /generate URL"
    )
    replay.add_argument("--new-tokens", type=int, choices=[0, 1], default=1)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        freeze_samples(
            args.samples,
            args.tokenizer,
            args.output,
            args.reference_method,
            args.add_bos_token,
        )
    else:
        summary = replay_frozen(
            args.prompts, args.native_url, args.output, args.new_tokens
        )
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
