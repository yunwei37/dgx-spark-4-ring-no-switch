"""CPU integration against lm_eval 0.4.13 and the shipped SGLang source.

Synthetic logits exercise protocol plumbing, never model accuracy or TP4 runtime.
"""

import ast
import copy
from enum import Enum
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lm_eval.models.openai_completions import LocalCompletionsAPI

import mmlu_native as native

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "artifacts/sglang-native/context/python/sglang/srt"
TOKENIZER = ROOT / "artifacts/sglang-native/official-tokenizer"


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Stage(Enum):
    DECODE = 0


def source_response(prompt_length):
    """Use actual token gather + result collector + native JSON formatter."""
    namespace = {
        "torch": torch,
        "LogprobStage": Stage,
        "LogprobResult": SimpleNamespace,
    }
    path = ENGINE / "layers/logprob_processor.py"
    for name in [
        "get_token_ids_logprobs_raw",
        "get_token_ids_logprobs",
        "get_top_logprobs_raw",
        "get_top_logprobs",
    ]:
        extract(path, name, namespace)
    compute = extract(path, "compute_logprobs", namespace)
    logits = torch.linspace(-4, 1, 129280, device="cpu").reshape(1, -1)
    logits[0, list(native.CANDIDATES)] = torch.tensor([3.0, 4.0, 2.0, 1.0])
    logits[0, 923] = 8.0  # Global argmax is outside A/B/C/D.
    logprobs = torch.log_softmax(logits, dim=-1)
    candidate_order = [420, 334, 345, 406]
    result = compute(
        None, logprobs.clone(), [1], [candidate_order], torch.tensor([923])
    )
    collector_path = (
        ENGINE / "managers/scheduler_components/logprob_result_processor.py"
    )
    collector = SimpleNamespace()
    collector._initialize_empty_logprob_containers = MethodType(
        extract(collector_path, "_initialize_empty_logprob_containers", namespace),
        collector,
    )
    add = extract(collector_path, "add_logprob_return_values", namespace)
    request = SimpleNamespace(
        logprob=SimpleNamespace(
            **{
                **{
                    f"input_{kind}_{part}": None
                    for kind in ["token_logprobs", "top_logprobs", "token_ids_logprobs"]
                    for part in ["val", "idx"]
                },
                **{
                    f"output_{kind}_{part}": []
                    for kind in ["token_logprobs", "top_logprobs", "token_ids_logprobs"]
                    for part in ["val", "idx"]
                },
                "top_logprobs_num": 1,
                "token_ids_logprob": list(native.CANDIDATES),
            }
        )
    )
    output = SimpleNamespace(
        next_token_logprobs=result.token_logprobs,
        next_token_top_logprobs_val=result.top_logprobs_val,
        next_token_top_logprobs_idx=result.top_logprobs_idx,
        next_token_token_ids_logprobs_val=result.token_ids_logprobs_val,
        next_token_token_ids_logprobs_idx=result.token_ids_logprobs_idx,
    )
    add(collector, 0, request, 0, [923], 0, output)
    serializer_path = ENGINE / "managers/tokenizer_manager.py"
    serializer = SimpleNamespace()
    serializer.detokenize_logprob_tokens = MethodType(
        extract(serializer_path, "detokenize_logprob_tokens", namespace), serializer
    )
    serialize = extract(serializer_path, "detokenize_top_logprobs_tokens", namespace)
    lp = request.logprob
    response = {
        "text": "synthetic",
        "meta_info": {
            "prompt_tokens": prompt_length,
            "input_token_logprobs": lp.input_token_logprobs_val,
            "output_token_ids_logprobs": serialize(
                serializer,
                lp.output_token_ids_logprobs_val,
                lp.output_token_ids_logprobs_idx,
                False,
            ),
            "output_top_logprobs": serialize(
                serializer,
                [v.tolist() for v in lp.output_top_logprobs_val],
                [v.tolist() for v in lp.output_top_logprobs_idx],
                False,
            ),
        },
    }
    return json.loads(json.dumps(response)), logprobs[
        0, list(native.CANDIDATES)
    ].tolist()


class NativeMMLUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response, cls.expected = source_response(3)

    def test_exact_source_gather_collector_json_and_id_mapping(self):
        scores, greedy = native.parse_response(self.response, 3)
        self.assertEqual(scores, self.expected)
        self.assertEqual(greedy, 923)
        self.assertEqual(self.response["meta_info"]["input_token_logprobs"], [])

    def test_actual_results_accept_eager_lists_and_delayed_tensors(self):
        namespace = {"torch": torch, "LogprobStage": Stage}
        producer = ENGINE / "layers/logprob_processor.py"
        extract(producer, "get_token_ids_logprobs_raw", namespace)
        gather = extract(producer, "get_token_ids_logprobs", namespace)
        processor = ENGINE / "managers/scheduler_components/batch_result_processor.py"
        logprobs = torch.log_softmax(torch.arange(15, dtype=torch.float32).reshape(3, 5), dim=-1)
        candidates = [[1, 3], None, [0]]
        expected = [logprobs[0, [1, 3]].tolist(), [], logprobs[2, [0]].tolist()]
        for phase in ("move_logprobs_to_cpu", "_normalize_decode_outputs"):
            normalize = extract(processor, phase, {"torch": torch})
            for delayed in (False, True):
                with self.subTest(phase=phase, delayed=delayed):
                    vals, idxs = gather(logprobs, candidates, no_copy_to_cpu=delayed)
                    output = SimpleNamespace(
                        input_token_logprobs=None,
                        next_token_logprobs=logprobs[:, 4],
                        next_token_top_logprobs_val=None,
                        next_token_token_ids_logprobs_val=vals,
                        next_token_token_ids_logprobs_idx=idxs,
                    )
                    batch = SimpleNamespace(
                        spec_algorithm=SimpleNamespace(is_none=lambda: True),
                        return_logprob=True,
                    )
                    if phase == "move_logprobs_to_cpu":
                        normalize(SimpleNamespace(), batch=batch, logits_output=output)
                        self.assertEqual(output.next_token_logprobs, logprobs[:, 4].tolist())
                    else:
                        tokens, scores = normalize(
                            SimpleNamespace(), batch=batch, result=None, logits_output=output,
                            next_token_ids=torch.tensor([4, 4, 4]),
                        )
                        self.assertEqual(tokens, [[4], [4], [4]])
                        self.assertEqual(scores, logprobs[:, 4].tolist())
                    self.assertEqual(output.next_token_token_ids_logprobs_val, expected)
                    self.assertEqual(output.next_token_token_ids_logprobs_idx, [[1, 3], [], [0]])

    def test_no_answer_mask_chat_template_or_prompt_logprobs(self):
        p = native.payload([0, 10, 11])
        self.assertEqual(p["input_ids"], [0, 10, 11])
        self.assertEqual(p["logprob_start_len"], 3)
        self.assertEqual(p["sampling_params"], {"temperature": 0, "max_new_tokens": 1})
        self.assertNotIn("messages", p)
        self.assertNotIn("logit_bias", p)

    def test_http_json_content_type_and_existing_authorization(self):
        def opened(req):
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.get_header("Content-type"), "application/json")
            self.assertEqual(
                req.get_header("Authorization"), "Bearer synthetic-test-only"
            )
            self.assertEqual(json.loads(req.data), native.payload([0, 10, 11]))
            return io.BytesIO(json.dumps(self.response).encode())

        with patch.object(native.urllib.request, "urlopen", opened):
            response = native.post_json(
                "http://unused.invalid/generate",
                native.payload([0, 10, 11]),
                headers={"Authorization": "Bearer synthetic-test-only"},
            )
        self.assertEqual(native.parse_response(response, 3)[0], self.expected)

    def test_actual_bounded_replay_reader_guard(self):
        class Hidden(Enum):
            FULL = 1
            NULL = 0

        fn = extract(
            ENGINE / "models/deepseek_v4.py",
            "_check_late_layer_tail_readers",
            {"CaptureHiddenMode": Hidden},
        )
        batch = SimpleNamespace(
            capture_hidden_mode=Hidden.NULL,
            return_logprob=True,
            extend_logprob_start_lens_cpu=[3],
            extend_seq_lens_cpu=[3],
        )
        fn(SimpleNamespace(), batch)
        batch.extend_logprob_start_lens_cpu = [2]
        with self.assertRaisesRegex(ValueError, "prompt tokens"):
            fn(SimpleNamespace(), batch)

    def test_actual_dspark_zero_tokens_is_not_prefill_only(self):
        spec = SimpleNamespace(speculative_algorithm="DSPARK")
        fn = extract(
            ENGINE / "managers/schedule_batch.py",
            "is_prefill_only",
            {"get_spec": lambda: spec},
        )
        req = SimpleNamespace(sampling_params=SimpleNamespace(max_new_tokens=0))
        self.assertFalse(fn(req))
        spec.speculative_algorithm = None
        self.assertTrue(fn(req))

    def test_actual_lm_eval_pair_encoding_grouping_order_and_cache(self):
        self.assertEqual(importlib.metadata.version("lm_eval"), "0.4.13")
        lm = LocalCompletionsAPI(
            model="deepseek-v4.1-flash",
            tokenizer=str(TOKENIZER),
            tokenizer_backend="huggingface",
            base_url="http://unused.invalid/v1/completions",
            max_length=1048576,
            num_concurrent=2,
        )
        self.assertEqual(
            [lm.tok_encode(" " + a) for a in "ABCD"], [[t] for t in native.CANDIDATES]
        )
        requests = [
            SimpleNamespace(args=(context, " " + a))
            for context in ["Question one\nAnswer:", "Question two\nAnswer:"]
            for a in "ABCD"
        ]
        requests.append(requests[1])
        calls, cached = [], []

        def post(url, body, headers=None):
            calls.append(copy.deepcopy(body))
            response = copy.deepcopy(self.response)
            response["meta_info"]["prompt_tokens"] = len(body["input_ids"])
            return response

        lm.cache_hook = SimpleNamespace(add_partial=lambda *args: cached.append(args))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(native, "post_json", post),
        ):
            native.attach(
                lm, "http://unused.invalid/generate", Path(tmp) / "trace.jsonl"
            )
            results = lm.loglikelihood(requests, disable_tqdm=True)
            trace = [
                json.loads(x)
                for x in (Path(tmp) / "trace.jsonl").read_text().splitlines()
            ]
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trace), 2)
        self.assertEqual(len(cached), 9)
        self.assertEqual(
            results,
            [(p, False) for p in self.expected] * 2 + [(self.expected[1], False)],
        )
        for req, context in zip(
            calls, ["Question one\nAnswer:", "Question two\nAnswer:"], strict=True
        ):
            self.assertEqual(req["input_ids"], lm.tok_encode(context))
        self.assertEqual(
            lm._encode_pair("Answer: ", "A"), lm._encode_pair("Answer:", " A")
        )

    def test_missing_nonfinite_duplicate_or_extra_positions_are_errors(self):
        bad = copy.deepcopy(self.response)
        bad["meta_info"]["output_token_ids_logprobs"][0].pop()
        with self.assertRaises(ValueError):
            native.parse_response(bad, 3)
        bad = copy.deepcopy(self.response)
        bad["meta_info"]["output_token_ids_logprobs"][0][0][0] = float("nan")
        with self.assertRaises(ValueError):
            native.parse_response(bad, 3)
        bad = copy.deepcopy(self.response)
        bad["meta_info"]["output_token_ids_logprobs"] *= 2
        with self.assertRaises(ValueError):
            native.parse_response(bad, 3)
        bad = copy.deepcopy(self.response)
        bad["meta_info"]["output_token_ids_logprobs"][0][0][1] = 334
        with self.assertRaises(ValueError):
            native.parse_response(bad, 3)

    def test_multitoken_never_falls_back_to_unsafe_prompt_scoring(self):
        with self.assertRaisesRegex(ValueError, "single-token"):
            native.loglikelihood_tokens(
                SimpleNamespace(), [(None, [1], [334, 406])], "unused"
            )

    def test_saved_samples_freeze_and_replay_records_disagreement(self):
        sample = {
            "doc_id": 7,
            "doc": {"answer": 1},
            "filter": "none",
            "prompt_hash": "reference-provenance",
            "arguments": [["A question\nAnswer:", " " + a] for a in "ABCD"],
            "filtered_resps": [[float(-i), False] for i in range(4)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "samples.jsonl").write_text(json.dumps(sample) + "\n")
            native.freeze_samples(
                [p / "samples.jsonl"], TOKENIZER, p / "frozen.jsonl", "vllm-oneforward"
            )
            frozen = json.loads((p / "frozen.jsonl").read_text())
            self.assertEqual(frozen["prompt_hash"], sample["prompt_hash"])

            def post(url, body):
                response = copy.deepcopy(self.response)
                response["meta_info"]["prompt_tokens"] = len(body["input_ids"])
                return response

            report = native.replay_frozen(
                p / "frozen.jsonl", "unused", p / "result.jsonl", post=post
            )
            self.assertEqual(report["reference_disagreements"], 1)
            self.assertEqual(report["correct"], 1)
            self.assertIn("score_delta", json.loads((p / "result.jsonl").read_text()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
