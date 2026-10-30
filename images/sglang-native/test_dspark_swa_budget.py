"""Exercise the real config resolver without loading GPU/model dependencies."""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REL = "python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py"


def resolve(source, *, v4=True, dspark=True, ids=(37, 38, 39), nextn=3):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_resolve_dflash_aux_hidden_state")
    cfg = NS(hf_config=NS(), is_deepseek_v4_arch=v4, num_nextn_predict_layers=nextn)
    dflash = NS(require_num_layers=lambda: 40, num_target_layers=None,
                resolve_target_layer_ids=lambda **kwargs: list(range(40)))
    draft = NS(require_markov=lambda: True, target_layer_ids=ids)
    env = dict(ModelConfig=NS(from_server_args=lambda *a, **kw: cfg),
               get_spec=lambda: NS(speculative_draft_model_path=None,
                                   speculative_draft_model_revision=None),
               _map_muse_target_layer_ids=lambda **kw: kw["layer_ids"],
               _resolve_dflash_draft_cell_size=lambda **kw: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<resolver>", "exec",
                 flags=__future__.annotations.compiler_flag), env)
    modules = {
        "sglang.srt.speculative.dflash_utils":
            NS(parse_dflash_draft_config=lambda **kw: dflash),
        "sglang.srt.speculative.dspark_components.dspark_config":
            NS(parse_dspark_draft_config=lambda **kw: draft),
    }
    result = NS()
    with patch.dict("sys.modules", modules):
        env[node.name](config=result, server_args=NS(),
                       model_config=NS(hf_config=NS(), hf_text_config=NS(num_hidden_layers=40)),
                       spec_algorithm=NS(is_dflash_family=lambda: True,
                                         is_dspark=lambda: dspark),
                       is_draft_worker=False)
    return result


class DraftBudgetRegression(unittest.TestCase):
    def test_real_resolver_stage_count_and_other_draft_preservation(self):
        original = (ROOT / "artifacts/sglang-native/recipe-cfe3cfb/runtime/sglang-canary" / REL).read_text()
        patched = (ROOT / "artifacts/sglang-native/v262/context" / REL).read_text()
        self.assertEqual(resolve(original).dflash_draft_num_layers, 40)
        for kwargs, expected in [
            ({}, 3),
            ({"ids": None}, 3),
            ({"ids": None, "nextn": 0}, 1),
            ({"v4": False}, 40),
            ({"dspark": False}, 40),
        ]:
            with self.subTest(**kwargs):
                result = resolve(patched, **kwargs)
                self.assertEqual(result.dflash_draft_num_layers, expected)
                if kwargs.get("dspark", True) and kwargs.get("ids", (37, 38, 39)) is not None:
                    self.assertEqual(result.dflash_target_layer_ids, [37, 38, 39])


if __name__ == "__main__":
    unittest.main()
