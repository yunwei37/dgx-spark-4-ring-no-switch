# Configuration decisions

Only settings that belonged to the successful test or a measured alternative
are retained here.

| Setting | Why retained | Evidence |
| --- | --- | --- |
| TP=4 for INT4 | One tensor rank per Spark matched the direct ring and passed | 12.99 tok/s base; 33.42 tok/s fixed MTP |
| NCCL Ring + Mesh plugin | Exercised both direct neighbors without a switch | Four ranks completed load and decode |
| `fastsafetensors` | Rank-selective sharded load was the successful path | 589.25 s base load |
| 2 GiB explicit KV cache | Tested 8K envelope with bounded memory | 94.38 GiB/rank base |
| full CUDA graph | Present in the successful measured profile | Three stable base trials |
| FP8 DeepSeek MLA KV cache | Present in the successful measured profile | Same bounded run |
| per-batch `torch.cuda.empty_cache()` during load | Prevented stale batch allocations without disabling caching globally | Successful load; global cache disable fell to about 1.88 tok/s |
| MTP=4, optional | Material speedup on the fixed workload but low reserve | 33.42 tok/s and 3.3-5.0 GiB free/rank |
| Qwen MTP 1 step / 2 draft tokens | Full-window retrieval and natural long output both passed | 48.81 tok/s, 86.86% acceptance |
| Qwen MTP 3 steps / 4 draft tokens rejected | Higher apparent speed was accompanied by deterministic output degeneration | 63.67 tok/s, but marker recovery failed twice |

Swap, periodic drop-cache loops, OOM daemons, floating autotuning parameters,
host firewall changes, and topology reconcilers are not retained. The available
evidence does not justify adding those entities to a reusable serving recipe.

The optional Qwen MTP file-view helper is a tested, bounded experiment, not a
new default profile. Native draft loading fell to 19–20s versus the historical
839–842s, with strict 262K retrieval passing; this was not matched-cache A/B.
The target stays complete/read-only and no model data is copied. Runtime JIT
warning and reduced effective token/concurrency pool remain disclosed in the
[experiment](blog/2026-08-31-qwen-mtp-file-selection.md).
