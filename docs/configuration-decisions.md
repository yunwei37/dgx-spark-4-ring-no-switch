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

Swap, periodic drop-cache loops, OOM daemons, floating autotuning parameters,
host firewall changes, and topology reconcilers are not retained. The available
evidence does not justify adding those entities to a reusable serving recipe.
