# GLM-5.2 on four DGX Sparks without a switch

The useful result of this experiment is not merely that a 504B-class model
produced tokens. It is a measured operating envelope for four unified-memory
machines connected as a direct ring—and a record of where that envelope ends.

For `QuantTrio/GLM-5.2-Int4-Int8Mix`, TP=4 and the direct NCCL Mesh ring reached
12.99 decode tokens/s at 8K context. Four-token MTP reached 33.42 tokens/s on a
fixed workload, though mixed prompts landed between 16.37 and 23.71 tokens/s.
That gap matters: speculative decoding depends on acceptance, not a toggle.

Loading exposed a second bottleneck. Local reads could exceed 680 MiB/s per
rank, yet end-to-end loading was far slower. Releasing CUDA cache after each
closed file batch reduced the base load to 589 seconds. Disabling CUDA caching
globally was worse for steady-state inference and was discarded. The result is
still about ten minutes, so it would be misleading to call loading solved or
claim a one-minute startup.

True ModelOpt NVFP4 ran through a different path: PP=4/TP=1. At 32,005 tokens it
stabilized near 4.40 decode tokens/s and passed the agent/tool test. At 60,469
tokens, unified-memory pressure delayed management traffic and forced recovery
of a rank. That is a failed safety boundary, not an extra benchmark point.

The service present before these experiments was restored afterward. The
reproduction files intentionally add no swap loops, OOM daemons, cache-dropping
timers, network reconcilers, or firewall policy. A direct ring is already a
special topology; it does not need invisible management machinery on top.
