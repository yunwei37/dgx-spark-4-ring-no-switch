# Loader and memory results

The first default loading path took approximately two hours. Storage was not
the only limit: representative rank-local reads reached 686-839 MiB/s while the
completed loader averaged only about 174 MiB/s per rank. Allocation lifetime
and deserialization work were material parts of the critical path.

The successful bounded change releases the CUDA allocator's unused cached
blocks after a completed safetensors file batch is closed. It is applied by a
hash-guarded build script so a different upstream file fails the build instead
of receiving a blind text patch.

A broader global CUDA-cache disable loaded but reduced decode to approximately
1.88 tokens/s, so it was rejected. Periodic host page-cache dropping was an
earlier experiment and is not retained. No swap manager or OOM daemon was
needed for the successful 8K INT4 result, and none is installed here.

The optimized base load still took 589.25 seconds, not one minute. The data does
not support a one-minute claim. MTP loading took 1,183.05 seconds because its
memory and loading work differed.
