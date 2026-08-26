# Architecture and boundaries

The tested system is four DGX Sparks arranged as a physical cycle. Each rank
has two direct ConnectX neighbors, so the collective can traverse the cycle
without an Ethernet or InfiniBand switch.

The ring is a data-plane property. Process rendezvous, SSH, API traffic, and
normal host administration remain on the existing management plane. The
launcher therefore requires the operator to name the current management
interface and rendezvous address. It does not infer peers or install persistent
network state.

The tested INT4 profile uses one vLLM process per node with TP=4. NCCL 2.29.7
and the pinned Mesh plugin provide subnet-aware collective routing over both
direct links. The tested NVFP4 profile instead used PP=4/TP=1 because its
runtime and memory envelope differed; it is documented but is not the first
image built by this repository.

Model data is one shared logical copy reshaped into rank-selectable shards.
Weights, storage credentials, node names, IP addresses, and production service
manifests are intentionally excluded.
