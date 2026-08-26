# Repository rules

- Work directly on `main`; do not create a branch or worktree unless the user
  explicitly asks for one.
- Keep this repository inference-only. Do not add training, host management,
  firewall, login, monitoring, or general cluster-administration machinery.
- Never commit model weights, credentials, private addresses, hostnames, raw
  prompts, responses, or private storage identifiers.
- Prefer runtime and operating-system defaults. Keep a non-default setting only
  when its rationale and matched A/B evidence are recorded under `docs/` and
  `benchmarks/`.
- Do not add a persistent daemon, timer, cache-drop loop, OOM killer, swap
  configuration, route reconciler, or network dispatcher as a side effect of a
  serving recipe. Such mechanisms may appear only as explicit experimental
  candidates until a bounded test demonstrates a material benefit without a
  stability or management-plane regression.
- Treat `partial`, `failed`, `unsafe`, and `not measured` as distinct from zero
  or success. Published performance numbers must name the exact model revision,
  image, profile, workload, and measurement boundary.
- Images and source dependencies must use immutable versions or digests; never
  publish a recipe that depends on `latest`.

