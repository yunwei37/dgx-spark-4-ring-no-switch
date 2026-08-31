# SPDX-License-Identifier: Apache-2.0
# Reconstructed from the public behavioural specification in the community
# GLM-5.3 repository issue #1. No unavailable image or private source was used.


def _dsa_unpadded_drafter_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    drafter_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        assert type(spec) is SlidingWindowSpec
        if spec.page_size_padded is None:
            drafter_spec = spec
        else:
            drafter_spec = replace(spec, page_size_padded=None)
        assert drafter_spec.page_size_padded is None, (
            "DSA drafter layers must not carry page_size_padded."
        )
        drafter_specs[layer_name] = drafter_spec
    return drafter_specs


def _dsa_sliding_window_max_pages(
    spec: SlidingWindowSpec, vllm_config: VllmConfig
) -> int:
    assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
        "DCP not support sliding window."
    )
    max_model_len = vllm_config.model_config.max_model_len
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    extra_retained_tokens = getattr(spec, "extra_retained_tokens", 0)
    num_tokens = min(
        spec.sliding_window
        - 1
        + extra_retained_tokens
        + max_num_batched_tokens,
        max_model_len,
    )
    return cdiv(num_tokens, spec.block_size) + 1


def _dsa_group_max_pages(
    kv_cache_spec: KVCacheSpec, vllm_config: VllmConfig
) -> int:
    assert isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
    per_layer_specs = list(kv_cache_spec.kv_cache_specs.values())
    if not per_layer_specs:
        return 0
    if all(type(spec) is SlidingWindowSpec for spec in per_layer_specs):
        assert all(
            spec.page_size_padded is None for spec in per_layer_specs
        ), "DSA drafter layers must not carry page_size_padded."
        return max(
            _dsa_sliding_window_max_pages(spec, vllm_config)
            for spec in per_layer_specs
        )
    return kv_cache_spec.max_memory_usage_pages(vllm_config)


def _dsa_drafter_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[str], dict[str, int], list[str], int, int] | None:
    if len(kv_cache_groups) != 2:
        return None

    target_group, drafter_group = kv_cache_groups
    if not isinstance(target_group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return None
    if not isinstance(drafter_group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return None

    target_names = list(target_group.layer_names)
    drafter_names = list(drafter_group.layer_names)
    if not target_names or not drafter_names:
        return None

    target_specs = target_group.kv_cache_spec.kv_cache_specs
    drafter_specs = drafter_group.kv_cache_spec.kv_cache_specs
    try:
        target_layer_specs = [target_specs[name] for name in target_names]
        drafter_layer_specs = [drafter_specs[name] for name in drafter_names]
    except KeyError:
        return None

    if not all(type(spec) is MLAAttentionSpec for spec in target_layer_specs):
        return None
    if not all(type(spec) is SlidingWindowSpec for spec in drafter_layer_specs):
        return None

    assert all(
        spec.page_size_padded is None for spec in drafter_layer_specs
    ), "DSA drafter layers must not carry page_size_padded."

    drafter_pages = {spec.page_size_bytes for spec in drafter_layer_specs}
    if len(drafter_pages) != 1:
        return None
    drafter_page = drafter_pages.pop()

    target_page_by_name = {
        name: target_specs[name].page_size_bytes for name in target_names
    }
    per_block = sum(target_page_by_name.values()) + drafter_page * len(
        drafter_names
    )
    return (target_names, target_page_by_name, drafter_names, drafter_page, per_block)


def _get_kv_cache_groups_dsa_drafter(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec] | None:
    if not kv_cache_spec:
        return None

    target_specs: dict[str, KVCacheSpec] = {}
    drafter_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        if type(spec) is SlidingWindowSpec:
            drafter_specs[layer_name] = spec
        elif type(spec) is MLAAttentionSpec:
            target_specs[layer_name] = spec
        else:
            return None

    if not target_specs or not drafter_specs:
        return None

    if is_kv_cache_spec_uniform(target_specs):
        return None

    target_uniform_spec = UniformTypeKVCacheSpecs.from_specs(target_specs)
    if target_uniform_spec is None:
        return None

    drafter_specs = _dsa_unpadded_drafter_specs(drafter_specs)
    drafter_pages = {spec.page_size_bytes for spec in drafter_specs.values()}
    if len(drafter_pages) != 1:
        return None

    drafter_uniform_spec = UniformTypeKVCacheSpecs.from_specs(drafter_specs)
    if drafter_uniform_spec is None:
        return None

    target_groups = _get_kv_cache_groups_uniform_type(target_uniform_spec)
    if len(target_groups) != 1:
        return None

    drafter_group = KVCacheGroupSpec(
        layer_names=list(drafter_uniform_spec.kv_cache_specs.keys()),
        kv_cache_spec=drafter_uniform_spec,
    )
    groups = [target_groups[0], drafter_group]
    if _dsa_drafter_tensor_layout(groups) is None:
        return None
    return groups
