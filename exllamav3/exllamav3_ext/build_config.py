"""Shared build configuration for the exllamav3 C++ extension.

Used by both setup.py (precompiled) and ext.py (JIT) to avoid duplicating
the ROCm source exclusion logic.
"""
import os

ROCM_EXCLUDE_DIRS = {'parallel', 'comp_units'}

ROCM_EXCLUDE_FILES = {
    'norm.cu', 'activation.cu', 'attention.cu', 'routing.cu',
    'softcap.cu', 'histogram.cu', 'sam.cpp',
    'cache/q_cache.cu',
    'quant/exl3_gemm.cu', 'quant/exl3_gemv.cu', 'quant/exl3_gemv_int8.cu',
    'quant/exl3_moe.cu', 'quant/exl3_kernel_map.cu',
    'quant/coop_autotune.cu', 'quant/quantize.cu', 'quant/util.cu',
    'generator/sampling_fused.cu',
    'libtorch/gated_delta_net.cpp', 'libtorch/blocksparse_mlp.cpp',
    'libtorch/mlp.cpp', 'libtorch/attention.cpp', 'libtorch/gated_rmsnorm.cpp',
    'libtorch/linear.cpp', 'libtorch/dsv4_attn.cpp', 'libtorch/dsv4_compressor.cpp',
    'libtorch/mla_attention.cpp',
    'dsv4_compress.cu', 'dsa_topk.cu', 'hc_mix.cu',
    'cpu/moe_handoff.cu', 'cpu/moe_mul1.cpp',
}


def get_sources(sources_dir, is_rocm, base_dir=None):
    """Walk the extension source directory and return the list of source files.

    Auto-generated hipify intermediates (_hip*, *.hip) are always skipped.

    Args:
        sources_dir: Absolute path to the extension source directory.
        is_rocm: Whether we're building for ROCm.
        base_dir: Base directory for computing relative paths. If None, uses
                  absolute paths (for ext.py JIT). If provided, uses relative
                  paths (for setup.py precompiled).
    """
    sources = []
    for root, _, files in os.walk(sources_dir):
        for file in files:
            if not file.endswith(('.c', '.cpp', '.cu')):
                continue
            if '_hip' in file or file.endswith('.hip'):
                continue
            rel_path = os.path.relpath(os.path.join(root, file), start=sources_dir)
            norm_rel = rel_path.replace('\\', '/')
            if is_rocm:
                parts = norm_rel.split('/')
                if any(d in parts for d in ROCM_EXCLUDE_DIRS):
                    continue
                if norm_rel in ROCM_EXCLUDE_FILES:
                    continue
            full = os.path.join(root, file)
            if base_dir is not None:
                sources.append(os.path.relpath(full, start=base_dir))
            else:
                sources.append(os.path.abspath(full))
    return sources
