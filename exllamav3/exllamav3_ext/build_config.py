"""Source list for the C++ extension, shared by setup.py and the JIT loader."""

import os

ROCM_EXCLUDE_DIRS = set()

ROCM_EXCLUDE_FILES = {
    'quant/exl3_gemv.cu', 'quant/exl3_gemv_int8.cu',
}
CUDA_EXCLUDE_FILES = {
    'exl3_rocm_stubs.cpp',
}


def maybe_set_rocm_home():
    """Set ROCM_HOME for torch's cpp_extension when the ROCm SDK is installed as
    Python wheels (AMD's TheRock index) and no other ROCm is discoverable.

    cpp_extension resolves the SDK location once at import time (env var, then
    hipcc on PATH, then /opt/rocm), and a wheel-provided SDK is in neither
    place, so this must run before cpp_extension is imported. The wheel-provided
    root also needs its first-use expansion, which get_devel_root() performs.
    Any failure leaves the environment untouched and defers to torch's own
    discovery chain."""

    if os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH"):
        return
    try:
        from rocm_sdk._devel import get_devel_root
        root = get_devel_root()
        if not (root / "bin" / "hipcc").exists():
            return
    except Exception:
        return
    os.environ["ROCM_HOME"] = str(root)


def get_sources(sources_dir, is_rocm, base_dir = None):
    """Walk the extension source directory and return the source file list. Stale
    hipify intermediates (*.hip, *_hip.*) are skipped. With base_dir the paths are
    relative to it (setup.py), otherwise absolute (JIT loader)."""

    # CUDA-only instantiations: fused-MoE scheduler and int8-GEMV comp units
    rocm_exclude_prefixes = ('exl3_gemv_int8_inst_',)

    sources = []
    for root, _, files in os.walk(sources_dir):
        for file in files:
            if not file.endswith(('.c', '.cpp', '.cu')): continue
            if file.endswith('.hip') or '_hip' in file: continue
            if is_rocm and file.startswith(rocm_exclude_prefixes): continue
            rel_path = os.path.relpath(os.path.join(root, file), start = sources_dir)
            norm_rel = rel_path.replace('\\', '/')
            if is_rocm:
                parts = norm_rel.split('/')
                if any(d in parts for d in ROCM_EXCLUDE_DIRS): continue
                if norm_rel in ROCM_EXCLUDE_FILES: continue
            elif norm_rel in CUDA_EXCLUDE_FILES:
                continue
            full = os.path.join(root, file)
            if base_dir is not None:
                sources.append(os.path.relpath(full, start = base_dir))
            else:
                sources.append(os.path.abspath(full))
    return sources
