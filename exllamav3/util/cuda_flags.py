"""
Compiler flags and source selection shared by the JIT extension build (ext.py) and the wheel build
(setup.py). setup.py loads this file by path before the package is importable, so it uses the standard
library only.
"""

from __future__ import annotations
import fnmatch
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys


def find_nvcc(cuda_home: str | None = None) -> list[str] | None:
    """The nvcc command torch's cpp_extension will run, as an argument list."""
    if override := os.environ.get("PYTORCH_NVCC"):
        return shlex.split(override, posix = os.name != "nt")
    exe = "nvcc.exe" if os.name == "nt" else "nvcc"
    for home in (cuda_home, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if home:
            path = os.path.join(home, "bin", exe)
            if os.path.isfile(path):
                return [path]
    path = shutil.which("nvcc")
    return [path] if path else None


def nvcc_has_compress_mode(nvcc: list[str] | None) -> bool:
    if not nvcc:
        return False
    try:
        out = subprocess.run(nvcc + ["--help"], capture_output = True, text = True, timeout = 60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "--compress-mode" in out


# Sources left out of ROCm builds: their kernels are written in inline PTX (tensor-core MMA, cp.async,
# ldmatrix) with no portable form yet. Their bindings are compiled out as well and the extension reports
# them missing through its HAS_* attributes (bindings.cpp), which the Python side checks before using them
HIP_EXCLUDED_SOURCES = {
    "hgemm_f16acc.cu",      # fp16-accumulator hgemm (hgemm_f16acc): GeForce tensor-core PTX; RDNA uses rocm/wmma_gemm.cu
    # Replaced on ROCm by an implementation of the same interface under rocm/
    "exl3_gemv.cu",         # -> rocm/quant/exl3_gemv_rdna.cu (exl3_gemv.cuh)
    # The GEMM kernel map, MoE host, comp units and MoE instance units are shared: the kernel headers select
    # the backend's inner (quant/exl3_gemm_kernel.cuh, quant/comp_units/exl3_moe_inst_common.cuh) and the
    # shape table / dispatch carry USE_ROCM arms. The M-tiled MoE instances exist for the CUDA kernel only
    # (EXL3_MOE_MTILE)
    "quant/comp_units/exl3_moe_inst_*_m32.cu",
    "quant/comp_units/exl3_moe_inst_*_m64.cu",
}


def is_hipify_output(filename: str) -> bool:
    """
    Files torch's hipify writes next to the sources on ROCm builds (foo.cu -> foo.hip, foo.cpp ->
    foo_hip.cpp, foo.cuh -> foo_hip.cuh, cuda_drv.h -> hip_drv.h). They are build products, and compiling
    them alongside the originals they were translated from fails, so they are never taken as sources.
    """
    return "_hip." in filename or filename.endswith(".hip") or filename.startswith("hip_")


def _hip_excluded(rel_path: str) -> bool:
    """HIP_EXCLUDED_SOURCES entries without a directory match a file name anywhere; entries with one
    match the path relative to the sources directory (fnmatch patterns)"""
    name = os.path.basename(rel_path)
    return any(
        fnmatch.fnmatch(rel_path if "/" in pattern else name, pattern)
        for pattern in HIP_EXCLUDED_SOURCES
    )


def extension_sources(sources_dir: str, hip: bool = False) -> list[str]:
    """Absolute paths of the extension's translation units for a CUDA or ROCm build. Sources under
    rocm/ (the RDNA kernels and the ROCm compat layer) belong to ROCm builds only."""
    base = os.path.abspath(sources_dir)
    sources = []
    for root, _, files in os.walk(base):
        for file in files:
            path = os.path.join(root, file)
            rel = os.path.relpath(path, base).replace(os.sep, "/")
            if not file.endswith((".c", ".cpp", ".cu")) or is_hipify_output(file):
                continue
            if hip and _hip_excluded(rel):
                continue
            if not hip and rel.startswith("rocm/"):
                continue
            sources.append(path)
    return sorted(sources)


def hip_cflags(debug: bool = False) -> list[str]:
    """
    Flags for the extension's HIP translation units (compiled by hipcc, which is clang-based and takes
    none of nvcc's options). No fast-math: clang's -ffast-math also assumes finite values, which would let
    the compiler drop the infinity and NaN handling the sampling and masking kernels rely on. -Wno-register:
    C++17 removed the register storage class and clang rejects it by default.

    -fgpu-flush-denormals-to-zero matches the fp32 flush-to-zero that --use_fast_math gives the CUDA build,
    which the deterministic kernels need to agree with it bit for bit (det_gemm.cuh).

    The device code objects are stored compressed (--offload-compress, the counterpart of nvcc's
    --compress-mode) unless EXLLAMA_EXT_COMPRESS=0: a wheel carrying every RDNA family would otherwise be
    several times the size of a single-target build.
    """
    flags = ["-O3", "-Wno-register", "-DHIPBLAS_USE_HIP_HALF", "-fgpu-flush-denormals-to-zero"]
    if os.environ.get("EXLLAMA_EXT_COMPRESS", "1") != "0":
        flags += ["--offload-compress"]
    if debug:
        flags += ["-g"]
    return flags


def use_rocm_sdk_devel(cpp_extension) -> None:
    """
    pip-installed ROCm SDK (TheRock wheels): with ROCM_HOME/ROCM_PATH unset, torch takes _rocm_sdk_core as the
    ROCm root, which holds the HIP runtime but not the library headers the build needs (hipBLAS, and thrust
    through torch's own headers). The SDK's development package is a complete ROCm tree; point torch at it
    instead. torch reads its module-level ROCM_HOME when it assembles the compile and link commands, so this
    takes effect for any build started afterwards. An explicit ROCM_HOME/ROCM_PATH is left alone.
    """
    if os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH"):
        return
    has_libs = lambda root: os.path.exists(os.path.join(root, "include", "hipblas", "hipblas.h"))
    if cpp_extension.ROCM_HOME and has_libs(cpp_extension.ROCM_HOME):
        return
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec is not None and spec.origin is not None:
        devel = os.path.dirname(os.path.realpath(spec.origin))
        if has_libs(devel):
            cpp_extension.ROCM_HOME = devel
            return
    # Current SDK wheels ship their development tree as an archive. The SDK CLI expands it
    # and returns its root; the expanded package name can vary by SDK version and platform.
    # The expansion is a one-time write of the whole tree into site-packages, hence the notice;
    # a failing CLI (unwritable site-packages, a broken SDK install) leaves torch's root in
    # place, and the build then reports the missing headers itself
    if importlib.util.find_spec("rocm_sdk_devel") is not None:
        print(" -- Locating the ROCm SDK development files (expanding them on first use)", flush = True)
        try:
            devel = subprocess.check_output(
                [sys.executable, "-m", "rocm_sdk", "path", "--root"], text = True
            ).strip()
        except (subprocess.CalledProcessError, OSError) as e:
            print(f" !! ROCm SDK development files not found ({e}); run `rocm-sdk init` or set ROCM_HOME", flush = True)
            return
        if has_libs(devel):
            cpp_extension.ROCM_HOME = devel


def hip_compiler_wrapper() -> str | None:
    """
    The compiler wrapper torch itself puts in front of nvcc and the host compiler on CUDA builds (ccache or
    sccache, whichever is on PATH; TORCH_NO_COMPILER_WRAPPER disables it). torch skips it on ROCm over an old
    hipcc incompatibility that ccache no longer has, so the ROCm build applies the same rule here.
    """
    if os.environ.get("TORCH_NO_COMPILER_WRAPPER"):
        return None
    for wrapper in ("ccache", "sccache"):
        if shutil.which(wrapper):
            return wrapper
    return None


def rewrite_hip_ninja_file(content: str, wrapper: str | None) -> str:
    """
    Adjust the build.ninja torch writes for a ROCm build.

    torch's HIP compile rule has no dependency file (its comment says -MD is unsupported by ROCm, which
    hipcc, being clang, does support), so a header edit rebuilt nothing until an including .cu changed or
    the objects were deleted by hand. Add the -MD -MF / depfile lines the CUDA rule gets; the hipified
    header copies (*_hip.cuh) the depfile lists are rewritten only when their source changes, so their
    mtimes are meaningful. With a wrapper, prefix the host and device compilers with it (see
    hip_compiler_wrapper).
    """
    if "rule cuda_compile" in content:
        head, _, tail = content.partition("rule cuda_compile\n")
        rule, _, rest = tail.partition("\n\n")
        if "depfile" not in rule:
            rule = rule.replace("$nvcc  $cuda_cflags", "$nvcc -MD -MF $out.d $cuda_cflags")
            rule = "  depfile = $out.d\n  deps = gcc\n" + rule
        content = head + "rule cuda_compile\n" + rule + "\n\n" + rest
    if wrapper:
        content = re.sub(r"^(cxx|nvcc) = (?!" + re.escape(wrapper) + r" )", r"\1 = " + wrapper + " ", content, flags = re.M)
    return content


def patch_hip_ninja_file_writer(cpp_extension) -> None:
    """
    Route torch's build.ninja through rewrite_hip_ninja_file. torch writes the file with its own
    _maybe_write, which leaves an unchanged file untouched (so no spurious rebuilds); wrapping that keeps
    the behavior, since the rewritten content is the same every time.
    """
    if getattr(cpp_extension, "_exllamav3_ninja_patched", False):
        return
    orig = cpp_extension._maybe_write
    wrapper = hip_compiler_wrapper()

    def _maybe_write(filename, new_content):
        if os.path.basename(filename) == "build.ninja":
            new_content = rewrite_hip_ninja_file(new_content, wrapper)
        return orig(filename, new_content)

    cpp_extension._maybe_write = _maybe_write
    cpp_extension._exllamav3_ninja_patched = True


def hip_include_flags(sources_dir: str) -> list[str]:
    """
    Flags for every translation unit of a ROCm build, host C++ and HIP alike: force-include the compat
    layer (rocm/compat.h) and put the stand-ins for CUDA-only headers (rocm/include) on the path.
    """
    rocm_dir = os.path.abspath(os.path.join(sources_dir, "rocm"))
    return ["-include", os.path.join(rocm_dir, "compat.h"), "-I" + os.path.join(rocm_dir, "include")]


def cuda_cflags(cuda_home: str | None = None, debug: bool = False, hip: bool = False) -> list[str]:
    """
    Flags for the extension's CUDA translation units (or HIP ones, see hip_cflags).

    Source line tables (-lineinfo) only serve profilers and debuggers and make up most of the
    embedded kernel images, so they are left out unless the build is a debug build or
    EXLLAMA_EXT_LINEINFO is set.

    The kernel images are stored compressed where the compiler offers it (--compress-mode).
    This packs the finished images and does not change the generated code; it keeps the
    extension well below the image size the Windows loader accepts as architectures are added.
    EXLLAMA_EXT_COMPRESS=0 turns it off, for drivers too old to load compressed images, and
    EXLLAMA_EXT_COMPRESS=require fails the build when the compiler lacks the option, so that a
    release build cannot fall back to uncompressed images unnoticed.
    """
    if hip:
        return hip_cflags(debug)
    flags = []
    if debug or os.environ.get("EXLLAMA_EXT_LINEINFO"):
        flags += ["-lineinfo"]
    flags += [
        "-O3", "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]
    compress = os.environ.get("EXLLAMA_EXT_COMPRESS", "1")
    if compress != "0":
        nvcc = find_nvcc(cuda_home)
        if nvcc_has_compress_mode(nvcc):
            flags += ["--compress-mode=size"]
        elif compress == "require":
            raise RuntimeError(
                f"EXLLAMA_EXT_COMPRESS=require, but nvcc ({' '.join(nvcc) if nvcc else 'not found'}) does not support "
                f"--compress-mode (CUDA 12.8 or later is needed)"
            )
    return flags
