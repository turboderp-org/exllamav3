"""
Compiler flags and source selection shared by the JIT extension build (ext.py) and the wheel build
(setup.py). setup.py loads this file by path before the package is importable, so it uses the standard
library only.
"""

from __future__ import annotations
import os
import shlex
import shutil
import subprocess


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
    "dflash2.cu",           # DFlash2 drafter kernels
    "hc_mix_tiled.cu",      # tiled int8 GatedResidual mix (gr_mix_tiled)
    "routing_gemm.cu",      # deterministic int8 router GEMM (routing_gemm_det, det_quant_weight)
    "hgemm_f16acc.cu",      # fp16-accumulator hgemm (hgemm_f16acc)
}


def is_hipify_output(filename: str) -> bool:
    """
    Files torch's hipify writes next to the sources on ROCm builds (foo.cu -> foo.hip, foo.cpp ->
    foo_hip.cpp, foo.cuh -> foo_hip.cuh, cuda_drv.h -> hip_drv.h). They are build products, and compiling
    them alongside the originals they were translated from fails, so they are never taken as sources.
    """
    return "_hip." in filename or filename.endswith(".hip") or filename.startswith("hip_")


def extension_sources(sources_dir: str, hip: bool = False) -> list[str]:
    """Absolute paths of the extension's translation units for a CUDA or ROCm build."""
    return sorted(
        os.path.abspath(os.path.join(root, file))
        for root, _, files in os.walk(sources_dir)
        for file in files
        if file.endswith((".c", ".cpp", ".cu"))
        and not is_hipify_output(file)
        and not (hip and file in HIP_EXCLUDED_SOURCES)
    )


def hip_cflags(debug: bool = False) -> list[str]:
    """
    Flags for the extension's HIP translation units (compiled by hipcc, which is clang-based and takes
    none of nvcc's options). No fast-math: clang's -ffast-math also assumes finite values, which would let
    the compiler drop the infinity and NaN handling the sampling and masking kernels rely on. -Wno-register:
    C++17 removed the register storage class and clang rejects it by default.
    """
    flags = ["-O3", "-Wno-register", "-DHIPBLAS_USE_HIP_HALF"]
    if debug:
        flags += ["-g"]
    return flags


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
