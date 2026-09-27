"""
nvcc flags shared by the JIT extension build (ext.py) and the wheel build (setup.py). setup.py
loads this file by path before the package is importable, so it uses the standard library only.
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


def cuda_cflags(cuda_home: str | None = None, debug: bool = False, hip: bool = False) -> list[str]:
    """
    Flags for the extension's CUDA translation units.

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
    flags = []
    if debug or os.environ.get("EXLLAMA_EXT_LINEINFO"):
        flags += ["-lineinfo"]
    flags += [
        "-O3", "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]
    compress = os.environ.get("EXLLAMA_EXT_COMPRESS", "1")
    if not hip and compress != "0":
        nvcc = find_nvcc(cuda_home)
        if nvcc_has_compress_mode(nvcc):
            flags += ["--compress-mode=size"]
        elif compress == "require":
            raise RuntimeError(
                f"EXLLAMA_EXT_COMPRESS=require, but nvcc ({' '.join(nvcc) if nvcc else 'not found'}) does not support "
                f"--compress-mode (CUDA 12.8 or later is needed)"
            )
    return flags
