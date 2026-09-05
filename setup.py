import importlib.util
import os

from setuptools import setup

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print("Cannot precompile unless torch is installed.")
    print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")

windows = os.name == "nt"

extra_cflags = []
extra_cuda_cflags = [
    "-lineinfo", "-O3", "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]

if windows:
    # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
    # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
    # Defined globally so it holds regardless of include order in any TU.
    # No -std flags here: torch's cpp_extension appends its own (unconditionally on the Windows
    # nvcc path), and a second -std argument is a fatal nvcc error, not an override.
    extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
    if ext_debug:
        extra_cflags += ["/Zi"]
        extra_cuda_cflags += []
else:
    extra_cflags += ["-Ofast"]
    extra_cuda_cflags += []
    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
        extra_cuda_cflags += []

if cuda_host_cxx := os.environ.get("CUDAHOSTCXX"):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

if torch and torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)
sources = [
    os.path.relpath(os.path.join(root, file), start=os.path.dirname(__file__))
    for root, _, files in os.walk(sources_dir)
    for file in files
    if file.endswith(('.c', '.cpp', '.cu'))
]

setup_kwargs = (
    {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }
    if precompile and torch
    else {}
)

setup(
    verbose=verbose,
    **setup_kwargs,
)
