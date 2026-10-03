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

# Shared with the JIT build; loaded by path because the package is not importable yet
_spec = importlib.util.spec_from_file_location(
    "cuda_flags", os.path.join(os.path.dirname(os.path.abspath(__file__)), "exllamav3", "util", "cuda_flags.py")
)
cuda_flags = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cuda_flags)

is_hip = bool(torch and torch_version.hip)

extra_cflags = []
extra_cuda_cflags = cuda_flags.cuda_cflags(
    cuda_home = cpp_extension.CUDA_HOME,
    debug = ext_debug,
    hip = is_hip,
) if precompile and torch else []

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

if not is_hip and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}
if is_hip:
    extra_compile_args["hipcc"] = extra_cuda_cflags
    extra_compile_args["hip"] = extra_cuda_cflags
if verbose:
    print("EXL3 flags:", "hip" if is_hip else "cuda", extra_cuda_cflags)

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)
sources = [
    os.path.relpath(os.path.join(root, file), start=os.path.dirname(__file__))
    for root, _, files in os.walk(sources_dir)
    for file in files
    if file.endswith(('.c', '.cpp', '.cu'))
    # Skip hipify outputs: they are regenerated in-place by torch's BuildExtension
    # on every ROCm rebuild, and compiling them alongside their non-hipified
    # counterparts produces duplicate-symbol link errors.
    and '_hip.' not in file and not file.startswith('hip_')
    # CUDA-only sources that do not hipify (inline PTX asm, ldmatrix, cuda::atomic):
    # excluded from HIP builds; their bindings are #ifndef USE_ROCM.
    and not (is_hip and file in (
        'dflash2.cu',
        'hc_mix_tiled.cu',
        'routing_gemm.cu',
    ))
]

setup_kwargs = {}
if precompile and cpp_extension is not None:
    setup_kwargs = {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                include_dirs=[sources_dir],
                libraries=(
                    ["hipblas"] if is_hip else
                    ["cublas"] if windows else
                    []
                ),
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }

setup(
    verbose=verbose,
    **setup_kwargs,
)
