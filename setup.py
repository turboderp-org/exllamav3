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

if torch and torch_version.hip:
    # no --use_fast_math (reassociates FP reductions)
    extra_cuda_cflags = ["-O3"]
    # ROCm 7.14's clang: deprecated `register` is a hard error
    extra_cflags += ["-Wno-register"]
    extra_cuda_cflags += ["-Wno-register"]

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
    extra_cflags += ["-O3" if (torch and torch_version.hip) else "-Ofast"]
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

# load build_config by path: importing the package would JIT the extension
_spec = importlib.util.spec_from_file_location(
    "exllamav3_build_config", os.path.join(sources_dir, "build_config.py"))
_build_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_config)
is_rocm = bool(torch and torch_version.hip)
if is_rocm and not os.environ.get("PYTORCH_ROCM_ARCH"):
    # arch_list.py can't be imported here (it would JIT the extension)
    from torch import cuda as _cuda
    _arch_list = []
    for _i in range(_cuda.device_count()):
        _name = getattr(_cuda.get_device_properties(_i), "gcnArchName", None)
        if _name:
            _name = _name.split(":", 1)[0]
            if _name.startswith("gfx") and _name not in _arch_list:
                _arch_list.append(_name)
    if _arch_list:
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(_arch_list)
sources = _build_config.get_sources(sources_dir, is_rocm, base_dir=os.path.dirname(__file__))


# ensure repo-relative includes in bindings.cpp resolve under hipcc
_ext_include_dirs = [sources_dir] if is_rocm else []

setup_kwargs = (
    {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                include_dirs=_ext_include_dirs,
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
