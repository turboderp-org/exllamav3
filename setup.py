from setuptools import setup
import importlib.util
import os

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

extra_cflags = ["/Ox"] if windows else ["-O3"]

if ext_debug:
    extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

extra_cuda_cflags = ["-lineinfo", "-O3"]

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

print (sources)

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

version_py = {}
with open("exllamav3/version.py", encoding="utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

setup(
    name="exllamav3",
    version=version,
    packages=[
        "exllamav3",
        "exllamav3.generator",
        "exllamav3.generator.sampler",
        "exllamav3.conversion",
        "exllamav3.models",
        "exllamav3.modules",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
        "exllamav3.tokenizer",
        "exllamav3.cache",
        "exllamav3.loader",
        "exllamav3.util",
    ],
    url="https://github.com/turboderp/exllamav3",
    license="MIT",
    author="turboderp",
    install_requires=[
        "torch>=2.6.0",
        "flash_attn>=2.7.4.post1",
        "tokenizers>=0.21.1",
        "numpy>=2.1.0",
        "rich",
        "typing_extensions",
        "ninja",
        "safetensors>=0.3.2"
    ],
    include_package_data=True,
    package_data = {
        "": ["py.typed"],
    },
    verbose=verbose,
    **setup_kwargs,
)
