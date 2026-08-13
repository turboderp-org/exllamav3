from __future__ import annotations
import importlib.machinery
import importlib.util
import torch
from torch.utils.cpp_extension import load
import os
import sys
from .util.arch_list import maybe_set_arch_list_env

extension_name = "exllamav3_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

def is_precompiled_extension_available():
    spec = importlib.util.find_spec(extension_name)
    if not spec or not spec.origin or not spec.loader:
        return False
    return any(
        spec.origin.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

if is_precompiled_extension_available():
    import exllamav3_ext
else:

    # Kludge to get compilation working on Windows

    if windows:

        def find_msvc():

            # Possible locations for MSVC, in order of preference

            program_files_x64 = os.environ["ProgramW6432"]
            program_files_x86 = os.environ["ProgramFiles(x86)"]

            msvc_dirs = \
            [
                a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
                for b in ["2022", "2019", "2017"]
                for a in [program_files_x64, program_files_x86]
                for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
            ]

            for msvc_dir in msvc_dirs:
                if not os.path.exists(msvc_dir): continue

                # Prefer the latest version

                versions = sorted(os.listdir(msvc_dir), reverse = True)
                for version in versions:

                    compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                    if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                        return compiler_dir

            # No path found

            return None

        import subprocess

        # Check if cl.exe is already in the path

        try:

            subprocess.check_output(["where", "/Q", "cl"])

        # If not, try to find an installation of Visual Studio and append the compiler dir to the path

        except subprocess.CalledProcessError as e:

            cl_path = find_msvc()
            if cl_path:
                if verbose:
                    print(" -- Injected compiler path:", cl_path)
                os.environ["path"] += ";" + cl_path
            else:
                print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)

    # compiler flags

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)

    extra_cflags = []
    extra_cuda_cflags = []

    if torch.version.hip:
        extra_cuda_cflags += ["-Ofast", "-DUSE_ROCM", "-Wno-register"]
        extra_cflags += ["-DUSE_ROCM"]
    else:
        extra_cuda_cflags += [
            "-lineinfo", "-O3", "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
        ]

    if windows:
        # TODO: preprocessor and lean_and_mean flags are needed for Windows cu132 build, verify that they don't break
        #       older cu128 builds
        # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
        # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
        # Defined globally so it holds regardless of include order in any TU (mirrors setup.py).
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

    if not windows and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
        extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

    if torch.version.hip:
        extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

    if verbose:
        if torch.version.hip:
            extra_cuda_cflags += ["-verbose"]
        else:
            extra_cuda_cflags += ["--ptxas-options=-v"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]

    # sources

    from .exllamav3_ext.build_config import get_sources as _get_sources
    is_rocm = bool(torch.version.hip)
    sources = _get_sources(sources_dir, is_rocm)

    # Load extension

    maybe_set_arch_list_env()
    exllamav3_ext = load(
        name = extension_name,
        sources = sources,
        extra_include_paths = [sources_dir],
        verbose = verbose,
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = extra_cuda_cflags,
        extra_cflags = extra_cflags
    )


# When a BC_* class is not compiled into the extension (e.g. on ROCm where the
# libtorch/ sources are excluded), make attribute access return a callable that
# yields None instead of raising AttributeError. This lets call sites write
# ``self.bc = ext.BC_Mamba2(...)`` unconditionally — they get None on platforms
# that lack the class, and the real object on platforms that have it.

if torch.version.hip:
    from .ext_fallbacks import _BCNone

    _bc_none = _BCNone()

    # BC_* constructors: return None when the class isn't compiled
    for _name in [
        'BC_Mamba2', 'BC_GatedDeltaNet', 'BC_GatedDeltaNetSplit',
        'BC_MLP', 'BC_GatedMLP', 'BC_BlockSparseMLP',
        'BC_Attention', 'BC_GatedRMSNorm',
        'BC_LinearEXL3', 'BC_LinearFP16',
        'BC_DSV4Compressor', 'BC_DSV4Attention', 'BC_DSV4BatchAttention',
        'BC_MLAttention', 'BC_SAM',
    ]:
        if not hasattr(exllamav3_ext, _name):
            setattr(exllamav3_ext, _name, _bc_none)

    # C++ functions from excluded source files: replace with PyTorch implementations
    from . import ext_fallbacks as _fb

    for _name in [
        'silu_mul', 'silu_oai_mul', 'gelu_mul', 'relu2_mul', 'relu_mul', 'xielu',
        'mul_sigmoid_', 'mul_sigmoid_broadcast_', 'mul_softplus_broadcast_',
        'add_sigmoid_gate', 'add_sigmoid_gate_proj', 'deinterleave_qg',
        'rms_norm', 'rms_norm_res_in', 'gated_rms_norm',
        'softcap',
    ]:
        if not hasattr(exllamav3_ext, _name):
            setattr(exllamav3_ext, _name, getattr(_fb, _name))

    # Constants and functions guarded by fused_sampler_enable in generator/sampler/custom.py.
    # Disable the fused sampler path on ROCm by setting the flag and providing stub values.
    if not hasattr(exllamav3_ext, 'FUSED_SAMPLER_MAX_BLOCKS'):
        setattr(exllamav3_ext, 'FUSED_SAMPLER_MAX_BLOCKS', 0)
    if not hasattr(exllamav3_ext, 'FUSED_SAMPLER_HIST_STRIDE'):
        setattr(exllamav3_ext, 'FUSED_SAMPLER_HIST_STRIDE', 0)
    os.environ.setdefault('EXL3_FUSED_SAMPLER', '0')
