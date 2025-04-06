from __future__ import annotations
import torch
from torch.utils.cpp_extension import load
import os, glob
import sys
import platform
import threading
from .util.arch_list import maybe_set_arch_list_env

extension_name = "exllamav3_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

build_jit = False
try:
    import exllamav3_ext
except ModuleNotFoundError:
    build_jit = True

if build_jit:

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

    # gcc / cl.exe flags

    if windows:
        extra_cflags = ["/Ox"]
    else:
        extra_cflags = ["-Ofast"]

    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

    # nvcc flags

    extra_cuda_cflags = ["-lineinfo", "-O3"]

    if torch.version.hip:
        extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

    if verbose:
        extra_cuda_cflags += ["--ptxas-options=-v"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]

    # sources

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)
    sources = [
        os.path.abspath(os.path.join(root, file))
        for root, _, files in os.walk(sources_dir)
        for file in files
        if file.endswith(('.c', '.cpp', '.cu'))
    ]

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
