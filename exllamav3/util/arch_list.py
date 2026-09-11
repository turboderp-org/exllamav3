import os
import torch

# Since Torch 2.3.0 an annoying warning is printed every time the C++ extension is loaded, unless the
# TORCH_CUDA_ARCH_LIST variable is set. The default behavior from pytorch/torch/utils/cpp_extension.py
# is copied in the function below, but without the warning.


def _rocm_device_arch(index: int) -> str:
    props = torch.cuda.get_device_properties(index)
    gcn_arch = getattr(props, "gcnArchName", "")
    gcn_arch = gcn_arch.split(":", 1)[0]
    if gcn_arch.startswith("gfx") and gcn_arch[3:].isdigit():
        return gcn_arch

    name = torch.cuda.get_device_name(index)
    if "9060" in name:
        return "gfx1200"
    if "9070" in name or "R9700" in name:
        return "gfx1201"
    if "7900 XTX" in name or "7900 XT" in name:
        return "gfx1100"
    if "7800 XT" in name:
        return "gfx1101"
    if "7700 XT" in name or "7800" in name:
        return "gfx1102"
    raise RuntimeError(
        f"Could not determine ROCm arch for device '{name}'. "
        f"Set PYTORCH_ROCM_ARCH env var manually "
        f"(e.g. export PYTORCH_ROCM_ARCH=gfx1201)."
    )


def maybe_set_arch_list_env():

    if os.environ.get('TORCH_CUDA_ARCH_LIST', None):
        return

    if torch.version.hip:
        if not os.environ.get('PYTORCH_ROCM_ARCH'):
            arch_list = []
            try:
                import subprocess
                result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith('Name:'):
                        name = line.split('Name:')[1].strip()
                        # Match only proper gfxXXX identifiers (e.g. gfx1100),
                        # not ISA target triples like amdgcn-amd-amdhsa--gfx11-generic
                        if name.startswith('gfx') and name[3:].isdigit():
                            if name not in arch_list:
                                arch_list.append(name)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: prefer the architecture reported by the runtime, then use known
            # product names when older Torch builds omit gcnArchName.
            if not arch_list:
                for i in range(torch.cuda.device_count()):
                    arch = _rocm_device_arch(i)
                    if arch not in arch_list:
                        arch_list.append(arch)

            if arch_list:
                os.environ["PYTORCH_ROCM_ARCH"] = ";".join(arch_list)
        return

    if not torch.version.cuda:
        return

    arch_list = []
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        # Strip known NVIDIA suffixes: 'a' (accelerated) or 'f' (family)
        supported_sm = [int(arch.split('_')[1].rstrip('af'))
                        for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
        if not supported_sm:
            continue
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
        # Capability of the device may be higher than what's supported by the user's
        # NVCC, causing compilation error. User's NVCC is expected to match the one
        # used to build pytorch, so we use the maximum supported capability of pytorch
        # to clamp the capability.
        capability = min(max_supported_sm, capability)
        arch = f'{capability[0]}.{capability[1]}'
        if arch not in arch_list:
            arch_list.append(arch)
    if not arch_list:
        return
    arch_list = sorted(arch_list)
    arch_list[-1] += '+PTX'

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

maybe_set_arch_list_env()