import os
import torch

# Since Torch 2.3.0 an annoying warning is printed every time the C++ extension is loaded, unless the
# TORCH_CUDA_ARCH_LIST variable is set. The default behavior from pytorch/torch/utils/cpp_extension.py
# is copied in the function below, but without the warning.

def maybe_set_arch_list_env():

    if torch.version.hip:
        # The kernels assume 32-lane warps throughout (warp masks, ballot widths, MMA
        # fragment layouts). RDNA parts run wave32; CDNA runs wave64 and is not
        # supported. Check every visible device at import so unsupported hardware
        # fails before any kernel launch, not just the quant paths that query DevCtx.
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            if props.warp_size != 32:
                raise RuntimeError(
                    f"exllamav3: ROCm device {props.name} has warp size "
                    f"{props.warp_size}; only wave32 (RDNA) targets are supported"
                )
        # ROCm: cpp_extension reads PYTORCH_ROCM_ARCH (not TORCH_CUDA_ARCH_LIST) and
        # expects gfx names (e.g. gfx1100). get_device_capability returns (major, minor)
        # which does not map to a gfx name, so use the device gcnArchName directly.
        if os.environ.get('PYTORCH_ROCM_ARCH', None):
            return
        arch_list = []
        for i in range(torch.cuda.device_count()):
            arch = torch.cuda.get_device_properties(i).gcnArchName.split(':')[0]
            if arch not in arch_list:
                arch_list.append(arch)
        if arch_list:
            os.environ["PYTORCH_ROCM_ARCH"] = ";".join(sorted(arch_list))
        return

    if os.environ.get('TORCH_CUDA_ARCH_LIST', None):
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