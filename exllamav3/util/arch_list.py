import os
import torch

# Since Torch 2.3.0 an annoying warning is printed every time the C++ extension is loaded, unless the
# TORCH_CUDA_ARCH_LIST variable is set. The default behavior from pytorch/torch/utils/cpp_extension.py
# is copied in the function below, but without the warning.

def maybe_set_arch_list_env():

    if torch.version.hip:
        maybe_set_rocm_arch_env()
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

def maybe_set_rocm_arch_env():
    """
    ROCm counterpart of the above: build for the visible devices' architectures unless PYTORCH_ROCM_ARCH
    names them. gcnArchName carries target features after the name (e.g. "gfx90a:sramecc+:xnack-"), which
    the offload-arch list does not take.

    The kernels assume 32-lane warps throughout, which only RDNA parts (gfx10 and later consumer/workstation
    GPUs) run; CDNA accelerators execute in wave64, so they are rejected here rather than failing obscurely
    inside the kernels.
    """
    archs = set()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        warp = getattr(props, "warp_size", 32)
        if warp != 32:
            raise RuntimeError(
                f"Device {i} ({props.name}, {props.gcnArchName}) executes {warp}-wide wavefronts. "
                f"exllamav3 on ROCm supports wave32 (RDNA) GPUs only."
            )
        archs.add(props.gcnArchName.split(":")[0])
    if archs and not os.environ.get("PYTORCH_ROCM_ARCH"):
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(sorted(archs))

maybe_set_arch_list_env()