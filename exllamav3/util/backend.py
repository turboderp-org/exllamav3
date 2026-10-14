"""
Backend facts and the per-backend defaults the modules consult.

The tuning defaults and kernel selections the Python side decides differently between the CUDA and ROCm
builds live here, so the choices are reviewable in one place; the modules import these names rather than
testing torch.version.hip themselves. Environment overrides keep their existing names (doc/env_vars.md).
"""

import os
import torch

# True for a ROCm torch (HIP runtime); the C++ extension's USE_ROCM
ROCM = bool(torch.version.hip)

_gfx_arch_cache = {}


def gfx_arch(device) -> str:
    """ROCm: the device's gfx target without feature flags (e.g. "gfx1100"), for per-architecture tunings
    measured on that target; "" on CUDA"""
    if not ROCM:
        return ""
    idx = device.index if isinstance(device, torch.device) else device
    if idx is None:
        idx = torch.cuda.current_device()
    arch = _gfx_arch_cache.get(idx)
    if arch is None:
        arch = _gfx_arch_cache[idx] = torch.cuda.get_device_properties(idx).gcnArchName.split(":")[0]
    return arch


def _env(name: str, cuda, rocm):
    """Environment override, else the backend's default (returned as a string, like the override)"""
    return os.environ.get(name, str(rocm if ROCM else cuda))


# --- Tuning defaults --------------------------------------------------------------------------------------

# Q/K/V projections as one sliced GEMM bundle (mgemm): the RDNA mgemv has no sliced form
QKV_SLICE = _env("EXL3_QKV_SLICE", 1, 0) != "0"

# int8-activation GEMV for mul1 tensors at small m (exl3_gemv_int8.cu, which holds the same default): off on
# ROCm, where the RDNA fdot2 GEMVs are faster at these shapes
INT8_GEMV_MODE = int(_env("EXL3_INT8_GEMV", 2, 0))

# Fused MoE prefill kernel: the RDNA kernel (rocm/quant/exl3_moe_inner_rdna.cuh) pipelines and tiles the
# rows itself, so it takes experts up to 512 rows and neither the batched reconstruct tier nor the 32 / 64-row
# tile instances apply to it
MOE_FUSED_ROWS = int(_env("EXL3_MOE_FUSED_ROWS", 128, 512))
MOE_BATCH_RECON = _env("EXL3_MOE_BATCH_RECON", 1, 0) != "0"
MOE_MTILE = _env("EXL3_MOE_MTILE", 1, 0) != "0"

# mHC launch-count folds at decode row counts (hc_mix_fused, hc_fuse.cuh): apply_ deferred into the next
# site's mix, the following RMSNorm run inside its finalize. Bit-identical to the unfused launches; on by
# default where the per-launch cost dominates these small kernels (RDNA)
HC_FOLD = _env("EXL3_HC_FOLD", 0, 1) != "0"

# Split-decode attention (triton_paged.py, bc_attn.py): AMD's Triton backend runs the split kernel at
# 8 warps / 1 stage without spilling to scratch, where 4 / 2 does. gfx1100 is measured separately: at
# head_dim <= 128 a 32-key tile at 2 warps / 2 stages beats the 8192 / head_dim tile across context
# lengths (more splits at short contexts, far fewer stalls at long ones); the QSA sparse split kernel
# keeps the pair below
ATTN_SPLIT_WARPS_STAGES = (8, 1) if ROCM else (4, 2)


def attn_decode_config(device, hd_pad: int) -> tuple[int, int, int]:
    """(block_n, num_warps, num_stages) for the paged decode attention split kernel; block_n is the
    kv tile, the top of the shared-memory ladder on devices that cannot hold it"""
    if hd_pad <= 128 and gfx_arch(device) == "gfx1100":
        return 32, 2, 2
    return max(16, 8192 // hd_pad), *ATTN_SPLIT_WARPS_STAGES


# --- Turing (sm_75) prefill ---------------------------------------------------------------------------------
#
# Triton lowers tl.dot to scalar FMA on sm_75 (no tensor-core path for that target), so the paged prefill
# kernels run at a small fraction of the part's HMMA rate and prefill time grows linearly with context. On
# sm_75 a prefill chunk attends over a dense fp16 copy of its window instead: the quantized-cache staging
# pass (which dequantizes the referenced window anyway) or a gather of fp16 pages, then PyTorch's
# memory-efficient SDPA (cutlass FMHA, tensor cores), or the fa75 kernel (exllamav3_ext/turing) for head_dim
# 256 where the cutlass kernel is slow. Decode and draft verification stay on the Triton kernels, which read
# the packed cache directly. EXL3_SDPA_PREFILL / EXL3_FA75 force either path on or off on any CUDA device

_cc_cache = {}


def compute_capability(device) -> tuple[int, int]:
    """(major, minor) of a CUDA device; (0, 0) on ROCm, where the HIP major is a gfx generation"""
    if ROCM:
        return (0, 0)
    idx = device.index if isinstance(device, torch.device) else device
    if idx is None:
        idx = torch.cuda.current_device()
    cc = _cc_cache.get(idx)
    if cc is None:
        cc = _cc_cache[idx] = torch.cuda.get_device_capability(idx)
    return cc


def _sm75_switch(name: str, device) -> bool:
    env = os.environ.get(name)
    if env is not None:
        return env.strip().lower() not in ("0", "", "false", "off", "no")
    return compute_capability(device) == (7, 5)


def sdpa_prefill(device) -> bool:
    """Prefill chunks attend over a dense fp16 window with SDPA / fa75 (sm_75 default; EXL3_SDPA_PREFILL)"""
    return _sm75_switch("EXL3_SDPA_PREFILL", device)


def fa75_prefill(device) -> bool:
    """The fa75 kernel for head_dim 256 on that path (sm_75 default; EXL3_FA75)"""
    return _sm75_switch("EXL3_FA75", device)


def qc_prefill_two_pass_min_q(device) -> int:
    """Query length from which quantized-cache prefill stages the window as fp16 (triton_paged.py):
    EXL3_QC_PF_TWO_PASS_MIN_Q, else 9 where the window path takes every prefill chunk, else 256"""
    env = os.environ.get("EXL3_QC_PF_TWO_PASS_MIN_Q")
    if env is not None:
        return int(env)
    return 9 if sdpa_prefill(device) else 256


# DSA decode (bc_dsa.py, bc_mla.py, dsa_triton.py): the MQA split kernel (dsa_mqa.py) where the shape allows
# it on ROCm, fewer key splits per row, and upstream's split kernel at 8 warps (spills far less to scratch on
# these parts) where it does not
DSA_MQA = ROCM
DSA_N_SPLITS = 8 if ROCM else 16
DSA_SPLIT_WARPS = 8 if ROCM else 4
