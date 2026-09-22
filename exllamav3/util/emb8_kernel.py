"""int8 embed_tokens gather-dequant via the emb8 kernel in exllamav3_ext.

The kernel is compiled into the extension at install time
(exllamav3_ext/cpu/emb8.cpp): fused gather + dequant to fp16/fp32, threaded,
fresh output allocation per call.

Numerics: fp16 output is bit-exact with the torch path (ATen CPU fp16
elementwise ops use opmath_t = float, and vcvtps2ph rounds to nearest even like
c10::Half). fp32 output is exact vs the fp32 reference dequant (no intermediate
rounding).

Set EXL3_EMB8_FUSED=0 to force the pure-torch path in Embedding._gather.
"""
from __future__ import annotations

import os

import torch

try:
    from ..ext import exllamav3_ext as ext
except ImportError:
    ext = None

_gate = os.environ.get("EXL3_EMB8_FUSED", "1").lower() not in ("0", "off", "no", "false")


def available() -> bool:
    return _gate and ext is not None and hasattr(ext, "emb8_dequant")


def dequant(q_table: torch.Tensor, scale_table: torch.Tensor, ids: torch.Tensor,
            out_dtype: torch.dtype) -> torch.Tensor:
    """Gather rows of q_table by ids and dequantize; [n, hidden] out."""
    if ids.dtype != torch.int64:
        ids = ids.to(torch.int64)
    return ext.emb8_dequant(q_table, scale_table, ids, out_dtype == torch.float32)


def emb8_dequant_torch(q_table: torch.Tensor, scale_table: torch.Tensor,
                       ids: torch.Tensor, out_dtype: torch.dtype = torch.float32
                       ) -> torch.Tensor:
    """Reference multi-pass torch implementation (kept for tests/benchmarks)."""
    q = q_table[ids]
    s = scale_table[ids]
    x = q.to(out_dtype).view(*ids.shape, q_table.shape[1] // 32, 32) * s.to(out_dtype).unsqueeze(-1)
    return x.view(*ids.shape, q_table.shape[1])