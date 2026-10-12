"""
Synthetic EXL3 weights and their references.

A random trellis is a valid EXL3 tensor: every 16-bit word decodes to some codebook value, so kernel and
module tests don't need a quantized checkpoint. The references here dequantize through the reconstruct
kernels, which isolates the kernel under test (GEMM/GEMV/MoE tiling, reductions) from the codebook decode
itself; the decode is covered against its own definition in the quantization tests.

Layout conventions (shared by the kernels and the checkpoint format):
    trellis   (k // 16, n // 16, 16 * K) int16, K bits per weight (half-integer K: 16 * K words)
    suh       (k,) fp16 input sign/scale, applied before the input Hadamard
    svh       (n,) fp16 output sign/scale, applied after the output Hadamard
"""

import torch

from exllamav3.modules.quant.exl3_lib.quantize import codebook_mcg_mult, codebook_mul1_mult

CODEBOOKS = ("3inst", "mcg", "mul1")


def codebook_flags(codebook: str) -> tuple[bool, bool]:
    """(mcg, mul1) kernel flags for a codebook name"""
    assert codebook in CODEBOOKS, codebook
    return codebook == "mcg", codebook == "mul1"


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device = "cpu").manual_seed(seed)


def rand_trellis(k: int, n: int, K: float, gen: torch.Generator | None = None, device = None) -> torch.Tensor:
    """Uniformly random trellis words for a (k, n) weight at K bits (K may be a half integer)"""
    words = int(16 * K)
    assert words == 16 * K, f"K = {K} is not a multiple of 1/16"
    t = torch.randint(0, 65536, (k // 16, n // 16, words), dtype = torch.int32, generator = gen).to(torch.int16)
    return t.to(device) if device is not None else t


def rand_scale(n: int, gen: torch.Generator | None = None, device = None) -> torch.Tensor:
    """Random-sign scales of magnitude 0.9..1.1, the typical shape of suh/svh"""
    mag = torch.rand((n,), generator = gen) * 0.2 + 0.9
    sign = torch.where(torch.rand((n,), generator = gen) < 0.5, -1.0, 1.0)
    t = (mag * sign).half()
    return t.to(device) if device is not None else t


def rand_linear(k: int, n: int, K: float, gen: torch.Generator | None = None, device = None) -> dict:
    """{"trellis", "suh", "svh"} for one random EXL3 linear"""
    return {
        "trellis": rand_trellis(k, n, K, gen, device),
        "suh": rand_scale(k, gen, device),
        "svh": rand_scale(n, gen, device),
    }


def rand_experts(E: int, H: int, I: int, K: float, gen: torch.Generator | None = None) -> dict[str, list]:
    """Random gate/up/down EXL3 experts: {"g" | "u" | "d": [(trellis, suh, svh)] * E}, gate/up (H -> I), down (I -> H)"""
    return {p: [(rand_trellis(k, n, K, gen), rand_scale(k, gen), rand_scale(n, gen)) for _ in range(E)]
            for p, (k, n) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H)))}


def checkpoint_tensors(key: str, k: int, n: int, K: float, gen: torch.Generator | None = None,
                       codebook: str = "mul1", bias: bool = False) -> dict[str, torch.Tensor]:
    """Tensors of one EXL3 linear as stored in a checkpoint (key.trellis/.suh/.svh + codebook marker)"""
    w = rand_linear(k, n, K, gen)
    t = {f"{key}.trellis": w["trellis"], f"{key}.suh": w["suh"], f"{key}.svh": w["svh"]}
    if codebook == "mcg":
        t[f"{key}.mcg"] = torch.tensor(codebook_mcg_mult, dtype = torch.uint32).view(torch.int)
    elif codebook == "mul1":
        t[f"{key}.mul1"] = torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
    if bias:
        t[f"{key}.bias"] = (torch.randn(n, generator = gen) * 0.1).half()
    return t


def dequant(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: float,
            codebook: str = "mul1") -> torch.Tensor:
    """Dense fp16 (k, n) weight equivalent to the EXL3 linear, Hadamards and scales folded in"""
    from exllamav3.ext import exllamav3_ext as ext
    mcg, mul1 = codebook_flags(codebook)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    w = torch.empty((k, n), dtype = torch.half, device = trellis.device)
    ext.reconstruct_had_slice(w, trellis, suh.to(trellis.device), svh.to(trellis.device), K, mcg, mul1, 0)
    return w


def linear_ref(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: float,
               codebook: str = "mul1") -> torch.Tensor:
    """Reference x @ W of an EXL3 linear in fp32 (fp16 output), through the dense dequantized weight"""
    w = dequant(trellis.to(x.device), suh, svh, K, codebook)
    return (x.float() @ w.float()).half()


def act_ref(act: str, gated: bool, g: torch.Tensor | None, u: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """Reference MLP activations as the fused kernels define them. Gated: act(g) * u for silu, gelu (tanh),
    relu2, or gpt-oss swiglu_oai ((u + 1) * g * sigmoid(1.702 g)); gateless: relu(u) * u. A nonzero limit clamps
    the activation from above and u to [-limit, limit]"""
    if not gated:
        x = torch.relu(u)
    elif act == "silu":
        x = torch.nn.functional.silu(g)
    elif act == "gelu":
        x = torch.nn.functional.gelu(g, approximate = "tanh")
    elif act == "relu2":
        x = torch.relu(g) ** 2
    elif act == "swiglu_oai":
        if limit:
            g = g.clamp(max = limit)
            u = u.clamp(-limit, limit)
        return (u + 1.0) * g * torch.sigmoid(1.702 * g)
    else:
        raise ValueError(act)
    if limit:
        u = u.clamp(-limit, limit)
        x = x.clamp(max = limit)
    return x * u
