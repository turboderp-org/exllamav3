"""
Differentiable LoRA-over-frozen-EXL3 linear (QLoRA proof of concept).

Background
----------
QLoRA freezes the (quantized) base weights and trains only the low-rank
adapter matrices A and B. Crucially, *no gradient ever needs to flow
through the quantizer*: the frozen weight behaves as a constant in the
computation graph, exactly like NF4 in bitsandbytes-based QLoRA. All we
need for a correct backward pass is:

  - ``grad_x``  -- to propagate loss to earlier layers, and
  - ``grad_A`` / ``grad_B`` -- to update the adapter.

both of which are ordinary matmuls once the effective FP16 weight has been
reconstructed from the EXL3 trellis. EXL3 already exposes that
reconstruction via ``LinearEXL3.get_weight_tensor()`` (the full effective
weight, with the sign flips ``suh``/``svh`` and Hadamard rotations folded
in), so the "scary" part -- differentiating a 3-bit trellis code -- never
enters the picture for QLoRA.

This module provides two things:

  1. ``reference_forward`` -- the dead-simple version that just runs plain
     torch ops with the dequantized weight detached to a constant, and
     lets autograd do everything. This is the ground truth.

  2. ``EXL3LoRAFunction`` -- a memory-efficient ``autograd.Function`` that
     does *not* stash the dequantized weight for the backward pass but
     instead re-derives it from the (cheap, already-on-device) trellis via
     a ``weight_fn`` closure. This is the shape of the kernel a real
     training path would use: it keeps the big FP16 weight out of the saved
     activation set. Its hand-written backward is validated against (1) and
     against ``torch.autograd.gradcheck`` in ``tests/test_qlora_grad.py``.

Nothing here touches the inference path. It is deliberately standalone so
the "is this even real?" question can be answered by a single gradcheck.
"""

from __future__ import annotations
from typing import Callable, Optional
import math
import torch
import torch.nn as nn


WeightFn = Callable[[], torch.Tensor]


def reference_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    scale: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Ground-truth forward using only plain torch ops.

    ``weight`` is the dequantized effective weight, shape ``[in, out]``,
    such that the base projection is ``y = x @ weight``. It is treated as a
    constant (its ``requires_grad`` is irrelevant -- we never want a grad
    for the frozen base). ``lora_a`` is ``[in, r]`` and ``lora_b`` is
    ``[r, out]``, matching the orientation ExLlamaV3's inference LoRA loader
    already uses (``x @ A @ B``).

    Autograd handles the entire backward for free; this exists so the
    custom Function below can be checked against it.
    """
    w_const = weight.detach()
    y = x @ w_const
    if lora_a is not None and lora_b is not None:
        y = y + scale * ((x @ lora_a) @ lora_b)
    if bias is not None:
        y = y + bias
    return y


class EXL3LoRAFunction(torch.autograd.Function):
    """
    Memory-efficient differentiable LoRA-over-frozen-weight linear.

    The forward reconstructs the dequantized weight via ``weight_fn`` (e.g.
    ``lambda: linear.inner.get_weight_tensor()``), uses it, and then throws
    it away. The backward re-reconstructs it rather than carrying the full
    FP16 weight through the saved-tensors set, mirroring how a real EXL3
    training kernel would trade a little recompute for a lot of activation
    memory.

    Tensor shapes
    -------------
      x      : ``[*, in]``
      weight : ``[in, out]``     (frozen, returned by ``weight_fn``)
      A      : ``[in, r]``       (trainable)
      B      : ``[r, out]``      (trainable)
      bias   : ``[out]``         (optional, trainable)
      y      : ``[*, out]``
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lora_a: Optional[torch.Tensor],
        lora_b: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale: float,
        weight_fn: WeightFn,
    ) -> torch.Tensor:
        weight = weight_fn()  # [in, out], constant w.r.t. autograd
        assert weight.dim() == 2, "weight_fn must return a 2D [in, out] tensor"

        y = x @ weight
        if lora_a is not None and lora_b is not None:
            # Adapters are kept as fp32 master weights but the matmul runs in
            # the compute dtype (bf16/fp16); cast for the product. (A no-op when
            # everything is already the same dtype, e.g. float64 gradcheck.)
            a = lora_a.to(x.dtype)
            b = lora_b.to(x.dtype)
            y = y + scale * ((x @ a) @ b)
        if bias is not None:
            y = y + bias.to(y.dtype)

        # Deliberately do NOT save `weight`; recompute it in backward.
        ctx.save_for_backward(x, lora_a, lora_b)
        ctx.scale = scale
        ctx.weight_fn = weight_fn
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, lora_a, lora_b = ctx.saved_tensors
        scale = ctx.scale
        weight = ctx.weight_fn()  # recomputed, not stored

        in_features = x.shape[-1]
        out_features = grad_y.shape[-1]
        xf = x.reshape(-1, in_features)
        gf = grad_y.reshape(-1, out_features)

        # grad wrt input: base path always present.
        grad_x = grad_y @ weight.transpose(-1, -2)

        grad_a = grad_b = grad_bias = None

        if lora_a is not None and lora_b is not None:
            # y_lora = scale * (x @ A) @ B
            # Math in the grad/compute dtype; grads cast back to the adapter's
            # (fp32 master) dtype before returning.
            a = lora_a.to(gf.dtype)
            b = lora_b.to(gf.dtype)
            g_through_b = gf @ b.transpose(-1, -2)               # [N, r]
            # contribution of LoRA branch to grad_x
            grad_x = grad_x + scale * (g_through_b @ a.transpose(-1, -2)).view_as(grad_x)
            if ctx.needs_input_grad[1]:
                grad_a = (scale * (xf.transpose(-1, -2) @ g_through_b)).to(lora_a.dtype)
            if ctx.needs_input_grad[2]:
                p = xf @ a                                        # [N, r]
                grad_b = (scale * (p.transpose(-1, -2) @ gf)).to(lora_b.dtype)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = gf.sum(dim=0)

        # Frozen weight (None), scale (None), weight_fn (None) get no grad.
        if not ctx.needs_input_grad[0]:
            grad_x = None

        return grad_x, grad_a, grad_b, grad_bias, None, None


class EXL3LoRAHadFunction(torch.autograd.Function):
    """
    Activation-side-transform variant of :class:`EXL3LoRAFunction` (audit A1).

    ``get_weight_tensor()`` builds the effective weight as
    ``W_eff = diag(suh) @ H @ W_inner @ H @ diag(svh)`` (H = the normalized
    128-block Hadamard, symmetric and orthogonal), which costs four extra
    full-weight passes + a dtype cast on EVERY reconstruction. Since the
    transforms are cheap on activations, this Function reconstructs only the
    inner weight and computes

        y_base = had(had(x * suh) @ W_inner) * svh

    -- the same math exllamav3's own ``reconstruct_hgemm`` inference path uses.
    The base matmul runs in the inner weight's dtype (the COMPUTE dtype when
    ``frozen_trellis_parts`` was built with one -- the reconstruct kernel
    emits it directly, so every ``.to()`` below is a no-op; whatever the mock
    closure returns under tests otherwise), with the result cast to the
    compute dtype. The backward adjoint uses the transforms' self-
    transposedness (H^T = H, diagonals symmetric):

        grad_x_base = had(had(grad_y * svh) @ W_inner^T) * suh

    computed in the GRAD dtype (a full-weight cast only when the inner dtype
    differs, e.g. an fp16 inner under fp32 debug compute).

    The frozen PiSSA offset, folded into the weight closure on the legacy path
    (``W_eff - offs_scale*(a0@b0)``, a full-weight addmm per reconstruction),
    is passed here as ``offs_a``/``offs_b`` and applied as a low-rank
    activation term instead: ``y -= offs_scale*(x@a0)@b0``.

    Tensor shapes: as :class:`EXL3LoRAFunction`, plus ``suh [in]``, ``svh
    [out]``, ``had [128, 128]``, ``offs_a [in, r]``, ``offs_b [r, out]``.
    ``inner_fn`` returns the UNtransformed ``[in, out]`` inner weight
    (``backbone.frozen_trellis_parts``), recomputed in backward, not stored.
    """

    @staticmethod
    def _base_matmul(x2d: torch.Tensor, weight: torch.Tensor, suh, svh, had):
        n_in, n_out = weight.shape
        xh = ((x2d * suh).view(-1, n_in // 128, 128) @ had).view(-1, n_in)
        y = xh @ weight
        y = ((y.view(-1, n_out // 128, 128) @ had).view(-1, n_out)) * svh
        return y

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lora_a: Optional[torch.Tensor],
        lora_b: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale: float,
        inner_fn: WeightFn,
        suh: torch.Tensor,
        svh: torch.Tensor,
        had: torch.Tensor,
        offs_a: Optional[torch.Tensor],
        offs_b: Optional[torch.Tensor],
        offs_scale: float,
    ) -> torch.Tensor:
        weight = inner_fn()  # [in, out] inner weight, constant w.r.t. autograd
        wd = weight.dtype
        n_in = weight.shape[0]
        xf = x.reshape(-1, n_in)

        yb = EXL3LoRAHadFunction._base_matmul(
            xf.to(wd), weight, suh.to(wd), svh.to(wd), had.to(wd))
        y = yb.to(x.dtype)

        if offs_a is not None:
            a0 = offs_a.to(x.dtype)
            b0 = offs_b.to(x.dtype)
            y = y - offs_scale * ((xf @ a0) @ b0)
        if lora_a is not None and lora_b is not None:
            a = lora_a.to(x.dtype)
            b = lora_b.to(x.dtype)
            y = y + scale * ((xf @ a) @ b)
        if bias is not None:
            y = y + bias.to(y.dtype)

        ctx.save_for_backward(x, lora_a, lora_b, offs_a, offs_b)
        ctx.scale = scale
        ctx.offs_scale = offs_scale
        ctx.inner_fn = inner_fn
        ctx.transforms = (suh, svh, had)
        ctx.has_bias = bias is not None
        return y.view(x.shape[:-1] + (weight.shape[1],))

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, lora_a, lora_b, offs_a, offs_b = ctx.saved_tensors
        scale = ctx.scale
        suh, svh, had = ctx.transforms

        in_features = x.shape[-1]
        out_features = grad_y.shape[-1]
        xf = x.reshape(-1, in_features)
        gf = grad_y.reshape(-1, out_features)
        gd = gf.dtype

        grad_x = grad_a = grad_b = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Adjoint of the base map: the same transform sandwich with W and
            # H transposed and suh/svh swapped (the diagonals are their own
            # transpose; the runtime H happens to be symmetric too, but the
            # adjoint doesn't rely on it). Runs in the grad dtype -- this is
            # where the one full-weight cast lives (fp16 grads would risk
            # overflow).
            weight = ctx.inner_fn()  # recomputed (or backward-cache hit)
            grad_x = EXL3LoRAHadFunction._base_matmul(
                gf, weight.to(gd).transpose(-1, -2),
                svh.to(gd), suh.to(gd), had.to(gd).transpose(-1, -2),
            ).view_as(x)

        if offs_a is not None and grad_x is not None:
            a0 = offs_a.to(gd)
            b0 = offs_b.to(gd)
            grad_x = grad_x - ctx.offs_scale * (
                (gf @ b0.transpose(-1, -2)) @ a0.transpose(-1, -2)).view_as(x)

        if lora_a is not None and lora_b is not None:
            a = lora_a.to(gd)
            b = lora_b.to(gd)
            g_through_b = gf @ b.transpose(-1, -2)               # [N, r]
            if grad_x is not None:
                grad_x = grad_x + scale * (
                    g_through_b @ a.transpose(-1, -2)).view_as(x)
            if ctx.needs_input_grad[1]:
                grad_a = (scale * (xf.transpose(-1, -2) @ g_through_b)).to(lora_a.dtype)
            if ctx.needs_input_grad[2]:
                p = xf @ a                                        # [N, r]
                grad_b = (scale * (p.transpose(-1, -2) @ gf)).to(lora_b.dtype)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = gf.sum(dim=0)

        return (grad_x, grad_a, grad_b, grad_bias,
                None, None, None, None, None, None, None, None)


def qlora_had_linear_forward(
    x: torch.Tensor,
    inner_fn: WeightFn,
    suh: torch.Tensor,
    svh: torch.Tensor,
    had: torch.Tensor,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
    offs_a: Optional[torch.Tensor] = None,
    offs_b: Optional[torch.Tensor] = None,
    offs_scale: float = 1.0,
) -> torch.Tensor:
    """Functional entry point for the activation-side-transform Function."""
    return EXL3LoRAHadFunction.apply(
        x, lora_a, lora_b, bias, scale, inner_fn, suh, svh, had,
        offs_a, offs_b, offs_scale)


def qlora_linear_forward(
    x: torch.Tensor,
    weight_fn: WeightFn,
    lora_a: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional entry point using the memory-efficient Function."""
    return EXL3LoRAFunction.apply(x, lora_a, lora_b, bias, scale, weight_fn)


def init_lora_a_(t: torch.Tensor, fan_in: int) -> torch.Tensor:
    """PEFT's LoRA-A init on an A stored as ``[in_features, r]``.

    PEFT stores A as ``[r, in]`` and uses torch's kaiming-uniform init with
    a=sqrt(5), whose bound is ``1/sqrt(fan_in)`` with fan_in read from dim 1 =
    ``in``. Our A is transposed, so the same call read fan_in = r and
    initialised A ``sqrt(in/r)`` too large (8.9x at r=32 on a 2560-wide input),
    which made the effective update per Adam step that much larger than PEFT's
    at the same lr (found 2026-10-11, doc/exl3_vs_bnb_qwen35_4b_2026-10-11.md).
    Explicit bound, independent of storage layout.
    """
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        return t.uniform_(-bound, bound)


class QLoRALinear(nn.Module):
    """
    nn.Module wrapper turning a frozen ExLlamaV3 ``Linear`` into a trainable
    QLoRA layer.

    The base weight is never copied or updated; on every forward it is
    reconstructed on the fly from the EXL3 trellis (or read directly, for an
    fp16 base) through ``weight_fn``. Only ``lora_a`` / ``lora_b`` are
    registered parameters and receive gradients.

    Parameters
    ----------
    weight_fn:
        Callable returning the frozen ``[in, out]`` effective weight. For a
        loaded EXL3 layer this is typically
        ``lambda: linear.inner.get_weight_tensor()``. The returned tensor's
        dtype/device define the compute dtype/device of the base projection.
    in_features, out_features:
        Padded feature dims of the wrapped layer.
    r, alpha:
        LoRA rank and scaling (``scale = alpha / r``, optionally rslora).
    """

    def __init__(
        self,
        weight_fn: WeightFn,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.weight_fn = weight_fn
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        denom = (r ** 0.5) if use_rslora else r
        self.scale = float(alpha) / float(denom)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # PEFT-style init: A ~ kaiming/normal, B = 0 so the adapter starts
        # as a no-op and training begins from the exact base model.
        self.lora_a = nn.Parameter(torch.empty(in_features, r, dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros(r, out_features, dtype=dtype, device=device))
        init_lora_a_(self.lora_a, in_features)   # PEFT bound 1/sqrt(in)

        # Bias from the base layer is frozen (not trained here).
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_in = self.dropout(x) if self.dropout is not None else x
        # Note: dropout differs between the two LoRA branches only if applied
        # before the base; here the base path uses the raw x and the adapter
        # uses dropped x. To keep the PoC's gradcheck exact we route both
        # through the same Function, so apply dropout to the adapter inputs
        # by folding it into A is not possible -- keep dropout off for grad
        # checking (default dropout=0.0).
        return EXL3LoRAFunction.apply(
            lora_in, self.lora_a, self.lora_b, self.bias, self.scale, self.weight_fn
        )

    @classmethod
    def from_exl3_linear(
        cls,
        linear,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> "QLoRALinear":
        """
        Build a trainable wrapper around a loaded ExLlamaV3 ``Linear``.

        Works for both EXL3 and fp16 inner layers, since both expose
        ``get_weight_tensor()`` returning the ``[in, out]`` effective weight.
        """
        inner = linear.inner
        device = linear.device

        def weight_fn() -> torch.Tensor:
            w = inner.get_weight_tensor()      # [in, out], half
            return w.to(dtype)

        bias = None
        get_bias = getattr(inner, "get_bias_tensor", None)
        if get_bias is not None:
            b = get_bias()
            if b is not None:
                bias = b.to(dtype)

        return cls(
            weight_fn=weight_fn,
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            use_rslora=use_rslora,
            bias=bias,
            dtype=dtype,
            device=device,
        )
