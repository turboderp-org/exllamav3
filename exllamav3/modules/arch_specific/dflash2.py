"""
DFlash2-specific modules: grouped dynamic convolutions, the conv-wrapped
transformer block, and the top-k candidate selector.

Reference: the ``DFlash2DraftModel`` implementation in the ``dflash`` package.

The residual stream stays in fp32, matching the standard transformer path.
RMSNorm and convolution prepare outputs are fp16; convolution finish returns
fp32 before the residual add.
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ...model.config import Config
from .. import Module, Linear, RMSNorm, Attention, GatedMLP
import os
from ...util.tensor import to2

from ...ext import exllamav3_ext as ext

# EXL3_DFLASH2_HOST_CB: keep the selector codebooks (raw fp16/bf16 [vocab, rank] or the packed
# integer values) in pinned host RAM and serve the CUDA kernels through a zero-copy device alias,
# exactly like Linear.pin_linears: the kernels take the alias (a real CUDA tensor whose pages live
# in host memory), so the whole bulk footprint leaves VRAM and every gather read goes over PCIe.
# The per-block scales/mins (kilobytes, touched per element) stay in VRAM. Env-gated for A/B testing.
_dflash2_host_cb = os.environ.get("EXL3_DFLASH2_HOST_CB", "0") != "0"


def _grouped_dynamic_convolve_torch(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.float().view(batch, length, groups, group_size)
    dynamic = dynamic.float().view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(min(base.shape[0], length)):
        values = blocks[:, : length - offset]
        kernel = base[offset].float().view(1, 1, groups, group_size)
        weights = kernel + dynamic[:, offset:, offset]
        output[:, offset:] += weights * values
    output = output.view(batch, length, hidden_size)
    if residual is not None:
        residual += output
        return residual
    return output.to(hidden.dtype)


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transcribed from dflash.model._grouped_dynamic_convolve.

    hidden  [b, l, H]; dynamic [b, l, taps, H//group_size] (any strides); base [taps, H].
    output[t] = sum_taps  base[tap] * x[t - tap] + dyn[tap][t] * x[t - tap]
    (causal; tap 0 = current position). Result has hidden's dtype, or, with residual (fp32
    [b, l, H]), is added into residual in place and residual is returned (the finish()
    variant's residual add, fused into the kernel).
    """
    if not hidden.is_cuda:
        return _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size, residual)
    hidden = hidden.contiguous()
    if residual is not None:
        ext.dflash2_dynconv(hidden, dynamic, base, residual, group_size, True)
        return residual
    output = torch.empty_like(hidden)
    ext.dflash2_dynconv(hidden, dynamic, base, output, group_size, False)
    return output


class DFlash2DynConv(Module):
    """Two-tap grouped dynamic conv (dflash ``GroupedDynamicCausalConv``).

    Checkpoint tensors (raw, unquantized, bf16):
      {key}.base_kernel        [2, kernel_size, hidden]   (prepare base, finish base)
      {key}.kernel_projection  Linear(hidden -> 2 * kernel_size * groups)

    prepare() uses fp16 input/output. finish() adds the fp32 result into the block's
    fp32 residual stream in place (one kernel, no separate add).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        kernel_size: int,
        group_size: int,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2DynConv"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size

        self.proj = Linear(
            config = config,
            key = f"{key}.kernel_projection",
            in_features = hidden_size,
            out_features = 2 * kernel_size * self.groups,
            qmap = qmap,
            trim_padded_out = True,
        )
        self.register_submodule(self.proj)

        self.base_kernel = None
        self.key_base_kernel = f"{key}.base_kernel"
        self.base_kernel_numel = 2 * kernel_size * hidden_size
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return self.base_kernel_numel + super().weights_numel()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.base_kernel = self.config.stc.get_tensor(
            self.key_base_kernel, self.device, optional = False, allow_bf16 = True
        )
        expected_shape = (2, self.kernel_size, self.hidden_size)
        if self.base_kernel.shape != expected_shape:
            raise ValueError(
                f"Expected {self.key_base_kernel} shape {expected_shape}, "
                f"got {tuple(self.base_kernel.shape)}"
            )

    @override
    def unload(self):
        self.base_kernel = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.base_kernel is not None:
            t[self.key_base_kernel] = self.base_kernel.contiguous()
        return t

    def prepare(self, x: torch.Tensor, params: dict):
        """x [b, l, H] (post-norm) -> (convolved half, finish-time dynamic half)"""
        x = x.half()
        dyn = self.proj.forward(x, params)
        dyn = dyn.view(*x.shape[:-1], 2, self.kernel_size, self.groups)
        y = _grouped_dynamic_convolve(
            x, dyn[..., 0, :, :], self.base_kernel[0], self.group_size)
        return y, dyn[..., 1, :, :]

    def finish(self, x: torch.Tensor, dynamic: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """Sublayer output x [b, l, H] (fp16 or fp32) -> conv result in fp32, or, with residual
        (fp32 [b, l, H]), residual += conv result in place (returned)"""
        if residual is not None:
            return _grouped_dynamic_convolve(
                x, dynamic, self.base_kernel[1], self.group_size, residual = residual)
        return _grouped_dynamic_convolve(
            x.float(), dynamic, self.base_kernel[1], self.group_size)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        y, dyn = self.prepare(x, params)
        return to2(self.finish(y, dyn), out_dtype, torch.float)


class DFlash2Block(Module):
    """Reference DFlash2 decoder layer (Qwen3DFlashDecoderLayer):

        r = x; x = attn_norm(x); x, k = attn_conv.prepare(x); x = attn(x);
        x = attn_conv.finish(x, k); x = r + x
        r = x; x = mlp_norm(x);  x, k = mlp_conv.prepare(x);  x = mlp(x);
        x = mlp_conv.finish(x, k); x = r + x

    Residual stream fp32; normed sub-ops fp16. The residual adds are fused into the
    finish() convolutions.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        attn: Attention,
        mlp: GatedMLP,
        attn_norm: RMSNorm,
        mlp_norm: RMSNorm,
        attn_conv: DFlash2DynConv,
        mlp_conv: DFlash2DynConv,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2Block"
        self.layer_idx = layer_idx
        self.attn = attn
        self.mlp = mlp
        self.attn_norm = attn_norm
        self.mlp_norm = mlp_norm
        self.attn_conv = attn_conv
        self.mlp_conv = mlp_conv
        for m in (attn, mlp, attn_norm, mlp_norm, attn_conv, mlp_conv):
            self.register_submodule(m)

    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        x = x.float() if x.dtype != torch.float else x
        y = self.attn_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.attn_conv.prepare(y, params)
        y = self.attn.forward(y, params)
        x = self.attn_conv.finish(y, kernel, residual = x)

        y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.mlp_conv.prepare(y, params)
        y = self.mlp.forward(y, params)
        x = self.mlp_conv.finish(y, kernel, residual = x)

        return to2(x, out_dtype, torch.float)


class DFlash2Selector(Module):
    """Top-k candidate selector (dflash ``CandidateSelector``).

    Checkpoint tensors (BARE keys — no .weight suffix):
      candidate_selector.predecessor_codebook  [vocab, rank]  (raw or quantized)
      candidate_selector.successor_codebook    [vocab, rank]  (raw or quantized)
      candidate_selector.hidden_projection     Linear(hidden -> rank, no bias)

    Quantized codebooks replace a raw tensor with per-32-block integer tensors: {key}.q
    ([vocab, qbytes+1] uint8 packed values), {key}.scales ([vocab, rank/32] fp16) and, for the
    asymmetric _1 forms, {key}.mins ([vocab, rank/32] fp16). Rates (actual bpw): Q8_0 8.5
    (lossless for the selector), Q4_1 5.0, Q4_0 4.5, Q3_1 4.0, Q3_0 3.5, Q2_1 3.0, Q2_0 2.5.
    Row-based (not tile-based), so the serving gather is a contiguous row read with no
    cross-row traffic (see dflash2.cu). Served by ext.dflash2_cb_gather_int +
    ext.dflash2_selector_walk_staged, CUDA only. Both codebooks are raw or both quantized.
    With EXL3_DFLASH2_HOST_CB=1 the bulk tensors stay in pinned host RAM and the kernels read
    them zero-copy (see _pin_alias); get_tensors then exports the pinned sources.

    walk(): top-k(16) per row from draft logits, then greedy chained walk
      S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>,  a = previous path token
    """

    def __init__(
        self,
        config: Config,
        key: str,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2Selector"
        self.vocab_size = vocab_size
        self.rank = rank
        self.top_k = top_k

        self.hidden_proj = Linear(
            config = config,
            key = f"{key}.hidden_projection",
            in_features = hidden_size,
            out_features = rank,
            trim_padded_out = True,
        )
        self.register_submodule(self.hidden_proj)

        self.key_pred = f"{key}.predecessor_codebook"
        self.key_succ = f"{key}.successor_codebook"
        self.pred_codebook = None
        self.succ_codebook = None
        self.pred_q = None
        self.succ_q = None
        self.quantized = False
        self.svh_AB = None
        self._pinned_store = {}
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return 2 * self.vocab_size * self.rank + super().weights_numel()

    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        # The selector is part of the module list so loading, autosplit and compilation account
        # for its tensors. Proposal generation invokes walk() after the shared target LM head.
        return to2(x, out_dtype, None)

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        # Codebooks come either raw (fp16/bf16 [vocab, rank]) or quantized as per-32-block integer
        # tensors ({key}.q / .scales / .mins); the .q key wins if present. Older trainers append
        # .weight to the bare codebook keys; try bare first, fall back to suffixed (same bytes).
        self.pred_codebook, self.pred_q = self._load_codebook(self.key_pred)
        self.succ_codebook, self.succ_q = self._load_codebook(self.key_succ)
        self.quantized = self.pred_q is not None
        if self.quantized != (self.succ_q is not None):
            raise ValueError(f"{self.key_pred} and {self.key_succ} must both be quantized or both raw")
        if self.quantized:
            if device.type != "cuda":
                raise ValueError(f"{self.key_pred}: quantized codebooks require a CUDA device")
            # The integer dequant yields the full row value (per-block scale+min folded in), so the
            # walk's column scale is a no-op
            self.svh_AB = torch.ones(self.rank, dtype = torch.half, device = device)
        else:
            expected_shape = (self.vocab_size, self.rank)
            for name, cb in ((self.key_pred, self.pred_codebook), (self.key_succ, self.succ_codebook)):
                if cb.shape != expected_shape:
                    raise ValueError(
                        f"Expected {name} shape {expected_shape}, got {tuple(cb.shape)}"
                    )


    def _load_codebook(self, key: str) -> tuple[torch.Tensor | None, dict | None]:
        # Host-resident mode (EXL3_DFLASH2_HOST_CB): the bulk tensor (raw codebook or the packed
        # integer values) loads into CPU memory and is page-locked + zero-copy aliased; no_defer
        # because the pinned copy must be complete when we take it. The per-block scales/mins are
        # kilobytes touched per element and stay in VRAM
        host = _dflash2_host_cb and self.device.type == "cuda"
        ld = torch.device("cpu") if host else self.device
        q = self.config.stc.get_tensor(f"{key}.q", ld, optional = True, no_defer = host, arena = not host)
        if q is None:
            cb = self.config.stc.get_tensor(key, ld, optional = True, allow_bf16 = True, no_defer = host, arena = not host)
            if cb is None:
                cb = self.config.stc.get_tensor(key + ".weight", ld, optional = False, allow_bf16 = True, no_defer = host, arena = not host)
            return (self._pin_alias(key, cb) if host else cb), None
        if self.rank != 256:
            raise ValueError(f"{key}: quantized codebooks require rank 256 (got {self.rank})")
        q = q.contiguous()
        if q.size(0) != self.vocab_size:
            raise ValueError(f"Expected {key}.q shape ({self.vocab_size}, ...), got {tuple(q.shape)}")
        scales = self.config.stc.get_tensor(f"{key}.scales", self.device, optional = False).contiguous()
        mins = self.config.stc.get_tensor(f"{key}.mins", self.device, optional = True)
        if mins is not None:
            mins = mins.contiguous()
        out = {
            "q": self._pin_alias(f"{key}.q", q) if host else q,
            "scales": scales,
            "mins": mins,
        }
        return None, out


    def _pin_alias(self, key: str, t: torch.Tensor) -> torch.Tensor:
        # Page-lock a copy of the bulk tensor and hand the kernels a zero-copy CUDA alias of
        # it (same pattern as Linear.pin_linears). The alias does not own the memory: the
        # pinned source lives in _pinned_store for the module's lifetime and doubles as the
        # export value in get_tensors
        if t.device.type != "cpu":
            raise ValueError(f"{key}: expected a CPU tensor to pin, got {t.device}")
        pinned = t.pin_memory()
        self._pinned_store[key] = pinned
        return ext.pinned_cuda_view(pinned, self.device.index if self.device.index is not None else 0)



    def codebook_targets(self) -> list[tuple[str, int]]:
        # Recipe-visible keys and element counts of the two codebooks (quantization opt-in:
        # create_q_strategy_from_recipe only budgets a codebook when the recipe names it)
        numel = self.vocab_size * self.rank
        return [(self.key_pred, numel), (self.key_succ, numel)]


    def convert_codebooks(self, args: dict, idx: int, devices: list, strategy: dict):
        # Conversion time: replace raw codebooks with per-32-block integer tensors (Q8_0 / Q4_1 /
        # Q4_0 / Q3_1 / Q3_0 / Q2_1 / Q2_0, selected by the recipe's actual-bpw K). Row-based, so
        # the serving gather is a contiguous row read with no cross-row traffic (see the comment
        # in exllamav3_ext/dflash2.cu). Both codebooks quantize or neither does.
        kpred = strategy.get(self.key_pred)
        ksucc = strategy.get(self.key_succ)
        if kpred is None and ksucc is None:
            return
        if (kpred is None or kpred == 16) and (ksucc is None or ksucc == 16):
            return
        qpred, qsucc = kpred is not None and kpred != 16, ksucc is not None and ksucc != 16
        if qpred != qsucc:
            raise ValueError(
                f"{self.key_pred} and {self.key_succ} must both be quantized or both raw "
                f"(got {kpred} / {ksucc})"
            )
        device = torch.device(devices[0])
        for key, K, raw in ((self.key_pred, kpred, self.pred_codebook), (self.key_succ, ksucc, self.succ_codebook)):
            if K is None or K == 16:
                continue
            bits, asymmetric = self._bpw_to_format(K)
            packed, scales, mins, weight_q = self._quantize_codebook_int(
                raw.float().to(device), bits, asymmetric, device
            )
            q = {"q": packed.cpu(), "scales": scales.cpu()}
            if mins is not None:
                q["mins"] = mins.cpu()
            if key == self.key_pred:
                self.pred_q = q
                self.pred_codebook = None
            else:
                self.succ_q = q
                self.succ_codebook = None
            mse = ((raw.float().to(device) - weight_q).pow(2).mean() / raw.float().to(device).pow(2).mean()).item()
            name = f"Q{bits}_{'1' if asymmetric else '0'}"
            print(
                f" -- Quantized: {key:{max(32, len(key))}}  {name}  "
                f"bpw: {K:5.2f}  rel-mse: {mse:.6e}",
                flush = True
            )


    @staticmethod
    def _bpw_to_format(K: float) -> tuple[int, bool]:
        # Recipe bit-rate (actual bpw incl. per-block scale/min overhead) -> (integer bits,
        # asymmetric). Each 0.5bpw step is a format: the _1 (asymmetric, per-block min) form is
        # the denser rate, the _0 (symmetric) form the half-step below it. 8-bit is Q8_0.
        table = {
            8.5: (8, False),   # Q8_0
            5.0: (4, True),    # Q4_1
            4.5: (4, False),   # Q4_0
            4.0: (3, True),    # Q3_1
            3.5: (3, False),   # Q3_0
            3.0: (2, True),    # Q2_1
            2.5: (2, False),   # Q2_0
        }
        if K not in table:
            raise ValueError(f"unsupported codebook bit-rate {K} (use 8.5, 5.0, 4.5, 4.0, 3.5, 3.0, or 2.5)")
        return table[K]


    def _quantize_codebook_int(self, W: torch.Tensor, bits: int, asymmetric: bool, device: torch.device):
        # Per-32-block integer quantize -> (packed [V, qbytes+1] uint8, scales [V, 8] fp16,
        # mins [V, 8] fp16 | None, dequantized [V, R] float for the mse log). The q buffer is
        # padded by one byte per row so the Q3 cross-byte read in the gather kernel is in-bounds
        V, R = W.shape
        Wb = W.view(V, R // 32, 32)
        if asymmetric:
            mn = Wb.min(-1, keepdim = True).values
            mx = Wb.max(-1, keepdim = True).values
            d = ((mx - mn) / ((1 << bits) - 1)).clamp_min(1e-9)
            q = torch.round((Wb - mn) / d).clamp(0, (1 << bits) - 1)  # unsigned [0, 2^b-1]
            scales = d.squeeze(-1).half().contiguous()
            mins = mn.squeeze(-1).half().contiguous()
            weight_q = (q * d + mn).view(V, R)
        else:
            amax = Wb.abs().max(-1, keepdim = True).values
            hi = (1 << (bits - 1)) - 1
            d = (amax / hi).clamp_min(1e-9)
            q_signed = torch.round(Wb / d).clamp(-hi - 1, hi)         # signed [-2^(b-1), 2^(b-1)-1]
            q = q_signed % (1 << bits)                               # b-bit two's complement
            scales = d.squeeze(-1).half().contiguous()
            mins = None
            weight_q = (q_signed * d).view(V, R)
        # pack the (now unsigned [0, 2^b-1]) integer values into a byte stream, LSB-first
        q = q.view(V, R).to(torch.uint8)
        if bits == 8:
            packed = q.contiguous()                                   # [V, R]
        elif bits == 4:
            qq = q.view(V, R // 2, 2)
            packed = (qq[:, :, 0] | (qq[:, :, 1] << 4)).contiguous()  # [V, R//2]
        elif bits == 2:
            qq = q.view(V, R // 4, 4)
            packed = (qq[:, :, 0] | (qq[:, :, 1] << 2) | (qq[:, :, 2] << 4) | (qq[:, :, 3] << 6)).contiguous()
        else:  # bits == 3
            R3 = R * 3
            bitmat = (q.unsqueeze(-1) >> torch.tensor([0, 1, 2], device = device, dtype = torch.uint8)) & 1
            stream = bitmat.flatten(1)                                # [V, R*3]
            nbytes = (R3 + 7) // 8
            stream = F.pad(stream, (0, nbytes * 8 - R3)).view(V, nbytes, 8).to(torch.int32)
            weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device = device, dtype = torch.int32)
            packed = (stream * weights).sum(-1).to(torch.uint8).contiguous()  # [V, nbytes]
        packed = F.pad(packed, (0, 1))                                # [V, qbytes+1]
        return packed, scales, mins, weight_q

    @override
    def unload(self):
        self.pred_codebook = None
        self.succ_codebook = None
        self.pred_q = None
        self.succ_q = None
        self.quantized = False
        self.svh_AB = None
        self._pinned_store.clear()
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.pred_codebook is not None:
            t[self.key_pred] = self.pred_codebook.contiguous()
            t[self.key_succ] = self.succ_codebook.contiguous()
        for key, q in ((self.key_pred, self.pred_q), (self.key_succ, self.succ_q)):
            if q is not None:
                for subkey, tensor in q.items():
                    if tensor is not None:
                        t[f"{key}.{subkey}"] = tensor.contiguous()
        t.update(self._pinned_store)  # host-resident mode: export the sources, not the aliases
        return t

    def walk(
        self,
        hidden: torch.Tensor,        # [b, rows, H] post-norm draft state
        logits: torch.Tensor,        # [b, rows, V] float draft logits
        anchor_ids: torch.Tensor,    # [b]
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Greedily rerank each row's top-k tokens, conditioned on the preceding token.
        Returns the path [b, rows] (and per-row winning scores [b, rows])."""
        out, conf = self.walk_block(hidden, logits, anchor_ids, return_confidence)
        if return_confidence:
            return out[:, 1:], conf[:, 1:]
        return out[:, 1:]


    def walk_block(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        return_confidence: bool = False,
        vocab_size: int | None = None,
        scale: float = 1.0,
        softcap: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """walk() in the generator's block layout: ids [b, rows + 1] = [anchor, path...] and,
        when requested, confidence [b, rows + 1] = [0, winning score...]. logits may be wider
        than vocab_size (padded head) and are scaled / softcapped on the fly. On CUDA the top-k
        and the whole chain run as two kernels (no per-row host round trip, no torch
        intermediates)."""
        vocab_size = vocab_size or logits.shape[-1]
        # The generator passes state[:, 1:], a strided view for more than one row, and .half() on
        # an fp16 state returns that view as is
        gate = self.hidden_proj.forward(hidden.half().contiguous(), params = {})
        anchor_ids = anchor_ids.long()
        bsz, rows = logits.shape[:2]
        cuda = hidden.is_cuda and (self.quantized or self.pred_codebook.dtype in (torch.half, torch.bfloat16))
        if cuda and self.top_k in (8, 16, 32) and logits.stride(-1) == 1:
            unary = torch.empty((bsz, rows, self.top_k), dtype = torch.float, device = hidden.device)
            cands = torch.empty((bsz, rows, self.top_k), dtype = torch.long, device = hidden.device)
            ext.dflash2_topk(logits, vocab_size, scale, softcap, unary, cands)
        else:
            logits = logits[..., :vocab_size].float() * scale
            if softcap > 0.0:
                logits = torch.tanh(logits / softcap) * softcap
            unary, cands = torch.topk(logits, self.top_k, dim = -1, sorted = False)
        if cuda:
            anchor_ids = anchor_ids.to(hidden.device, non_blocking = anchor_ids.is_pinned()).contiguous()
            if self.quantized:
                return self._walk_staged(unary.float().contiguous(), cands.long().contiguous(), gate, anchor_ids, return_confidence)
            out = torch.empty((bsz, rows + 1), dtype = torch.long, device = hidden.device)
            conf = torch.empty((bsz, rows + 1), dtype = torch.float, device = hidden.device) if return_confidence else None
            ext.dflash2_selector_walk(
                unary.float().contiguous(), cands.long().contiguous(), gate.contiguous(),
                self.pred_codebook, self.succ_codebook, anchor_ids, out, conf,
            )
            return out, conf
        if self.quantized:
            return self._walk_torch_q(unary.float(), cands.long(), gate, anchor_ids, return_confidence)
        return self._walk_torch(unary.float(), cands.long(), gate.float(), anchor_ids, return_confidence)


    def _walk_torch(self, unary, cands, gate, anchor_ids, return_confidence):
        pred = anchor_ids
        path = [pred]
        confidence = [torch.zeros_like(pred, dtype = torch.float)]
        for i in range(unary.shape[1]):
            a_emb = F.embedding(pred, self.pred_codebook).float()
            b_emb = F.embedding(cands[:, i], self.succ_codebook).float()
            scores = unary[:, i] + torch.einsum("br,bkr->bk", a_emb * gate[:, i], b_emb)
            score, idx = torch.max(scores, dim = -1)
            pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
            path.append(pred)
            confidence.append(score)
        out = torch.stack(path, dim = 1)
        return out, (torch.stack(confidence, dim = 1) if return_confidence else None)


    def _gather_q(self, q: dict, ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty((ids.numel(), self.rank), dtype = torch.half, device = ids.device)
        ext.dflash2_cb_gather_int(q["q"], q["scales"], q["mins"], ids, out)
        return out


    def _walk_staged(self, unary, cands, gate, anchor_ids, return_confidence):
        # Dequantize every row the walk can touch, two launches before the walk starts: succ rows
        # are the candidates, pred rows the anchor plus the candidates of rows 0..rows-2 (kernel
        # slot 0 = anchor, slot 1 + (i-1)*k + c = row of cands[i-1, c]). The integer dequant folds
        # the per-block scale+min into the rows; the walk applies the gate (and a no-op svh_AB) per
        # position.
        bsz, rows, k = cands.shape
        dev = cands.device
        conf = torch.empty((bsz, rows + 1), dtype = torch.float, device = dev) if return_confidence else None
        ids_a = torch.cat((anchor_ids[:, None], cands[:, :rows - 1, :].reshape(bsz, -1)), dim = 1).reshape(-1)
        stA = self._gather_q(self.pred_q, ids_a).view(bsz, 1 + (rows - 1) * k, self.rank)
        stB = self._gather_q(self.succ_q, cands.reshape(-1)).view(bsz, rows, k, self.rank)
        out = torch.empty((bsz, rows + 1), dtype = torch.long, device = dev)
        ext.dflash2_selector_walk_staged(
            unary, cands, gate.contiguous(), stA, stB, self.svh_AB, anchor_ids, out, conf
        )
        return out, conf


    def _walk_torch_q(self, unary, cands, gate, anchor_ids, return_confidence):
        # torch fallback for quantized codebooks (only reached for top_k outside {8, 16, 32} or
        # odd logit strides); dequantization still goes through the CUDA gather kernel
        bsz, rows, k = cands.shape
        rank = self.rank
        ids_a = torch.cat((anchor_ids[:, None], cands[:, :rows - 1, :].reshape(bsz, -1)), dim = 1).reshape(-1)
        stA = self._gather_q(self.pred_q, ids_a).float().view(bsz, -1, rank)
        stB = self._gather_q(self.succ_q, cands.reshape(-1)).float().view(bsz, rows, k, rank)
        gate = gate.float()
        svh_AB = self.svh_AB.float()
        pred = anchor_ids
        slot = torch.zeros(bsz, dtype = torch.long, device = cands.device)
        path = [pred]
        confidence = [torch.zeros_like(pred, dtype = torch.float)]
        for i in range(rows):
            a = stA.gather(1, slot[:, None, None].expand(-1, 1, rank)).view(bsz, rank)
            w = a * gate[:, i] * svh_AB
            scores = unary[:, i] + torch.einsum("br,bkr->bk", w, stB[:, i])
            score, idx = torch.max(scores, dim = -1)
            pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
            slot = 1 + i * k + idx
            path.append(pred)
            confidence.append(score)
        out = torch.stack(path, dim = 1)
        return out, (torch.stack(confidence, dim = 1) if return_confidence else None)
