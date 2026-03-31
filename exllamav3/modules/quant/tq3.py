from __future__ import annotations
import torch
from ...model.config import Config
from ...ext import exllamav3_ext as ext

class LinearTQ3:
    """
    TQ3 (TurboQuant 3-level) quantized linear layer.

    Storage:
      tq3_packed: uint32, shape (in_features // 16, out_features)
        - Stores 2 bitplanes per 32-weight block
        - Row i*2   = nonzero mask for block i
        - Row i*2+1 = positive mask for block i
      tq3_scale: fp16, shape (in_features // 32, out_features)
        - Per-block scale factor
      suh: fp16, shape (in_features,) — Hadamard pre-rotation signs (optional)
      svh: fp16, shape (out_features,) — Hadamard post-rotation signs (optional)
      bias: fp16, shape (out_features,) — optional

    Strategy A implementation: dequant to fp16 weight matrix, then standard matmul.
    """

    quant_type: str = "tq3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        tq3_packed: torch.Tensor,         # uint32, (in_features // 16, out_features)
        tq3_scale: torch.Tensor,          # fp16,   (in_features // 32, out_features)
        suh: torch.Tensor | None = None,  # fp16, (in_features,) — pre-rotation signs
        svh: torch.Tensor | None = None,  # fp16, (out_features,) — post-rotation signs
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        key: str | None = None
    ):
        assert tq3_packed is not None, "tq3_packed is required"
        assert tq3_scale is not None, "tq3_scale is required"
        assert tq3_packed.dtype == torch.int32 or tq3_packed.dtype == torch.int, \
            f"tq3_packed must be int32, got {tq3_packed.dtype}"
        assert tq3_scale.dtype == torch.half, "tq3_scale must be fp16"

        if bias is not None and bias.dtype == torch.float:
            bias = bias.to(torch.half)

        self.key = key
        self.in_features = in_features
        self.out_features = out_features
        self.tq3_packed = tq3_packed
        self.tq3_scale = tq3_scale
        self.suh = suh
        self.svh = svh
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype

        # Pre-compute dequantized weight for Strategy A
        self._weight_cache = None


    def unload(self):
        self._weight_cache = None


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("tq3_packed", self.tq3_packed),
                ("tq3_scale", self.tq3_scale),
                ("suh", self.suh),
                ("svh", self.svh),
                ("bias", self.bias),
            ] if tensor is not None
        }


    def _dequant_weight(self) -> torch.Tensor:
        """
        Dequantize TQ3 packed weight to fp16 matrix.

        For each block of 32 input features:
          1. Read 2 bitplanes (nonzero mask, positive mask)
          2. Reconstruct ternary: val = nonzero * (2*positive - 1)
          3. Multiply by block scale
          4. Apply inverse WHT (Hadamard rotation)
        """
        if self._weight_cache is not None:
            return self._weight_cache

        num_blocks = self.in_features // 32
        device = self.tq3_packed.device

        # Unpack bitplanes -> ternary values
        # tq3_packed shape: (num_blocks * 2, out_features)
        bp0 = self.tq3_packed[0::2, :]  # nonzero masks, shape (num_blocks, out_features)
        bp1 = self.tq3_packed[1::2, :]  # positive masks, shape (num_blocks, out_features)

        # Expand bitplanes to per-element ternary values
        # Each uint32 encodes 32 values
        bit_indices = torch.arange(32, device=device).view(1, 32, 1)  # (1, 32, 1)

        bp0_exp = bp0.unsqueeze(1)  # (num_blocks, 1, out_features)
        bp1_exp = bp1.unsqueeze(1)  # (num_blocks, 1, out_features)

        nonzero = ((bp0_exp >> bit_indices) & 1).to(torch.float16)  # (num_blocks, 32, out_features)
        positive = ((bp1_exp >> bit_indices) & 1).to(torch.float16)

        # Ternary value: nonzero * (2*positive - 1), but where nonzero=0 we want 0
        # So: nonzero * (2*positive - 1) = nonzero * 2 * positive - nonzero
        ternary = nonzero * (2.0 * positive - 1.0)  # {-1, 0, +1}

        # Apply per-block scale
        scales = self.tq3_scale.unsqueeze(1)  # (num_blocks, 1, out_features)
        ternary = ternary * scales

        # Reshape to full weight matrix: (in_features, out_features)
        w = ternary.reshape(self.in_features, self.out_features)

        # Apply Hadamard rotations if present
        if self.suh is not None:
            w = w * self.suh.unsqueeze(1)
            # Apply 128-element Hadamard blocks along input dimension
            w = self._apply_had_rows(w, 128)

        if self.svh is not None:
            w = w * self.svh.unsqueeze(0)
            # Apply 128-element Hadamard blocks along output dimension
            w = self._apply_had_cols(w, 128)

        self._weight_cache = w
        return w


    @staticmethod
    def _apply_had_rows(w: torch.Tensor, block_size: int) -> torch.Tensor:
        """Apply block-diagonal Hadamard transform along rows (input dim)."""
        rows, cols = w.shape
        assert rows % block_size == 0
        w = w.view(rows // block_size, block_size, cols).float()
        # Use the recursive Hadamard (Walsh-Hadamard) via butterfly operations
        n = block_size
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = w[:, j, :].clone()
                    y = w[:, j + h, :].clone()
                    w[:, j, :] = x + y
                    w[:, j + h, :] = x - y
            h *= 2
        w = w / (block_size ** 0.5)
        return w.reshape(rows, cols).half()


    @staticmethod
    def _apply_had_cols(w: torch.Tensor, block_size: int) -> torch.Tensor:
        """Apply block-diagonal Hadamard transform along columns (output dim)."""
        return LinearTQ3._apply_had_rows(w.T, block_size).T


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Strategy A: Dequantize weight, then standard matmul.
        """
        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, self.in_features)
        dtype = out_dtype or self.out_dtype or torch.half

        w = self._dequant_weight()  # (in_features, out_features), cached

        y = torch.empty(
            (x.shape[0], self.out_features),
            dtype = dtype,
            device = x.device
        )

        if dtype == x.dtype:
            torch.matmul(x, w, out = y)
        else:
            ext.hgemm(x, w, y)

        if self.bias is not None:
            y += self.bias

        return y.view(out_shape)


    def get_weight_tensor(self) -> torch.Tensor:
        return self._dequant_weight()


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.tq3_packed.device
        self._weight_cache = None
        self.tq3_packed = self.tq3_packed.cpu()
        self.tq3_scale = self.tq3_scale.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        self.tq3_packed = self.tq3_packed.to(self.swap_device)
        self.tq3_scale = self.tq3_scale.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearTQ3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "tq3_packed": producer.send(self.tq3_packed),
            "tq3_scale": producer.send(self.tq3_scale),
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "bias": producer.send(self.bias),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(exported["suh"], cuda = True)
            svh = consumer.recv(exported["svh"], cuda = True, slice_dim = 0, first = first, last = last)
            tq3_packed = consumer.recv(exported["tq3_packed"], cuda = True, slice_dim = 1, first = first, last = last)
            tq3_scale = consumer.recv(exported["tq3_scale"], cuda = True, slice_dim = 1, first = first, last = last)
            bias = consumer.recv(exported["bias"], cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            # Input splitting: need to slice packed/scale along rows
            suh = consumer.recv(exported["suh"], cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(exported["svh"], cuda = True)
            # packed has 2 rows per 32 input features
            p_first = (first // 32) * 2
            p_last = (last // 32) * 2
            tq3_packed = consumer.recv(exported["tq3_packed"], cuda = True, slice_dim = 0, first = p_first, last = p_last)
            s_first = first // 32
            s_last = last // 32
            tq3_scale = consumer.recv(exported["tq3_scale"], cuda = True, slice_dim = 0, first = s_first, last = s_last)
            bias = consumer.recv(exported["bias"], cuda = True) if (first == 0) else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearTQ3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            tq3_packed = tq3_packed,
            tq3_scale = tq3_scale,
            suh = suh,
            svh = svh,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
