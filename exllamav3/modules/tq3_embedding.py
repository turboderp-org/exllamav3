from __future__ import annotations
import torch


class TQ3Embedding:
    """
    TQ3-compressed embedding table.

    Stores the embedding matrix in TQ3 format (2.5 bits/value) and
    dequantizes only the requested rows on lookup.

    Memory savings: 128K vocab x 4096 hidden = ~1GB fp16 -> ~160MB TQ3 (6.4x)

    Storage:
      tq3_packed: int32, shape (vocab_size // 16, hidden_size)
        - 2 bitplanes per 32-row block (interleaved)
        - Row 2*i   = nonzero mask for block i
        - Row 2*i+1 = positive mask for block i
      tq3_scale: fp16, shape (vocab_size // 32, hidden_size)
        - Per-block scale factor
    """

    def __init__(
        self,
        tq3_packed: torch.Tensor,   # (vocab_size // 16, hidden_size), int32
        tq3_scale: torch.Tensor,    # (vocab_size // 32, hidden_size), fp16
        vocab_size: int,
        hidden_size: int,
    ):
        self.tq3_packed = tq3_packed
        self.tq3_scale = tq3_scale
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


    @staticmethod
    def from_weight(weight: torch.Tensor) -> TQ3Embedding:
        """
        Create TQ3Embedding from a full fp16 embedding weight matrix.

        Args:
            weight: (vocab_size, hidden_size) fp16 tensor (already padded to multiple of 32)
        """
        vocab_size, hidden_size = weight.shape
        assert vocab_size % 32 == 0, f"vocab_size must be multiple of 32, got {vocab_size}"

        device = weight.device
        num_blocks = vocab_size // 32

        # Reshape to blocks of 32 rows
        w_blocks = weight.float().view(num_blocks, 32, hidden_size)

        # Per-block scale (max abs per 32-row block, per column)
        scales = w_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
        scales_flat = scales.squeeze(1)  # (num_blocks, hidden_size)

        # Normalize
        w_norm = w_blocks / scales

        # Lloyd-Max ternary: boundary at +/- 0.5
        nonzero = (w_norm.abs() >= 0.5).int()
        positive = ((w_norm > 0) & (nonzero == 1)).int()

        # Pack bitplanes
        bit_indices = torch.arange(32, device=device).view(1, 32, 1)
        bp0 = (nonzero << bit_indices).sum(dim=1).to(torch.int32)
        bp1 = (positive << bit_indices).sum(dim=1).to(torch.int32)

        tq3_packed = torch.zeros(num_blocks * 2, hidden_size, dtype=torch.int32, device=device)
        tq3_packed[0::2] = bp0
        tq3_packed[1::2] = bp1

        return TQ3Embedding(
            tq3_packed=tq3_packed,
            tq3_scale=scales_flat.half(),
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for input_ids, dequantizing on-the-fly.

        Only decompresses the rows that are actually needed,
        not the entire embedding table.

        Args:
            input_ids: (batch_size, seq_len) or (seq_len,) int tensor

        Returns:
            embeddings: (..., hidden_size) fp16 tensor
        """
        orig_shape = input_ids.shape
        ids_flat = input_ids.view(-1)  # (N,)

        # Determine which blocks each id belongs to
        block_ids = ids_flat // 32        # which 32-row block
        bit_positions = ids_flat % 32     # position within block

        # Gather the relevant bitplanes and scales
        bp0 = self.tq3_packed[block_ids * 2]      # nonzero masks, (N, hidden_size)
        bp1 = self.tq3_packed[block_ids * 2 + 1]  # positive masks, (N, hidden_size)
        scales = self.tq3_scale[block_ids]         # (N, hidden_size)

        # Extract the specific bit for each id
        bit_mask = (1 << bit_positions).unsqueeze(1).to(torch.int32)  # (N, 1)

        nonzero = ((bp0 & bit_mask) != 0).to(torch.float16)  # (N, hidden_size)
        positive = ((bp1 & bit_mask) != 0).to(torch.float16)

        # Reconstruct: val = nonzero * (2*positive - 1) * scale
        ternary = nonzero * (2.0 * positive - 1.0)
        embeddings = ternary * scales

        return embeddings.view(*orig_shape, self.hidden_size)


    def storage_size(self) -> int:
        """Return storage size in bytes."""
        return (
            self.tq3_packed.numel() * 4 +  # int32
            self.tq3_scale.numel() * 2      # fp16
        )


    def to(self, device):
        self.tq3_packed = self.tq3_packed.to(device)
        self.tq3_scale = self.tq3_scale.to(device)
        return self
