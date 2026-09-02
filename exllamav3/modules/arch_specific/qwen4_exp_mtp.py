from __future__ import annotations
from typing_extensions import override
import torch
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm
from ...util.tensor import get_for_device, to2

"""
MTP input layer for Qwen3.8-Flash-Next.

The trunk tap is the PRE-collapse hyper-connection stream stack (pre_fc_norm_hidden is
hc_mult * hidden wide), exported by the trunk's final mixer. Combine, per draft token:

    streams_i = fc_hidden(grouped_norm(stack)_i) + fc_embedding(norm(embed(token)))

i.e. the shared fc_hidden applies per stream and the embedding branch broadcasts into every
stream, so the MTP block's own hyper connections start from the trunk's stream identities
(stream_tap = True). With stream_tap = False the normed streams are averaged first and the
combined state is broadcast instead (trunk-entry style). There is no reference implementation
for this head; the choice is empirical and must be confirmed by acceptance rate on the full
model (cf. the hy3 post-norm-tap finding).
"""


class Qwen4ExpMTPInputLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        stream_tap: bool = True,
        out_dtype: torch.dtype | None = torch.float,
        qbits_key = "mtp_bits",
    ):
        super().__init__(config, key, None)
        self.module_name = "Qwen4ExpMTPInputLayer"
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.rms_eps = rms_norm_eps
        self.stream_tap = stream_tap
        self.out_dtype = out_dtype

        self.pre_fc_norm_embedding = RMSNorm(
            config = config,
            key = f"{key}.pre_fc_norm_embedding",
            rms_norm_eps = rms_norm_eps,
            constant_bias = 1.0,
            out_dtype = torch.half,
        )
        self.fc_hidden = Linear(
            config = config,
            key = f"{key}.fc_hidden",
            in_features = hidden_size,
            out_features = hidden_size,
            qmap = "block.mtp.fc",
            out_dtype = torch.half,
            qbits_key = qbits_key,
            select_hq_bits = 1,
        )
        self.fc_embedding = Linear(
            config = config,
            key = f"{key}.fc_embedding",
            in_features = hidden_size,
            out_features = hidden_size,
            qmap = "block.mtp.fc",
            out_dtype = torch.half,
            qbits_key = qbits_key,
            select_hq_bits = 1,
        )
        self.register_submodule(self.pre_fc_norm_embedding)
        self.register_submodule(self.fc_hidden)
        self.register_submodule(self.fc_embedding)

        self.norm_hidden_w_raw = None
        self.norm_hidden_w = None

        # Populated by attach_to()
        self.attached_model = None

        self.caps.update({"x_cpu": True})

    @override
    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        w = self.config.stc.get_tensor(f"{self.key}.pre_fc_norm_hidden.weight", device, no_defer = True)
        self.norm_hidden_w_raw = w
        self.norm_hidden_w = (w.float() + 1.0).view(self.hc_mult, self.hidden_size).contiguous()

    @override
    def unload(self):
        super().unload()
        self.norm_hidden_w_raw = self.norm_hidden_w = None

    @override
    def get_tensors(self):
        return {f"{self.key}.pre_fc_norm_hidden.weight": self.norm_hidden_w_raw.contiguous()}

    @override
    def weights_numel(self):
        return self.hc_mult * self.hidden_size

    # The module key is the bare "mtp" prefix, which would swallow every mtp.* tensor in the
    # default prefix-based compile collection (duplicating the decoder block's and mixer's
    # tensors); enumerate this module's own tensors explicitly instead
    def _compile_keys(self):
        return [
            f"{self.key}.pre_fc_norm_hidden",
            f"{self.key}.pre_fc_norm_embedding",
            f"{self.key}.fc_hidden",
            f"{self.key}.fc_embedding",
        ]

    @override
    def get_compile_sizes(self, stc):
        return [s for k in self._compile_keys() for s in stc.get_tensor_sizes(k)]

    @override
    def get_compile_tensors(self, stc):
        tensors = {}
        for k in self._compile_keys():
            tensors.update(stc.get_tensors(k, allow_bf16 = True))
        return tensors

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None, "Qwen4Exp MTP requires target_hidden"
        assert target_hidden.shape[:-1] == x.shape, \
            f"Qwen4Exp MTP token/state shape mismatch: {tuple(x.shape)} vs {tuple(target_hidden.shape)}"
        bsz, seq = x.shape
        y = get_for_device(params, "target_hidden", self.device)
        stack = y.view(bsz, seq, self.hc_mult, self.hidden_size).float()
        normed = stack * torch.rsqrt(stack.pow(2).mean(-1, keepdim = True) + self.rms_eps) \
            * self.norm_hidden_w
        h = self.fc_hidden.forward(normed.half().contiguous(), params)         # (b, s, H, D)

        # Token embedding via the attached model
        emb = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        emb = self.pre_fc_norm_embedding.forward(emb.to(self.device), params)
        emb = self.fc_embedding.forward(emb, params)                           # (b, s, D)

        if self.stream_tap:
            streams = h.float() + emb.float().unsqueeze(-2)
        else:
            streams = (h.float().mean(dim = -2) + emb.float()) \
                .unsqueeze(-2).expand(-1, -1, self.hc_mult, -1)
        return to2(streams.contiguous(), out_dtype, self.out_dtype)
