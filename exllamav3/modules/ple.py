from __future__ import annotations
from typing_extensions import override
import math
import torch
import torch.nn.functional as F
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from .ngram_embedding import NGramEmbedding
from ..ext import exllamav3_ext as ext
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.config import Config
from ..util.tensor import get_for_device

"""
PLE (per-layer embedding) injection layer (Qwen3.8-Flash-Next): feeds hashed n-gram features into
every residual stream of the hyper-connection stack.

The n-gram embedding (2560) projects to one key per stream (10240) and one shared value (2560).
Each stream's normed activation gates the value through a signed-sqrt dot-product gate, and a
depthwise DILATED causal conv (kernel ple_conv_kernel_size, dilation ngram_size) over the normed
gated values adds local lexical context. The layer's output is added to the raw stream stack by
the caller:  streams <- streams + ple(streams, token_history).

Recurrent state per sequence: the (kernel-1)*dilation trailing positions of the normed gated
values (conv_state), plus the ngram_size-1 previous token ids (carried in token_history by the
caller). Norms are grouped RMS (rms per 2560-stream, zero-init weight applied as 1 + w).
"""


class PLELayerState:
    """
    Per-slot recurrent state for one PLE layer, following the ShortConvLayerState conventions:
    the live state occupies the leading window columns, history forwards (speculative decoding)
    write right-aligned trailing columns, and rewind restores the window from them. Two tensors:
    the dilated-conv state (trailing normed gated values) and the carried token-id context for
    the n-gram hashing (eos-filled at sequence start).
    """

    def __init__(
        self,
        module: PLELayer,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        self.module = module
        self.win = module.conv_state_len
        self.ctx = module.ple_embedding.context_len
        hc_hidden = module.hc_mult * module.hidden_size
        self.conv_state = torch.empty(
            (max_batch_size, hc_hidden, self.win + max_history),
            dtype = torch.half,
            device = "meta",
        )
        self.id_state = torch.empty(
            (max_batch_size, self.ctx + max_history),
            dtype = torch.long,
            device = "meta",
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id

    def get_checkpoint_size(self):
        return self.conv_state.shape[1] * self.win * 2 + self.ctx * 8

    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in (self.conv_state, self.id_state))

    def alloc(self, device):
        self.conv_state = torch.zeros_like(self.conv_state, device = device)
        # The carried token-id context stays on the CPU: the n-gram hashing runs host-side on
        # the (pinned) input ids, so the id history must never round-trip through the device
        self.id_state = torch.full_like(self.id_state, self.module.ple_embedding.eos_token_id,
                                        device = "cpu")
        self.device = device

    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.id_state = torch.empty_like(self.id_state, device = "meta")
        self.device = None

    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()
            self.id_state[idx].fill_(self.module.ple_embedding.eos_token_id)

    def get_state_tensors(self):
        return self.conv_state, self.id_state

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens <= last_history
        if last_history > 0:
            p = self.conv_state.shape[-1] - num_tokens
            temp = self.conv_state[slot, :, p - self.win : p].clone()
            self.conv_state[slot, :, :self.win].copy_(temp)
            p = self.id_state.shape[-1] - num_tokens
            temp = self.id_state[slot, p - self.ctx : p].clone()
            self.id_state[slot, :self.ctx].copy_(temp)

    def stash(self, slot, position: int = 0):
        return (self.conv_state[slot, :, :self.win].cpu(), self.id_state[slot, :self.ctx].cpu())

    def unstash(self, slot, stashed, position: int = 0):
        self.conv_state[slot, :, :self.win].copy_(stashed[0])
        self.id_state[slot, :self.ctx].copy_(stashed[1])

    def tp_export(self, plan):
        return {
            "cls": PLELayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


class PLELayer(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        hc_mult: int,
        ple_embed_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        eos_token_id: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        qmap: str | None = None,
        stream_from_disk: bool | None = None,
        out_dtype: torch.dtype | None = None,
        mm_token_id: int | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.mm_token_id = mm_token_id
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation = ngram_size
        self.conv_state_len = (conv_kernel_size - 1) * self.conv_dilation
        self.gate_scale = 1.0 / math.sqrt(hidden_size)
        self.out_dtype = out_dtype
        hc_hidden = hc_mult * hidden_size

        self.ple_embedding = NGramEmbedding(
            config = config,
            key = f"{key}.ple_embedding.ngram_embedding",
            ngram_size = ngram_size,
            heads_per_ngram = heads_per_ngram,
            ple_embed_dim = ple_embed_dim,
            eos_token_id = eos_token_id,
            stream_from_disk = stream_from_disk,
        )
        self.key_proj = Linear(
            config = config,
            key = f"{key}.key_proj",
            in_features = ple_embed_dim,
            out_features = hc_hidden,
            qmap = qmap,
            out_dtype = torch.half,
        )
        self.value_proj = Linear(
            config = config,
            key = f"{key}.value_proj",
            in_features = ple_embed_dim,
            out_features = hidden_size,
            qmap = qmap,
            out_dtype = torch.half,
        )
        # Grouped RMS norms over the stream stack (weight is hc_mult rows of hidden channels,
        # zero-init, applied as 1 + w)
        def norm(name):
            return RMSNorm(config, f"{key}.{name}", rms_norm_eps, constant_bias = 1.0,
                           groups = hc_mult)
        self.norm_key = norm("norm_key")
        self.norm_query = norm("norm_query")
        self.norm_conv = norm("norm_conv")
        self.register_submodule(self.ple_embedding)
        self.register_submodule(self.key_proj)
        self.register_submodule(self.value_proj)
        self.register_submodule(self.norm_key)
        self.register_submodule(self.norm_query)
        self.register_submodule(self.norm_conv)

        # Recurrent state registration: negative layer_idx keeps the state key distinct from the
        # decoder block that shares this layer index in the cache's recurrent-layer map
        assert layer_idx < 0
        self.layer_idx = layer_idx
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = PLELayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}

        self.conv_w = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        for rl in self.recurrent_layers:
            rl.alloc(device)
        self.conv_w = self.config.stc.get_tensor(f"{self.key}.conv1d.weight", device,
                                                 float2half = True, no_defer = True).contiguous()

    @override
    def unload(self):
        for rl in self.recurrent_layers:
            rl.free()
        super().unload()
        self.conv_w = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.conv1d.weight": self.conv_w.contiguous(),
        }

    @override
    def weights_numel(self):
        # Everything this module contributes to the converted shards: the (fp16) projections,
        # the three grouped norm weights and the conv kernel — but NOT the n-gram embedding,
        # whose table lives out-of-line in ngram_embedding.safetensors and accounts for itself.
        # Shape-derived, since conversion reads the count after unload()
        hc_hidden = self.hc_mult * self.hidden_size
        return self.key_proj.weights_numel() + self.value_proj.weights_numel() \
            + 3 * hc_hidden + hc_hidden * self.conv_kernel_size

    @override
    def optimizer_targets(self):
        return self.key_proj.optimizer_targets() + self.value_proj.optimizer_targets()

    def _short_conv(self, x: torch.Tensor, conv_state: torch.Tensor | None):
        """
        x: (bsz, seq, hc_hidden) normed gated values. conv_state: (bsz, hc_hidden,
        conv_state_len) trailing positions from the previous chunk, or None at sequence start.
        Returns (silu(conv(x)) (bsz, seq, hc_hidden), the full padded column stream
        (bsz, hc_hidden, conv_state_len + seq) — its trailing conv_state_len columns are the
        next chunk's conv state; the full stream supports history writes for rewind).
        """
        bsz, seq, ch = x.shape
        xt = x.transpose(1, 2)                                       # (bsz, ch, seq)
        if conv_state is None:
            conv_state = xt.new_zeros((bsz, ch, self.conv_state_len))
        xt = torch.cat((conv_state.to(xt.dtype), xt), dim = -1)      # (bsz, ch, state + seq)
        y = F.conv1d(xt, self.conv_w, groups = ch, dilation = self.conv_dilation)
        return F.silu(y).transpose(1, 2), xt

    def forward_streams(
        self,
        streams: torch.Tensor,
        token_history: torch.Tensor,
        params: dict,
        conv_state: torch.Tensor | None = None,
    ):
        """
        streams: (bsz, seq, hc_mult, hidden) fp32 residual stack;
        token_history: (bsz, (ngram_size - 1) + seq) token ids (eos-padded at sequence start).
        Returns (delta (bsz, seq, hc_mult, hidden) fp32 to add to the streams, full conv column
        stream — see _short_conv). The layer sits at the front of the forward pass and was
        host-bound issued op-by-op from python; the whole sequence runs as one ext call
        (ple_forward_streams), with the op-by-op form kept below as the reference.
        """
        bsz, seq = streams.shape[:2]
        H, D = self.hc_mult, self.hidden_size
        emb = self.ple_embedding.forward(token_history, params)               # (bsz, seq, ple_dim)
        if self.key_proj.quant_type == "fp16" and self.value_proj.quant_type == "fp16" \
                and streams.dtype == torch.float and streams.is_contiguous() \
                and self.key_proj.inner.bias is None and self.value_proj.inner.bias is None:
            delta = torch.empty_like(streams)
            conv_stream = torch.empty((bsz, H * D, self.conv_state_len + seq),
                                      dtype = torch.half, device = streams.device)
            ext.ple_forward_streams(
                streams, emb.contiguous(),
                self.key_proj.inner.weight, self.value_proj.inner.weight,
                self.norm_key.weight.data, self.norm_query.weight.data,
                self.norm_conv.weight.data, self.conv_w,
                conv_state.contiguous() if conv_state is not None else None,
                self.norm_key.rms_norm_eps, self.gate_scale, self.conv_dilation,
                delta, conv_stream,
            )
            return delta, conv_stream
        return self.forward_streams_reference(streams, emb, params, conv_state)

    def forward_streams_reference(self, streams, emb, params, conv_state = None):
        """Op-by-op form of forward_streams (torch + individual ext kernels)."""
        bsz, seq = streams.shape[:2]
        H, D = self.hc_mult, self.hidden_size
        key = self.key_proj.forward(emb, params).view(bsz, seq, H, D)
        key = self.norm_key.forward(key, params, out_dtype = torch.float)
        value = self.value_proj.forward(emb, params)                          # (bsz, seq, hidden) fp16
        query = self.norm_query.forward(streams, params, out_dtype = torch.float)
        # per-stream key/query dots as a batched (1, D) x (D, 1) matmul, then the fused gate
        # kernel: gated = sigmoid(signed_sqrt(dot * scale)) * value broadcast over streams
        gate = torch.bmm(query.view(-1, 1, D), key.reshape(-1, D, 1)).view(bsz, seq, H)
        gated = torch.empty((bsz, seq, H, D), dtype = torch.float, device = value.device)
        ext.ple_gate(gate, value, gated, self.gate_scale)
        normed = self.norm_conv.forward(gated, params, out_dtype = torch.half).flatten(-2)
        conv_out, conv_stream = self._short_conv(normed, conv_state)
        delta = gated + conv_out.view(bsz, seq, H, D)
        return delta, conv_stream

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        """
        Standalone-module form: x is the (bsz, seq, hc_mult, hidden) stream stack; token ids come
        from params["input_ids"] (put there by the Embedding module). With recurrent states in
        params, the carried token context and conv state make forwards incremental (same
        conventions as ShortConv: state window in [:, ..., :width], history writes right-aligned
        for rewind).
        """
        bsz, seq = x.shape[:2]
        ids = params.get("input_ids")
        if ids is None:
            # autosplit measuring forward: no token ids in params; a dummy history gives the
            # same memory profile (the real n-gram gather is a few hundred rows at most)
            ids = torch.zeros((bsz, seq), dtype = torch.long)
        # the hashing runs host-side; in the hot path the generator's ids are already CPU
        ids = ids.to("cpu", torch.int64)
        if self.mm_token_id is not None and int(ids.max()) >= FIRST_MM_EMBEDDING_INDEX:
            # multimodal spans carry embedding alias ids; the reference hashes the literal
            # placeholder token there, so substitute it (the carried id context then matches
            # what HF's would hold as well)
            ids = ids.clone()
            ids[ids >= FIRST_MM_EMBEDDING_INDEX] = self.mm_token_id
        ctx = self.ple_embedding.context_len
        win = self.conv_state_len
        save_history = params.get("recurrent_history", False)

        rsg = params.get("recurrent_states")
        if rsg:
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, id_state = rsl.get_state_tensors()
            slots = get_for_device(params, "recurrent_slots", "cpu").tolist()
            assert len(slots) == bsz
            prev_ids = torch.stack([id_state[s, :ctx] for s in slots])
            history = torch.cat((prev_ids, ids), dim = 1)
            cs = torch.stack([conv_state[s, :, :win] for s in slots]).to(x.device)
            delta, conv_stream = self.forward_streams(x, history, params, conv_state = cs)
            for i, s in enumerate(slots):
                if save_history:
                    w = min(conv_state.shape[-1], conv_stream.shape[-1])
                    conv_state[s, :, -w:].copy_(conv_stream[i, :, -w:])
                    wi = min(id_state.shape[-1], history.shape[-1])
                    id_state[s, -wi:].copy_(history[i, -wi:])
                else:
                    conv_state[s, :, :win].copy_(conv_stream[i, :, -win:])
                    id_state[s, :ctx].copy_(history[i, -ctx:])
        else:
            assert params.get("position", 0) == 0, \
                "PLELayer requires recurrent states for forwards past position 0"
            pad = ids.new_full((bsz, ctx), self.ple_embedding.eos_token_id)
            delta, _ = self.forward_streams(x, torch.cat((pad, ids), dim = 1), params)
        return x + delta
