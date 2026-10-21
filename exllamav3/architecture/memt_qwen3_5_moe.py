"""
MemT-1.0-Nano (MemTensor): Qwen3.6-35B-A3B backbone + native memory blocks
on the ten full-attention layers. EXPERIMENTAL -- see doc/memt_plan.md.

config.json layout (release): the top level carries ``model_type:
memt_qwen3_5_moe``, ``architectures: [MemTQwen3_5MoeForCausalLM]`` and the
``memt`` memory object; the backbone's own HF config sits under
``backbone_configs`` (a ``qwen3_5_moe`` config with its ``text_config``). The
base ``Config``'s ``config_key`` merges that sub-dict into the top level, after
which the stock Qwen3.5-MoE parser reads the backbone unchanged.

Weights: the backbone tensors use the stock ``model.language_model.*`` keys;
the memory tensors are ``model.language_model.layers.{i}.self_attn.memt.*``
(bf16, carried through conversion unquantized).

Memory state is per session, passed to the forward through ``params`` (see
``modules/arch_specific/memt_memory.py``); the model-level helpers below
create, commit, save and load it in the release's snapshot layout
(safetensors, one fp32 ``memory.{layer}`` tensor of ``[H, D, D]`` per memory
layer).
"""

from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import no_default
from ..modules import TransformerBlock, Attention
from ..modules.arch_specific.memt_memory import MemTMemory
from .qwen3_5 import Qwen3_5VLMoeBaseConfig, Qwen3_5BaseModel


MEMT_SNAPSHOT_FORMAT = "memt-memory-v1"


class MemTQwen3_5MoeConfig(Qwen3_5VLMoeBaseConfig):
    arch_string = "MemTQwen3_5MoeForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            "text_config",
            MemTQwen3_5MoeModel,
            None,
            None,
            config_key = "backbone_configs",
            **kwargs
        )
        self.assert_cfg(str, "model_type", "memt_qwen3_5_moe", True)
        bb_type = self.read_cfg(str, "backbone_configs->model_type", None)
        if bb_type is not None and bb_type != "qwen3_5_moe":
            raise ValueError(f"MemT: unsupported backbone model_type {bb_type!r} (expected qwen3_5_moe)")
        from ..training.memt import parse_memt_block   # lazy: heavy package, see memt_memory.py
        self.memt = parse_memt_block(
            self.read_cfg(dict, "memt", no_default),
            hidden_size = self.hidden_size,
            num_q_heads = self.num_q_heads,
            head_dim = self.head_dim,
            rms_norm_eps = self.rms_norm_eps,
            layer_types = self.layer_types,
        )


class MemTQwen3_5MoeModel(Qwen3_5BaseModel):
    config_class = MemTQwen3_5MoeConfig

    def __init__(
        self,
        config: MemTQwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(
            config = config,
            key_prefix = "model.language_model",
            use_moe = True,
            **kwargs,
        )
        self.memt_layout = config.memt
        self.memt_modules: dict[int, MemTMemory] = {}
        blocks = {m.layer_idx: m for m in self.modules if isinstance(m, TransformerBlock)}
        for idx in self.memt_layout.layers:
            blk = blocks[idx]
            assert isinstance(blk.attn, Attention), \
                f"MemT: layer {idx} is not a full-attention layer"
            mem = MemTMemory(
                config = config,
                key = f"{blk.attn.key}.memt",
                layer_idx = idx,
                spec = self.memt_layout.spec,
            )
            blk.attn.attach_memt(mem)
            self.memt_modules[idx] = mem
        self.caps.update({"memt": True})

    # --- session state helpers ---------------------------------------------

    def memt_zero_state(self, device = "cpu") -> dict[int, torch.Tensor]:
        """A fresh (all-zero) memory for a new trajectory: fp32 ``[1, H, D, D]`` per
        memory layer, the rank the authors' transformers code and snapshot files use."""
        shape = self.memt_layout.spec.state_shape
        return {i: torch.zeros((1, *shape), dtype = torch.float32, device = device)
                for i in self.memt_layout.layers}

    def memt_check_state(self, states: dict[int, torch.Tensor]):
        shape = self.memt_layout.spec.state_shape
        assert set(states.keys()) == set(self.memt_layout.layers), \
            f"MemT state layers {sorted(states)} != model memory layers {list(self.memt_layout.layers)}"
        for i, s in states.items():
            assert s.dtype == torch.float32 and tuple(s.shape[-3:]) == shape, \
                f"MemT state {i}: dtype {s.dtype} shape {tuple(s.shape)}, expected fp32 {shape}"

    @torch.inference_mode()
    def memt_commit(
        self,
        input_ids: torch.Tensor,
        states: dict[int, torch.Tensor],
        params: dict | None = None,
        document: bool = False,
    ) -> dict[int, torch.Tensor]:
        """
        Store text into memory and return the successor state (``states`` is
        left untouched). Mirrors ``commit_memory`` of the authors' transformers
        code (modeling_memt.py):

        * a turn (``document=False``): every token reads the incoming snapshot
          while the write runs through the whole turn ``[1, t]`` (prompt and
          reply as generated); the backbone attends across the whole turn. One
          un-split forward here (the HF code splits into ``chunk_tokens`` events
          but carries the KV cache, which is the same computation).
        * a document (``document=True``): ``chunk_tokens`` (2048) chunks, each a
          separate forward with NO attention across chunks, each reading the
          state the previous chunk wrote.
        """
        self.memt_check_state(states)
        candidate = {i: s.detach().clone() for i, s in states.items()}
        chunk = self.memt_layout.chunk_tokens if document else input_ids.shape[1]
        for start in range(0, input_ids.shape[1], max(chunk, 1)):
            event = input_ids[:, start:start + chunk]
            read = candidate if document else states
            write = {i: s.clone() for i, s in candidate.items()}
            p = dict(params or {})
            p["memt_read"] = read
            p["memt_commit"] = write
            self.forward(event, p)
            candidate = write
        return candidate

    def memt_save_state(self, states: dict[int, torch.Tensor], path: str, identity: str = ""):
        """Write the release's snapshot layout: fp32 ``memory.{layer}`` tensors of
        ``[1, H, D, D]``. The authors' loaders also check an ``identity`` metadata
        string (sha256 over backbone_configs + memt + chat template + vocab, see
        ``MemTSession._state_identity``); pass it to produce a file their code
        accepts, else the file loads here only."""
        from safetensors.torch import save_file
        self.memt_check_state(states)
        tensors = {f"memory.{i}": s.detach().to("cpu", torch.float32).reshape(1, *self.memt_layout.spec.state_shape).contiguous()
                   for i, s in states.items()}
        save_file(tensors, path, metadata = {"format": MEMT_SNAPSHOT_FORMAT, "identity": identity})

    def memt_load_state(self, path: str, device = "cpu") -> dict[int, torch.Tensor]:
        from safetensors import safe_open
        states = {}
        with safe_open(path, framework = "pt", device = "cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format", MEMT_SNAPSHOT_FORMAT) != MEMT_SNAPSHOT_FORMAT:
                raise ValueError(f"MemT snapshot format {meta.get('format')!r} is not {MEMT_SNAPSHOT_FORMAT!r}")
            for k in f.keys():
                if not k.startswith("memory."):
                    raise ValueError(f"MemT snapshot: unexpected tensor {k!r}")
                states[int(k[len("memory."):])] = f.get_tensor(k).to(device, torch.float32)
        if not states:
            return self.memt_zero_state(device)
        self.memt_check_state(states)
        for s in states.values():
            if not torch.isfinite(s).all():
                raise ValueError("MemT snapshot: non-finite state")
        return states
