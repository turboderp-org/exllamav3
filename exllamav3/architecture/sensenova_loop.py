"""SenseNova Loop inference; shared physical layers use layer_map.

The text path is the Loop forward below.  The optional vision tower is the stock Qwen3.5 one
(Qwen3VLVisionModel); it is only loaded when vision_config is present in config.json."""
import json
import math
import os
import torch
from .qwen3_5 import Qwen3_5VLBaseConfig, Qwen3_5BaseModel
from .qwen3_vl import Qwen3VLVisionModel
from ..modules.transformer import TransformerBlock
from ..modules.gated_delta_net import GatedDeltaNet
from ..modules.loop_router import LoopRouter, merge


def loop_plan(raw):
    loop = raw.get("loop_settings")
    if not isinstance(loop, dict):
        raise ValueError("SenseNova Loop requires loop_settings")
    n = raw["text_config"]["num_hidden_layers"]
    start, end, count = (loop.get(k) for k in ("start", "end", "eval_count"))
    if any(type(v) is not int for v in (n, start, end, count)) or not 0 <= start < end <= n or count < 1 or loop.get("counts") != [count]:
        raise ValueError("Invalid or non-fixed SenseNova Loop execution plan")
    layer_types = raw["text_config"].get("layer_types")
    if (not isinstance(layer_types, list) or len(layer_types) != n
        or set(layer_types) - {"full_attention", "linear_attention"} or "full_attention" not in layer_types):
        raise ValueError("Invalid physical layer types")
    mhc = raw.get("mhc")
    if not isinstance(mhc, dict):
        raise ValueError("SenseNova Loop requires mHC settings")
    a, b, streams, iters = (mhc.get(k) for k in ("start", "end", "streams", "iterations"))
    if (any(type(v) is not int for v in (a, b, streams, iters)) or streams not in (2, 4) or iters < 1
        or not any(l <= a < b <= r for l, r in ((0, start), (start, end), (end, n)))
        or type(mhc.get("fixed_identity")) is not bool):
        raise ValueError("Invalid mHC settings or mHC crosses a loop boundary")
    if set(mhc) - {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}:
        raise ValueError("Unknown mHC settings")
    scale = mhc.get("pre_scale", 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid mHC pre_scale")
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n))


class SenseNovaLoopConfig(Qwen3_5VLBaseConfig):
    arch_string = "SenseNovaLoopForConditionalGeneration"

    def __init__(self, directory, **kwargs):
        # Text-only checkpoints (no vision_config or preprocessor_config.json) skip the vision component.
        with open(os.path.join(directory, "config.json"), encoding = "utf8") as f:
            has_vision = isinstance(json.load(f).get("vision_config"), dict) \
                and os.path.isfile(os.path.join(directory, "preprocessor_config.json"))
        # No MTP head in Loop checkpoints
        super().__init__(
            directory, "text_config", SenseNovaLoopModel, Qwen3VLVisionModel if has_vision else None, None, **kwargs
        )
        plan = loop_plan(self.config_dict)
        if self.layer_map_str is not None or (self.layer_map is not None and self.layer_map != plan):
            raise ValueError("Custom layer maps are not supported for SenseNova Loop")
        self.layer_map = plan
        self.mhc_settings = self.config_dict["mhc"]


class LoopBlock(TransformerBlock):
    def forward(self, x, params, out_dtype = None):
        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0
        # Previous non-mHC blocks may defer their residual addition.
        pending = params.pop("resid_pending", None)
        if pending is not None:
            x = x + pending
        s = self.loop_settings
        if self.layer_idx == s["start"]:
            x = x.unsqueeze(-2).expand(*x.shape[:-1], s["streams"], x.shape[-1]).contiguous()
        else:
            x = x.unflatten(-1, (s["streams"], -1))
        branch, post, mix = self.loop_attn.forward(x, params)
        branch = self.attn_norm.forward(branch, params, out_dtype = torch.half)
        branch = self.attn.forward(branch, params)
        # Prefill stops at the last KV module, layer 31, after the Loop region.
        if params.get("prefill") and not export_state:
            return x.flatten(-2)
        x = merge(x, branch, post, mix)
        branch, post, mix = self.loop_ffn.forward(x, params)
        branch = self.mlp_norm.forward(branch, params, out_dtype = torch.half)
        branch = self.mlp.forward(branch, params)
        x = merge(x, branch, post, mix)
        if export_state:
            states = params.get("export_states")
            if not states:
                states = params["export_states"] = []
            states.append(x.mean(-2).half().clamp_(-65504.0, 65504.0))
        return x.mean(-2) if self.layer_idx == s["end"] - 1 else x.flatten(-2)


class SenseNovaLoopModel(Qwen3_5BaseModel):
    config_class = SenseNovaLoopConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, "model.language_model", False, **kwargs)
        s = config.mhc_settings
        for block in self.modules:
            # HF stores the convolution input in BF16.  Rounding all tokens before
            # convolution also makes the current chunk consistent with cached history.
            if isinstance(block, TransformerBlock) and isinstance(block.attn, GatedDeltaNet):
                block.attn.conv_round_bf16 = True
            if isinstance(block, TransformerBlock) and s["start"] <= block.layer_idx < s["end"]:
                block.__class__ = LoopBlock
                block.loop_settings = s
                block.loop_attn = LoopRouter(config, block.key + ".mhc_attn", s, config.hidden_size)
                block.loop_ffn = LoopRouter(config, block.key + ".mhc_ffn", s, config.hidden_size)
                block.register_submodule(block.loop_attn)
                block.register_submodule(block.loop_ffn)
