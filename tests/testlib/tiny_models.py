"""
Tiny random-weight checkpoints written in an architecture's native format, for module- and model-level tests
that need a whole model but no real weights. Each builder writes config.json plus model.safetensors into a
directory and returns it.

DeepSeek-V4 (DSV4_TINY, make_dsv4_checkpoint): DeepSeek's native tensor namespace (the naming of the real
V4-Flash release), exercising all three attention layer types (sliding / CSA / HCA), hash + sqrtsoftplus MoE
routing, mHC, sinks, partial rope with the yarn compress table and the output de-rotation.
"""

import json
import os

import torch
from safetensors.torch import save_file

DSV4_TINY = dict(
    hidden_size = 256,
    num_attention_heads = 8,
    head_dim = 64,
    qk_rope_head_dim = 16,
    q_lora_rank = 128,
    o_groups = 4,
    o_lora_rank = 64,
    sliding_window = 8,
    index_n_heads = 4,
    index_head_dim = 16,
    index_topk = 3,
    num_hidden_layers = 6,
    compress_ratios = [0, 0, 4, 128, 4, 0],
    compress_rate_csa = 4,
    compress_rate_hca = 8,
    hc_mult = 4,
    hc_sinkhorn_iters = 20,
    hc_eps = 1e-6,
    moe_intermediate_size = 64,
    n_routed_experts = 8,
    num_experts_per_tok = 2,
    n_shared_experts = 1,
    num_hash_layers = 2,
    routed_scaling_factor = 1.5,
    swiglu_limit = 10.0,
    rms_norm_eps = 1e-6,
    rope_theta = 10000.0,
    compress_rope_theta = 160000.0,
    vocab_size = 512,
)


def dsv4_config_json(c: dict) -> dict:
    cfg = dict(c)
    cfg.update({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "norm_topk_prob": True,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 0,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 128,
        },
    })
    return cfg


def make_dsv4_checkpoint(out_dir, seed: int = 7, **overrides) -> str:
    """Random-weight DeepSeek-V4 checkpoint in DeepSeek's native tensor names; overrides replace DSV4_TINY keys"""
    torch.manual_seed(seed)
    c = dict(DSV4_TINY, **overrides)
    H = c["hidden_size"]
    hd = c["head_dim"]
    nh = c["num_attention_heads"]
    t = {}

    def w(name, *shape, scale = 0.02, dtype = torch.bfloat16):
        t[name] = (torch.randn(*shape, dtype = torch.float) * scale).to(dtype)

    def normw(name, dim):
        t[name] = (1.0 + 0.1 * torch.randn(dim)).to(torch.bfloat16)

    w("embed.weight", c["vocab_size"], H)
    w("head.weight", c["vocab_size"], H)
    normw("norm.weight", H)
    hc = c["hc_mult"]
    w("hc_head_fn", hc, hc * H, scale = 0.02, dtype = torch.float)
    t["hc_head_base"] = (0.1 * torch.randn(hc)).float()
    t["hc_head_scale"] = (1.0 + 0.1 * torch.randn(1)).float()

    for i, ratio in enumerate(c["compress_ratios"][:c["num_hidden_layers"]]):
        L = f"layers.{i}"
        w(f"{L}.attn.wq_a.weight", c["q_lora_rank"], H)
        normw(f"{L}.attn.q_norm.weight", c["q_lora_rank"])
        w(f"{L}.attn.wq_b.weight", nh * hd, c["q_lora_rank"])
        w(f"{L}.attn.wkv.weight", hd, H)
        normw(f"{L}.attn.kv_norm.weight", hd)
        w(f"{L}.attn.wo_a.weight", c["o_groups"] * c["o_lora_rank"], nh * hd // c["o_groups"])
        w(f"{L}.attn.wo_b.weight", H, c["o_groups"] * c["o_lora_rank"])
        t[f"{L}.attn.attn_sink"] = (0.5 * torch.randn(nh)).float()
        if ratio == 4:      # CSA
            m = c["compress_rate_csa"]
            w(f"{L}.attn.compressor.wkv.weight", 2 * hd, H)
            w(f"{L}.attn.compressor.wgate.weight", 2 * hd, H)
            t[f"{L}.attn.compressor.ape"] = (0.1 * torch.randn(m, 2 * hd)).float()
            normw(f"{L}.attn.compressor.norm.weight", hd)
            di = c["index_head_dim"]
            w(f"{L}.attn.indexer.compressor.wkv.weight", 2 * di, H)
            w(f"{L}.attn.indexer.compressor.wgate.weight", 2 * di, H)
            t[f"{L}.attn.indexer.compressor.ape"] = (0.1 * torch.randn(m, 2 * di)).float()
            normw(f"{L}.attn.indexer.compressor.norm.weight", di)
            w(f"{L}.attn.indexer.wq_b.weight", c["index_n_heads"] * di, c["q_lora_rank"])
            w(f"{L}.attn.indexer.weights_proj.weight", c["index_n_heads"], H)
        elif ratio == 128:  # HCA
            m = c["compress_rate_hca"]
            w(f"{L}.attn.compressor.wkv.weight", hd, H)
            w(f"{L}.attn.compressor.wgate.weight", hd, H)
            t[f"{L}.attn.compressor.ape"] = (0.1 * torch.randn(m, hd)).float()
            normw(f"{L}.attn.compressor.norm.weight", hd)
        normw(f"{L}.attn_norm.weight", H)
        normw(f"{L}.ffn_norm.weight", H)
        for tag in ("attn", "ffn"):
            w(f"{L}.hc_{tag}_fn", (2 + hc) * hc, hc * H, scale = 0.02, dtype = torch.float)
            t[f"{L}.hc_{tag}_base"] = (0.1 * torch.randn((2 + hc) * hc)).float()
            t[f"{L}.hc_{tag}_scale"] = (1.0 + 0.1 * torch.randn(3)).float()
        w(f"{L}.ffn.gate.weight", c["n_routed_experts"], H, scale = 0.05)
        if i < c["num_hash_layers"]:
            t[f"{L}.ffn.gate.tid2eid"] = torch.randint(
                0, c["n_routed_experts"], (c["vocab_size"], c["num_experts_per_tok"]), dtype = torch.long)
        else:
            t[f"{L}.ffn.gate.bias"] = (0.05 * torch.randn(c["n_routed_experts"])).float()
        for e in range(c["n_routed_experts"]):
            w(f"{L}.ffn.experts.{e}.w1.weight", c["moe_intermediate_size"], H, scale = 0.05)
            w(f"{L}.ffn.experts.{e}.w3.weight", c["moe_intermediate_size"], H, scale = 0.05)
            w(f"{L}.ffn.experts.{e}.w2.weight", H, c["moe_intermediate_size"], scale = 0.05)
        w(f"{L}.ffn.shared_experts.w1.weight", c["moe_intermediate_size"], H, scale = 0.05)
        w(f"{L}.ffn.shared_experts.w3.weight", c["moe_intermediate_size"], H, scale = 0.05)
        w(f"{L}.ffn.shared_experts.w2.weight", H, c["moe_intermediate_size"], scale = 0.05)

    os.makedirs(out_dir, exist_ok = True)
    save_file(t, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(dsv4_config_json(c), fp, indent = 2)
    return out_dir
