"""
Tiny random-weight checkpoints written in an architecture's native format, for module- and model-level tests
that need a whole model but no real weights. Each builder writes config.json plus model.safetensors into a
directory and returns it.

Qwen4Exp (QWEN4_TINY, make_qwen4_checkpoint): four decoder blocks with GDN, QSA, hyperconnections, a small PLE table
and a byte-level tokenizer, for cached inference tests without downloaded weights.

DeepSeek-V4 (DSV4_TINY, make_dsv4_checkpoint): DeepSeek's native tensor namespace (the naming of the real
V4-Flash release), exercising all three attention layer types (sliding / CSA / HCA), hash + sqrtsoftplus MoE
routing, mHC, sinks, partial rope with the yarn compress table and the output de-rotation.
"""

import json
import os

import torch
from safetensors.torch import save_file


QWEN4_TINY = dict(
    hidden_size = 256,
    num_hidden_layers = 4,
    vocab_size = 512,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 64,
    full_attention_interval = 2,
    linear_num_key_heads = 2,
    linear_num_value_heads = 4,
    linear_key_head_dim = 64,
    linear_value_head_dim = 64,
    linear_conv_kernel_dim = 4,
    moe_intermediate_size = 128,
    shared_expert_intermediate_size = 128,
    num_experts = 4,
    num_experts_per_tok = 2,
    rms_norm_eps = 1e-6,
    hc_count = 4,
    indexer_n_heads = 3,
    indexer_head_dim = 64,
    indexer_budget = 16,
    ple_layer_ids = [2],
    ple_embed_dim = 256,
    ngram_size = 2,
    heads_per_ngram = 1,
    ple_conv_kernel_size = 4,
    eos_token_id = 2,
    bos_token_id = 1,
    max_position_embeddings = 2048,
    tie_word_embeddings = False,
)


def make_qwen4_checkpoint(out_dir, seed: int = 7) -> str:
    """Four small Qwen4Exp blocks, including PLE, GDN, QSA and real hyperconnection weights."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    c = dict(QWEN4_TINY)
    rng = torch.Generator().manual_seed(seed)
    tensors = {}

    def w(key, *shape, scale = 0.02):
        tensors[key] = (torch.randn(shape, generator = rng) * scale).half()

    def zero(key, *shape):
        tensors[key] = torch.zeros(shape, dtype = torch.half)

    def hc(key, combine = True):
        zero(key + ".hc_norm.weight", 1024)
        w(key + ".input_mix_weight_down.weight", 64, 1024)
        w(key + ".input_mix_weight_up.weight", 1024, 64)
        if combine: w(key + ".block_inject_weight.weight", 4, 1024)

    prefix = "model.language_model"
    w(prefix + ".embed_tokens.weight", 512, 256)
    w("lm_head.weight", 512, 256)
    hc(prefix + ".hyper_connection_mixer", False)
    for i in range(4):
        block = f"{prefix}.layers.{i}"
        hc(block + ".attn_hyper_connection")
        hc(block + ".mlp_hyper_connection")
        if i % 2 == 0:
            attn = block + ".linear_attn"
            for name, rows in [("in_proj_qkv", 512), ("in_proj_z", 256), ("in_proj_b", 4), ("in_proj_a", 4)]:
                w(f"{attn}.{name}.weight", rows, 256)
            w(attn + ".out_proj.weight", 256, 256)
            w(attn + ".conv1d.weight", 512, 1, 4, scale = 0.2)
            tensors[attn + ".A_log"] = torch.zeros(4, dtype = torch.float)
            tensors[attn + ".dt_bias"] = torch.full((4,), -2.0, dtype = torch.bfloat16)
            tensors[attn + ".norm.weight"] = torch.ones(64, dtype = torch.bfloat16)
        else:
            attn = block + ".self_attn"
            for name, rows in [("q_proj", 512), ("k_proj", 128), ("v_proj", 128), ("o_proj", 256)]:
                w(f"{attn}.{name}.weight", rows, 256)
            zero(attn + ".q_norm.weight", 64)
            zero(attn + ".k_norm.weight", 64)
            w(attn + ".indexer.index_qk_proj.weight", 256, 256)
            zero(attn + ".indexer.q_layernorm.weight", 64)
            zero(attn + ".indexer.k_layernorm.weight", 64)
        w(block + ".mlp.gate.weight", 4, 256)
        w(block + ".mlp.shared_expert_gate.weight", 1, 256)
        for expert in [f"experts.{e}" for e in range(4)] + ["shared_expert"]:
            for name, shape in [("up_proj", (128, 256)), ("gate_proj", (128, 256)), ("down_proj", (256, 128))]:
                w(f"{block}.mlp.{expert}.{name}.weight", *shape)
    ple = prefix + ".layers.1.ple"
    w(ple + ".ple_embedding.ngram_embedding.weight", 257, 256)
    tensors[ple + ".ple_embedding.ngram_heads_offsets"] = torch.tensor([0])
    tensors[ple + ".ple_embedding.ngram_heads_vocab_sizes"] = torch.tensor([257])
    tensors[ple + ".ple_embedding.layer_multipliers"] = torch.tensor([1, 31])
    w(ple + ".key_proj.weight", 1024, 256)
    w(ple + ".value_proj.weight", 256, 256)
    for name in ["norm_key", "norm_query", "norm_conv"]: zero(f"{ple}.{name}.weight", 4, 256)
    w(ple + ".conv1d.weight", 1024, 1, 4)

    os.makedirs(out_dir, exist_ok = True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": c}, fp)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({token: i + 3 for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    vocab.update({f"<unused{i}>": i for i in range(len(vocab), 512)})
    tok = Tokenizer(models.BPE(vocab, [], unk_token = "<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<unk>", "<s>", "</s>"])
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as fp:
        json.dump({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, fp)
    return str(out_dir)

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
            # Distinct experts per token, as in the real table: a repeat would add one expert twice, and the
            # racing additions to one row would make the output vary from run to run
            t[f"{L}.ffn.gate.tid2eid"] = torch.rand(c["vocab_size"], c["n_routed_experts"]).argsort(-1)[
                :, : c["num_experts_per_tok"]].long().clone()
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
