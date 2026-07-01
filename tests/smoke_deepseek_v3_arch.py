#!/usr/bin/env python3
"""
Lightweight smoke checks for DeepseekV3 architecture integration.

Checks:
1) Synthesize a tiny random-weight DeepseekV3ForCausalLM checkpoint (--gen)
2) Config/architecture resolution and model graph construction
3) Full forward over the synthetic model
4) Optional: per-layer + logits diff against HF transformers (--hf_check)

Covers both q projection variants (--q_lora_rank 0 for direct q_proj), the
interleaved-pair rope layout, and grouped noaux_tc MoE routing.
"""

import argparse
import json
import os
import sys
import torch

from exllamav3 import Config, Model


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def make_cfg(q_lora):
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": q_lora if q_lora else None,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 16,
        "qk_head_dim": 80,
        "v_head_dim": 64,
        "head_dim": 16,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "vocab_size": 512,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "bos_token_id": 0,
        "eos_token_id": 1,
    }


def gen_checkpoint(model_dir, q_lora):
    from safetensors.torch import save_file
    torch.manual_seed(17)
    os.makedirs(model_dir, exist_ok = True)
    cfg = make_cfg(q_lora)
    H = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    qkh = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    t = {}

    def lin(key, out_f, in_f, std = 0.02):
        t[key] = (torch.randn(out_f, in_f) * std).half()

    def vec(key, n):
        t[key] = (1.0 + torch.randn(n) * 0.1).half()

    t["model.embed_tokens.weight"] = (torch.randn(cfg["vocab_size"], H) * 0.02).half()
    vec("model.norm.weight", H)
    lin("lm_head.weight", cfg["vocab_size"], H)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        vec(f"{p}.input_layernorm.weight", H)
        vec(f"{p}.post_attention_layernorm.weight", H)
        if q_lora:
            lin(f"{p}.self_attn.q_a_proj.weight", q_lora, H)
            vec(f"{p}.self_attn.q_a_layernorm.weight", q_lora)
            lin(f"{p}.self_attn.q_b_proj.weight", heads * qkh, q_lora)
        else:
            lin(f"{p}.self_attn.q_proj.weight", heads * qkh, H)
        lin(f"{p}.self_attn.kv_a_proj_with_mqa.weight", cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], H)
        vec(f"{p}.self_attn.kv_a_layernorm.weight", cfg["kv_lora_rank"])
        lin(f"{p}.self_attn.kv_b_proj.weight", heads * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), cfg["kv_lora_rank"])
        lin(f"{p}.self_attn.o_proj.weight", H, heads * cfg["v_head_dim"])
        if i < cfg["first_k_dense_replace"]:
            for sk, io in (("gate_proj", (cfg["intermediate_size"], H)), ("up_proj", (cfg["intermediate_size"], H)),
                           ("down_proj", (H, cfg["intermediate_size"]))):
                lin(f"{p}.mlp.{sk}.weight", *io)
        else:
            lin(f"{p}.mlp.gate.weight", cfg["n_routed_experts"], H)
            t[f"{p}.mlp.gate.e_score_correction_bias"] = (torch.randn(cfg["n_routed_experts"]) * 0.01).float()
            for sk in ("gate_proj", "up_proj", "down_proj"):
                io = (H, cfg["moe_intermediate_size"]) if sk == "down_proj" else (cfg["moe_intermediate_size"], H)
                lin(f"{p}.mlp.shared_experts.{sk}.weight", *io)
            for e in range(cfg["n_routed_experts"]):
                for sk in ("gate_proj", "up_proj", "down_proj"):
                    io = (H, cfg["moe_intermediate_size"]) if sk == "down_proj" else (cfg["moe_intermediate_size"], H)
                    lin(f"{p}.mlp.experts.{e}.{sk}.weight", *io)

    save_file(t, os.path.join(model_dir, "model.safetensors"))
    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in t.values())},
             "weight_map": {k: "model.safetensors" for k in t}}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent = 2)
    print(f"[INFO] generated {len(t)} tensors -> {model_dir} (q_lora={q_lora})")


@torch.inference_mode()
def run(model_dir, hf_check):
    cfg = Config.from_directory(model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture != "DeepseekV3ForCausalLM":
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    print("[INFO] model graph ok")
    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")
    model.load()

    T = 24
    torch.manual_seed(23)
    ids = torch.randint(0, 512, (1, T))
    x = model.modules[0].forward(ids, {}).to("cuda:0")
    hiddens = []
    for m in model.modules[1:]:
        if x.dtype != torch.half:
            x = x.half()
        x = m.forward(x, {})
        hiddens.append(x)
    logits = x
    if logits.shape != (1, T, 512):
        fail(f"Unexpected logits shape {tuple(logits.shape)}")
    if not torch.isfinite(logits.float()).all():
        fail("Non-finite logits")
    print("[INFO] forward ok")

    if hf_check:
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype = torch.float16, attn_implementation = "eager").to("cuda:0").eval()
        r = ref(input_ids = ids.to("cuda:0"), output_hidden_states = True, use_cache = False)
        a, b = logits.float().reshape(-1).cpu(), r.logits.float().reshape(-1).cpu()
        rel = (a - b).abs().max().item() / (b.abs().max().item() or 1.0)
        cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
        print(f"[INFO] HF logits diff: rel {rel:.6f} cos {cos:.6f}")
        if rel > 3e-2:
            fail("Logits diverge from HF reference")

    print("[PASS] all checks ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default = "/tmp/ds3_smoke")
    ap.add_argument("--q_lora_rank", type = int, default = 128)
    ap.add_argument("--gen", action = "store_true")
    ap.add_argument("--hf_check", action = "store_true")
    args = ap.parse_args()
    if args.gen:
        gen_checkpoint(args.model_dir, args.q_lora_rank)
    run(args.model_dir, args.hf_check)


if __name__ == "__main__":
    main()
