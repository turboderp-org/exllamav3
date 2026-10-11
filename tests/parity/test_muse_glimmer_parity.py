"""
MuseGlimmer (MuseGlimmerForConditionalGeneration) parity against HF transformers, text model and vision tower.

Tiny random-weight checkpoint in the real release's tensor namespace. Text: Gemma2-style sandwich norms (centered,
split pre/post eps), scaleless embedding norm, scaleless shared QK-norm with qk_scale_factor folded into sm_scale,
sigmoid full-gate attention, per-layer RoPE with NoPE on the full-attention layers, SWA on the rest, and the
output_multiplier -> tanh softcap head; gated on KL and argmax of the full-sequence logits. Vision: linear patch
embedding with temporal duplication, bilinearly interpolated learned position embeddings, [w, h] interleaved
1-indexed 2D rope, window attention reorder, pixel shuffle and the gelu adapter stack + scaleless projection norm;
gated on the relative Frobenius error and per-token cosine of the projected embeddings. Reference: HF fp32 eager.
Mutation tests: qk_scale_factor unfolded from sm_scale must fail the text gate, a zeroed vision rope the vision gate.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from testlib.parity import Gate, exl3_logits, free_cuda, load_hf, logit_gate, logit_stats, min_cos, mutate, rfn, token_ids

pytestmark = pytest.mark.hf

TEXT = dict(
    vocab_size = 512,
    hidden_size = 128,
    intermediate_size = 256,
    num_hidden_layers = 4,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 32,
    sliding_window = 16,
    rms_norm_eps = 1e-5,
    post_norm_eps = 1e-8,
    qk_scale_factor = 3.87,
    output_multiplier = 0.19611613513818404,
    final_logit_softcapping = 20.0,
    layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    layer_rope_theta = [500000.0, 500000.0, 500000.0, 0],
)

# Vision dims are multiples of 128 so the tower stays quantizable (EXL3 Hadamard block alignment)
VISION = dict(
    hidden_size = 128,
    num_hidden_layers = 5,
    intermediate_size = 256,
    num_attention_heads = 4,
    patch_size = 4,
    patch_temporal = 2,
    merge_size = 2,
    pos_emb_height = 8,
    pos_emb_width = 8,
    layer_norm_eps = 1e-5,
    layer_types = ["window_attention", "window_attention", "window_attention", "full_attention", "full_attention"],
)

PROJECTOR_HIDDEN = 128
PATCH_DIM = VISION["patch_temporal"] * 3 * VISION["patch_size"] ** 2
OUT_HIDDEN = VISION["hidden_size"] * VISION["merge_size"] ** 2

SEED = 7
SEQ_LEN = 96            # several sliding windows deep
# 40 x 24 px image at patch 4 / merge 2: patch grid 10 x 6, merged 5 x 3; the window side is pos_emb_height (8)
# patches, so windows pad unevenly in both dimensions
GRID_THW = (1, 10, 6)


def config_json() -> dict:
    return {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 500,
        "video_token_id": 501,
        "out_hidden_size": OUT_HIDDEN,
        "projector_hidden_size": PROJECTOR_HIDDEN,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_activation": "silu",
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            **TEXT,
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_act": "gelu",
            "max_position_embeddings": VISION["pos_emb_height"] * VISION["pos_emb_width"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            **VISION,
        },
    }


def processor_json() -> dict:
    return {
        "processor_class": "MuseGlimmerProcessor",
        "image_processor": {
            "image_processor_type": "MuseGlimmerImageProcessor",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "resample": 1,
            "rescale_factor": 0.00392156862745098,
            "patch_size": VISION["patch_size"],
            "temporal_patch_size": VISION["patch_temporal"],
            "merge_size": VISION["merge_size"],
            "max_image_tokens": 64,
        },
    }


def make_checkpoint(out_dir: str, seed: int) -> str:
    torch.manual_seed(seed)
    c = TEXT
    v = VISION
    H = c["hidden_size"]
    t = {}

    def w(name, *shape, scale = 0.02):
        t[name] = (torch.randn(*shape, dtype = torch.float) * scale).to(torch.bfloat16)

    def normw(name, dim, centered = False):
        t[name] = ((0.0 if centered else 1.0) + 0.1 * torch.randn(dim)).to(torch.bfloat16)

    def biasw(name, dim):
        t[name] = (0.02 * torch.randn(dim)).to(torch.bfloat16)

    lm = "model.language_model"
    # Large head scale: spreads the logits so argmax comparison is meaningful, and pushes them into the nonlinear
    # region of the tanh softcap
    w("lm_head.weight", c["vocab_size"], H, scale = 2.0)
    w(f"{lm}.embed_tokens.weight", c["vocab_size"], H, scale = 1.0)
    normw(f"{lm}.norm.weight", H)

    for i in range(c["num_hidden_layers"]):
        L = f"{lm}.layers.{i}"
        normw(f"{L}.input_layernorm.weight", H, centered = True)
        normw(f"{L}.post_attention_layernorm.weight", H, centered = True)
        normw(f"{L}.pre_feedforward_layernorm.weight", H, centered = True)
        normw(f"{L}.post_feedforward_layernorm.weight", H, centered = True)
        w(f"{L}.self_attn.q_proj.weight", c["num_attention_heads"] * c["head_dim"], H)
        w(f"{L}.self_attn.k_proj.weight", c["num_key_value_heads"] * c["head_dim"], H)
        w(f"{L}.self_attn.v_proj.weight", c["num_key_value_heads"] * c["head_dim"], H)
        w(f"{L}.self_attn.gate_proj.weight", c["num_attention_heads"] * c["head_dim"], H)
        w(f"{L}.self_attn.o_proj.weight", H, c["num_attention_heads"] * c["head_dim"])
        w(f"{L}.mlp.gate_proj.weight", c["intermediate_size"], H)
        w(f"{L}.mlp.up_proj.weight", c["intermediate_size"], H)
        w(f"{L}.mlp.down_proj.weight", H, c["intermediate_size"])

    vt = "model.vision_tower"
    VH = v["hidden_size"]
    w(f"{vt}.patch_embedder.patch_embedding.weight", VH, PATCH_DIM, scale = 0.05)
    w(f"{vt}.patch_embedder.position_embedding_table.weight", v["pos_emb_height"] * v["pos_emb_width"], VH,
      scale = 0.05)
    normw(f"{vt}.ln_pre.weight", VH)
    biasw(f"{vt}.ln_pre.bias", VH)
    normw(f"{vt}.ln_post.weight", VH)
    biasw(f"{vt}.ln_post.bias", VH)

    for i in range(v["num_hidden_layers"]):
        L = f"{vt}.layers.{i}"
        for n in ["norm1", "norm2"]:
            normw(f"{L}.{n}.weight", VH)
            biasw(f"{L}.{n}.bias", VH)
        for n in ["q_proj", "k_proj", "v_proj", "proj"]:
            w(f"{L}.attn.{n}.weight", VH, VH, scale = 0.05)
            biasw(f"{L}.attn.{n}.bias", VH)
        w(f"{L}.mlp.fc1.weight", v["intermediate_size"], VH, scale = 0.05)
        biasw(f"{L}.mlp.fc1.bias", v["intermediate_size"])
        w(f"{L}.mlp.fc2.weight", VH, v["intermediate_size"], scale = 0.05)
        biasw(f"{L}.mlp.fc2.bias", VH)

    w("model.vision_adapter.fc1.weight", PROJECTOR_HIDDEN, OUT_HIDDEN, scale = 0.05)
    w("model.vision_adapter.fc2.weight", PROJECTOR_HIDDEN, PROJECTOR_HIDDEN, scale = 0.05)
    w("model.vision_projection.weight", H, PROJECTOR_HIDDEN, scale = 0.05)

    os.makedirs(out_dir, exist_ok = True)
    save_file(t, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(config_json(), fp, indent = 2)
    with open(os.path.join(out_dir, "processor_config.json"), "w") as fp:
        json.dump(processor_json(), fp, indent = 2)
    return out_dir


def make_pixels(seed: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Random pre-normalized patches (grid_h * grid_w, PATCH_DIM) with the temporal duplication the patchifier
    produces"""
    g = torch.Generator().manual_seed(seed)
    single = torch.randn(grid_h * grid_w, PATCH_DIM // VISION["patch_temporal"], generator = g)
    return single.repeat(1, VISION["patch_temporal"]).contiguous()


@pytest.fixture(scope = "module")
def tiny(tmp_path_factory, device):
    """(model dir, ids, pixels, HF logits, HF vision embeddings)"""
    from transformers import MuseGlimmerForConditionalGeneration
    model_dir = make_checkpoint(str(tmp_path_factory.mktemp("muse_tiny")), seed = SEED)
    ids = token_ids(SEQ_LEN, TEXT["vocab_size"], seed = SEED + 1)
    pixels = make_pixels(SEED + 2, GRID_THW[1], GRID_THW[2])
    model = load_hf(MuseGlimmerForConditionalGeneration, model_dir, device = device, dtype = torch.float32)
    with torch.inference_mode():
        logits = model(input_ids = ids.to(device), use_cache = False).logits[0].float().cpu()
        vemb = model.model.get_image_features(
            pixel_values = pixels.to(device),
            image_grid_thw = torch.tensor([list(GRID_THW)], device = device),
        ).pooler_output[0].float().cpu()
    del model
    free_cuda()
    return model_dir, ids, pixels, logits, vemb


def exl3_vision(model_dir: str, pixels: torch.Tensor, device, zero_rope: bool = False) -> torch.Tensor:
    from exllamav3 import Config, Model
    config = Config.from_directory(model_dir)
    vmodel = Model.from_config(config, component = "vision")
    vmodel.load(torch.device(device))
    try:
        vparams = vmodel.make_vision_params(GRID_THW)
        if zero_rope:
            vparams["inv_freq"] = vparams["inv_freq"] * 0.0
        with torch.inference_mode():
            return vmodel.forward(pixels.half().unsqueeze(0), vparams)[0].float().cpu()
    finally:
        vmodel.unload()
        free_cuda()


def text_gate(got: torch.Tensor, ref: torch.Tensor, tag: str) -> Gate:
    gate = Gate(tag)
    logit_gate(gate, logit_stats(got, ref), kl_mean = 1e-4, kl_max = 1e-3, argmax = 0.99)
    return gate


def vision_gate(got: torch.Tensor, ref: torch.Tensor, tag: str) -> Gate:
    gate = Gate(tag)
    gate.true("shape", got.shape == ref.shape, f"{tuple(got.shape)} vs {tuple(ref.shape)}")
    if got.shape == ref.shape:
        gate.lt("rfn", rfn(got, ref), 5e-3)
        gate.gt("min token cos", min_cos(got, ref), 0.999)
    return gate


def _unfold_qk_scale(model):
    head_dim = model.config.head_dim
    mutate((getattr(m, "attn", None) for m in model.modules), "sm_scale", head_dim ** -0.5)


def test_text_logits(tiny, device):
    model_dir, ids, _, ref, _ = tiny
    text_gate(exl3_logits(model_dir, ids, device), ref, "MuseGlimmer text vs HF fp32 eager").assert_passes()


def test_vision_embeddings(tiny, device):
    model_dir, _, pixels, _, ref = tiny
    vision_gate(exl3_vision(model_dir, pixels, device), ref, "MuseGlimmer vision vs HF fp32 eager").assert_passes()


def test_text_mutation_qk_scale(tiny, device):
    model_dir, ids, _, ref, _ = tiny
    got = exl3_logits(model_dir, ids, device, configure = _unfold_qk_scale)
    text_gate(got, ref, "MuseGlimmer text with qk_scale_factor unfolded").assert_fails()


def test_vision_mutation_rope(tiny, device):
    model_dir, _, pixels, _, ref = tiny
    vision_gate(exl3_vision(model_dir, pixels, device, zero_rope = True), ref,
                "MuseGlimmer vision with the 2D rope zeroed").assert_fails()
