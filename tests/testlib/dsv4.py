"""
DeepSeek-V4 test helpers over the tiny random checkpoint (testlib.tiny_models.make_dsv4_checkpoint): drive the
module list directly (no generator) through the stateless nc path or the cached path with manual state advance,
and compare logits against the chunk-shape noise floor.

    config, model = build_tiny_dsv4(out_dir, seed = 13)          # create caches, then model.load(device)
    ref =fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})
    floor_kl, floor_am = noise_floor(model, ids, ref)
    got = fwd_cached(model, ids, cache.get_new_state(), [100, 107, 108])
    assert_logits_close(got[-32:], ref[-32:], kl_tol, arg_tol, "uneven chunks")

The random tiny model amplifies ulp-level differences through MoE routing near-ties, so path comparisons are held
to a tolerance derived from a measured noise floor (an fp16-tiling-scale perturbation at the embedding run through
the same nc path), not to an absolute epsilon.
"""

import torch

from testlib.tiny_models import make_dsv4_checkpoint


def build_tiny_dsv4(out_dir, seed: int, **overrides):
    """(config, model) for a fresh tiny checkpoint in out_dir; create caches on the model, then model.load()"""
    from exllamav3 import Config, Model
    make_dsv4_checkpoint(str(out_dir), seed = seed, **overrides)
    config = Config.from_directory(str(out_dir))
    return config, Model.from_config(config)


def attention_layers(model, layer_type: str | None = None) -> list:
    """The model's DSV4Attention modules in layer order, optionally only one layer type (sliding | csa | hca)"""
    from exllamav3.modules.dsv4 import DSV4Attention
    return [m for m in model if isinstance(m, DSV4Attention) and (layer_type is None or m.layer_type == layer_type)]


def fwd_modules(model, ids, params) -> torch.Tensor:
    """Run ids through every module of the model with params; (seq, vocab) fp32 logits on the CPU"""
    params["input_ids"] = ids   # hash-MoE routing
    x = ids
    with torch.inference_mode():
        for m in model.modules:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
    return x[0].float().cpu()


def fwd_cached(model, ids, state, chunks) -> torch.Tensor:
    """Cached-path forward of ids in chunks of the given sizes, advancing the recurrent state as
    advance_recurrent_states would; (seq, vocab) logits"""
    outs = []
    a = 0
    for size in chunks:
        b = min(a + size, ids.shape[1])
        if b <= a:
            break
        params = {"attn_mode": "flash_attn", "recurrent_states": [state]}
        outs.append(fwd_modules(model, ids[:, a:b], params))
        state.position += b - a
        state.post_advance()
        a = b
    return torch.cat(outs, dim = 0)


def logit_stats(got, ref) -> tuple[float, float]:
    """(argmax agreement, mean KL(ref || got)) of two (rows, vocab) logit tensors, in float64"""
    am = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    lp_r = torch.log_softmax(ref.double(), -1)
    lp_g = torch.log_softmax(got.double(), -1)
    kld = (lp_r.exp() * (lp_r - lp_g)).sum(-1).mean().item()
    return am, kld


def assert_logits_close(got, ref, kl_tol: float, arg_tol: float, tag: str = ""):
    am, kld = logit_stats(got, ref)
    assert am >= arg_tol and kld < kl_tol, (
        f"{tag}: argmax {am * 100:.2f}% (min {arg_tol * 100:.1f}%), KL {kld:.6f} (max {kl_tol:.6f}), "
        f"maxdiff {(got - ref).abs().max().item():.4f}"
    )


def noise_floor(model, ids, ref, last: int = 32) -> tuple[float, float]:
    """(KL, argmax agreement) on the last positions of the nc path with an fp16-tiling-scale perturbation (2e-4
    Gaussian) added to the embedding output, against the unperturbed reference: the chunk-shape GEMM noise
    floor that cached-vs-nc comparisons must sit at or below"""
    params = {"attn_mode": "flash_attn_nc", "input_ids": ids}
    x = ids
    with torch.inference_mode():
        for i, m in enumerate(model.modules):
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
            if i == 0:
                x = x + torch.randn_like(x) * 2e-4
    got = x[0].float().cpu()[-last:]
    am, kld = logit_stats(got, ref[-last:])
    return kld, am


def chunk_splits(seq: int, pattern: list[int]) -> list[tuple[int, int]]:
    """[(a, b)] row ranges covering seq: the pattern's chunk sizes, then one chunk with the remainder"""
    out, a = [], 0
    for p in pattern:
        b = min(a + p, seq)
        if b > a:
            out.append((a, b))
        a = b
    if a < seq:
        out.append((a, seq))
    return out
