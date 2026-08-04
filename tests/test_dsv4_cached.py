"""
Cached-path (attn_mode flash_attn: rings + pools + DSA kernels) vs stateless nc path parity
for the full tiny DeepSeek-V4 model, plus rewind consistency. Drives the module list directly
(no generator); state advance is done manually like advance_recurrent_states would.

    python tests/test_dsv4_cached.py --device cuda:1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from compare_deepseek_v4_hf_ import make_checkpoint, TINY
from exllamav3 import Config, Model
from exllamav3.cache.cache import Cache


def fwd_modules(model, ids, params):
    params["input_ids"] = ids   # hash-MoE routing
    x = ids
    with torch.inference_mode():
        for m in model.modules:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
    return x[0].float().cpu()


def fwd_cached(model, ids, state, chunks):
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


def compare(tag, got, ref, kl_tol, arg_tol):
    am = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    lp_r = torch.log_softmax(ref.double(), -1)
    lp_g = torch.log_softmax(got.double(), -1)
    kld = (lp_r.exp() * (lp_r - lp_g)).sum(-1).mean().item()
    ok = am >= arg_tol and kld < kl_tol
    print(f"  {'PASS' if ok else 'FAIL'} {tag}: argmax {am*100:.2f}% KL {kld:.6f} "
          f"maxdiff {(got - ref).abs().max().item():.4f}")
    return ok


def noise_floor(model, ids, ref):
    """Chunk-shape GEMM noise reference: an fp16-tiling-scale perturbation at the embedding,
    run through the same nc path. The random tiny model amplifies ulp noise through MoE
    routing near-ties (the standard chaos-floor methodology); cached-vs-nc must sit at or
    below this floor, not at an absolute epsilon."""
    params = {"attn_mode": "flash_attn_nc", "input_ids": ids}
    x = ids
    with torch.inference_mode():
        for i, m in enumerate(model.modules):
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
            if i == 0:
                x = x + torch.randn_like(x) * 2e-4
    got = x[0].float().cpu()[-32:]
    lp_r = torch.log_softmax(ref[-32:].double(), -1)
    lp_g = torch.log_softmax(got.double(), -1)
    kld = (lp_r.exp() * (lp_r - lp_g)).sum(-1).mean().item()
    am = (got.argmax(-1) == ref[-32:].argmax(-1)).float().mean().item()
    return kld, am


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = args.device
    out_dir = "/tmp/dsv4_tiny_cached"
    make_checkpoint(out_dir, seed = 13)

    config = Config.from_directory(out_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 2)
    model.load(device)

    torch.manual_seed(21)
    seq = 315
    ids = torch.randint(0, TINY["vocab_size"], (1, seq), dtype = torch.long)
    ref = fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})
    floor_kl, floor_am = noise_floor(model, ids, ref)
    kl_tol = max(5e-4, 1.5 * floor_kl)
    arg_tol = min(0.99, floor_am - 0.05)
    print(f"  noise floor: KL {floor_kl:.6f} argmax {floor_am*100:.1f}% -> "
          f"tolerances KL {kl_tol:.6f} argmax {arg_tol*100:.1f}%")

    ok = True
    for chunks, tag in [
        ([seq], "single chunk"),
        ([100, 107, 108], "uneven chunks"),
        ([256, 30, 29], "page-aligned first"),
        ([300] + [1] * 15, "prefill + decode steps"),
    ]:
        with torch.inference_mode():
            state = cache.get_new_state()
        got = fwd_cached(model, ids, state, chunks)
        # nc discards trailing sub-window compressor rows per chunk; only the FINAL positions
        # of each run see identical entry sets, so compare the last 32 positions
        ok &= compare(f"cached vs nc, {tag}", got[-32:], ref[-32:], kl_tol, arg_tol)
        state.free()

    # Chunks larger than the SWA ring (768 rows at tiny window 8): exercises the temp-window
    # path and the ring rebase branch (the perf.py crash condition)
    seq2 = 2048
    torch.manual_seed(22)
    ids2 = torch.randint(0, TINY["vocab_size"], (1, seq2), dtype = torch.long)
    ref2 = fwd_modules(model, ids2, {"attn_mode": "flash_attn_nc"})
    for chunks, tag in [
        ([1024, 1024], "2x1024 (> ring)"),
        ([768, 640, 640], "mixed > ring"),
        ([2048], "single 2048"),
        ([1024, 1] * 8, "big + decode interleaved"),
    ]:
        with torch.inference_mode():
            state = cache.get_new_state()
        got = fwd_cached(model, ids2, state, chunks)
        ok &= compare(f"big-chunk cached vs nc, {tag}", got[-32:], ref2[-32:], kl_tol, arg_tol)
        state.free()

    # Rewind consistency: logits for re-decoded tokens must match the first pass exactly
    with torch.inference_mode():
        state = cache.get_new_state()
    pre = fwd_cached(model, ids[:, :300], state, [300])
    first = fwd_cached(model, ids[:, 300:312], state, [1] * 12)
    state.rewind(12)
    second = fwd_cached(model, ids[:, 300:312], state, [1] * 12)
    err = (first - second).abs().max().item()
    ok &= err == 0.0
    print(f"  {'PASS' if err == 0.0 else 'FAIL'} rewind-and-replay: maxdiff {err:.2e}")
    state.free()

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
