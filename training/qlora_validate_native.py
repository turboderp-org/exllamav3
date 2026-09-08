"""
Validate the transformers-free differentiable Llama forward against the native
(correct) exllamav3 inference forward.

This is the correctness gate for QLoRA-on-EXL3 via the native path: before
training anything, prove that ``NativeLlamaQLoRA`` (pure autograd PyTorch on the
quantized weights) reproduces the logits that exllamav3's own kernels produce.
If the top-1 tokens match and the per-token loss is low, the differentiable
backbone is sound and any adapter trained on it is meaningful.

Unlike the HuggingFace integration, nothing here depends on a ``transformers``
version -- it reuses exllamav3's loaded weights and RoPE table directly.

Usage:
    # single device
    python training/qlora_validate_native.py --model /path/to/exl3_model

    # layer-autosplit across all visible GPUs (smoke-test the device-aware
    # forward); --check-backward also exercises cross-device gradient flow
    python training/qlora_validate_native.py --model /path/to/exl3_model \
        --parallel split --check-backward

On a small model that fits one card, force a real split boundary by capping the
per-device budget, e.g. ``--use-per-device 1 8`` (≈1 GB on cuda:0 spills the rest
to cuda:1). On a model too big for one card, plain ``--parallel split`` balances
naturally. The printed block-device distribution shows where the boundary landed.
"""

import argparse
from collections import Counter
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training import backbone
from exllamav3.training.native_llama import NativeLlamaQLoRA


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Water is made of hydrogen and",
]


def check_vision(model, net, tokenizer, config, image_path, prompt, device, cdt):
    """
    Image+text gate (--image): the differentiable forward fed the frozen
    vision features through the training splice (``mm``) must reproduce the
    native inference forward fed the same ``MMEmbedding`` -- the exact path
    the generator runs for a multimodal prompt (indexed embeddings, mRoPE
    alt frequencies, Gemma4 bidirectional image spans). One vision forward
    produces the features for BOTH sides, so this isolates the text tower's
    multimodal handling: the embedding splice, the 3-D position ids + per-band
    rotation, the deepstack adds, the span mask. Also cross-checks the Python
    mRoPE position ids against the extension kernel on an mRoPE tower.
    """
    print("\n" + "-" * 78)
    print(f"vision check: image+text forward vs native ({image_path})")
    from PIL import Image
    from exllamav3.training import vision as tv
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    from vision_data import resolve_placeholder_id
    if "vision" not in config.model_classes:
        print("  this architecture has no vision component -- skipping")
        return True
    vision_model = Model.from_config(config, component="vision")
    vision_model.load(device=device, progressbar=True)
    image = Image.open(image_path)
    image.load()
    with torch.inference_mode():
        mme = vision_model.get_image_embeddings(tokenizer=tokenizer, image=image)
    n_ds = len(mme.deepstack_embeddings) if mme.deepstack_embeddings is not None else 0
    print(f"  image -> {mme.mm_length} feature tokens, grid {mme.grid_thw}, "
          f"merge {mme.mrope_merge_size}, deepstack maps {n_ds}; tower: "
          f"mrope={net.has_mrope}, deepstack after blocks {sorted(net._deepstack) or 'none'}, "
          f"bidirectional spans={net._bidir_mm}")

    # Native oracle: the generator's prefill params for a multimodal prompt.
    text = model.default_chat_prompt(mme.text_alias + "\n" + prompt)
    ids = tokenizer.encode(text, encode_special_tokens=True, embeddings=[mme])   # [1, t]
    params = {"indexed_embeddings": [mme]}
    if model.caps.get("mrope"):
        freqs, _ = model.g_rope.get_mrope_freqs(ids, [mme], ids.shape[-1])
        params["inv_freq"] = freqs
    with torch.inference_mode():
        logits_native = model.forward(ids.to(device), params).float()          # [1, t, V]

    # Differentiable side: the same features through the training splice.
    feats = tv.features_from_mme(mme, store_dtype=torch.float32)
    ids_cpu = ids[0].tolist()
    mm_pos = [i for i, t in enumerate(ids_cpu) if t >= FIRST_MM_EMBEDDING_INDEX]
    start = mm_pos[0] - feats.slot_offsets[0]
    placeholder = resolve_placeholder_id(config, tokenizer)
    train_ids = [placeholder if t >= FIRST_MM_EMBEDDING_INDEX else t for t in ids_cpu]
    assert train_ids[start:start + len(feats.token_ids)] == \
        [placeholder if t == tv.IMAGE_SLOT else t for t in feats.token_ids], \
        "image token layout does not line up with the tokenized prompt"
    t = len(train_ids)
    mm = tv.build_mm_batch([[(start, feats)]], t, cdt)
    pos = None
    ok = True
    if net.has_mrope:
        pos = tv.mrope_position_ids(t, [(mm_pos[0], feats.n_tokens, feats.grid_thw)],
                                    feats.merge_size)
        # The kernel the generator uses -> must agree with the Python mirror.
        from exllamav3.ext import exllamav3_ext as ext
        ref = torch.zeros((3, t), dtype=torch.long)
        ext.gen_mrope_pos_ids(ref, ids[0].contiguous(), feats.merge_size,
                              [(mme.first_index, mme.last_index)], [tuple(mme.grid_thw)])
        same = torch.equal(pos, ref)
        ok &= same
        print(f"  mRoPE position ids (python) == gen_mrope_pos_ids kernel: "
              f"{'OK' if same else 'MISMATCH'}  (text resumes at {int(pos[0, -1])} "
              f"for {t} tokens)")
        pos = pos.unsqueeze(1).to(device)                                     # [3, 1, t]
    with torch.no_grad():
        logits_diff = net.logits(torch.tensor([train_ids], device=device),
                                 mm=mm, position_ids=pos).float()
    logits_diff = logits_diff.to(logits_native.device)

    ln, ld = logits_native[0, -1], logits_diff[0, -1]
    top1_native, top1_diff = int(ln.argmax()), int(ld.argmax())
    match = top1_native == top1_diff
    argmax_n = logits_native[0].argmax(-1)
    argmax_d = logits_diff[0].argmax(-1)
    agree_all = (argmax_n == argmax_d).float().mean().item()
    after = mm_pos[-1] + 1
    agree_text = (argmax_n[after:] == argmax_d[after:]).float().mean().item()
    agree_img = (argmax_n[mm_pos] == argmax_d[mm_pos]).float().mean().item()
    max_abs = (ln - ld).abs().max().item()
    cos = torch.cosine_similarity(ln, ld, dim=0).item()
    is_lowp = cdt in (torch.float16, torch.bfloat16)
    good = (match and agree_text >= 0.9) or (is_lowp and agree_text >= 0.8 and cos >= 0.999)
    ok &= good
    dec = lambda i: repr(tokenizer.decode(torch.tensor([[i]]), decode_special_tokens=True)[0])
    print(f"  prompt: {prompt!r} (+ image, {t} tokens)")
    print(f"  native next-token : {dec(top1_native)}")
    print(f"  diff   next-token : {dec(top1_diff)}   {'OK' if match else 'MISMATCH'}")
    print(f"  per-position argmax agreement: all {agree_all*100:.1f}% | "
          f"text after image {agree_text*100:.1f}% | image positions {agree_img*100:.1f}%")
    print(f"  last-token logits: max|Δ|={max_abs:.4f}  cos={cos:.6f}")
    print("  vision check:", "PASS" if ok else "FAIL")
    vision_model.unload()
    return ok


def check_backward(model, tokenizer, prompt, device, cdt, attn_impl="auto",
                   use_liger=False):
    """
    Smoke-test cross-device gradient flow: attach a tiny adapter, run one
    loss.backward() with gradient checkpointing, and assert that gradients
    reached adapters on *every* device the decoder is split across. This is the
    part of the device-aware forward that the forward-only gate can't cover --
    autograd flowing back through the cross-device hidden-state migrations.
    """
    print("\n" + "-" * 78)
    print("backward smoke: cross-device gradient flow through the split")
    try:
        net = NativeLlamaQLoRA(model, r=4, alpha=8.0,
                               target_modules=["q_proj", "down_proj"],
                               compute_dtype=cdt, gradient_checkpointing=True,
                               attn_impl=attn_impl, use_liger=use_liger)
    except ValueError:
        # A pure-MoE model with no shared expert has no plain down_proj (the
        # routed experts need the explicit expert_* targets); q_proj alone
        # still exercises the cross-device backward this smoke is for.
        net = NativeLlamaQLoRA(model, r=4, alpha=8.0,
                               target_modules=["q_proj"],
                               compute_dtype=cdt, gradient_checkpointing=True,
                               attn_impl=attn_impl, use_liger=use_liger)
    net.train()
    ids = tokenizer.encode(prompt, add_bos=True).to(device)
    loss = net.compute_loss(ids, ids.clone())
    loss.backward()

    expected = sorted({str(d) for d in net._block_devices})
    have = set()
    missing = []
    for w in net._wrappers:
        if w.r <= 0:
            continue
        # B inits to zero, so on the first step the gradient flows to B (grad_A
        # is exactly zero while B == 0); check B to confirm the adapter was hit.
        g = w.lora_b.grad
        if g is not None and g.abs().sum().item() > 0:
            have.add(str(w.lora_b.device))
        else:
            missing.append(w.key)
    ok = (set(expected) <= have) and not missing
    print(f"  loss = {loss.item():.4f}")
    print(f"  adapters received grad on : {sorted(have)}")
    print(f"  decoder split devices     : {expected}")
    if missing:
        print(f"  MISSING grad on {len(missing)} adapters, e.g. {missing[:3]}")
    print("  backward smoke:", "PASS" if ok else "FAIL")
    return ok


def check_liger_parity(model, tokenizer, prompt, device, cdt, attn_impl="auto"):
    """
    The Liger correctness gate (Session 10 #3), two tiers. The old --use-liger
    coverage only proved backward *runs* and reaches every device, so a
    wrong-VALUE gradient -- exactly the in_place=True corruption fixed in #119
    -- sailed through with a healthy-looking loss.

    Tier 1 (fp32 math gate): torch-vs-liger with fp32 compute. At fp32 the two
    paths compute the same math with only kernel reassociation between them, so
    gradients must agree near-exactly; a miss here means the liger backward
    FORMULA is wrong (or a buffer is being corrupted), independent of any
    half-precision noise story.

    Tier 2 (noise-band gate, only when --compute-dtype is half): the same
    compare at the actual training dtype, with tolerances calibrated to the
    measured benign spread. Box-measured on Semancer-12B (48 layers, bf16,
    liger RMSNorm only -- GeGLU keeps the SwiGLU kernel out): median cos
    0.9976 / rel 7.1e-2, worst 0.9818 / 0.20 at layer 1, identical across
    runs. The divergence accumulates toward the earliest layers (deepest
    backward), which is the reassociation signature; #119-class corruption is
    orders of magnitude outside either tier.

    Each tier builds two identically-seeded adapter nets over the same frozen
    base, runs one loss.backward() each on the same batch, and compares the
    loss plus every adapter gradient.
    """
    print("\n" + "-" * 78)
    print("liger parity: torch vs liger loss/grad on identically-seeded adapters")

    ids = tokenizer.encode(prompt, add_bos=True).to(device)

    def build(use_liger, dtype):
        # Same seed -> identical kaiming init of every lora_a (B starts at 0),
        # so the two nets are the same function and gradients are comparable.
        torch.manual_seed(0)
        try:
            net = NativeLlamaQLoRA(
                model, r=8, alpha=16.0,
                # gate/up/down exercise the Liger SwiGLU (silu models); q_proj sits
                # after the input RMSNorm so it sees the Liger norm's backward too.
                target_modules=["q_proj", "gate_proj", "up_proj", "down_proj"],
                compute_dtype=dtype, gradient_checkpointing=True,
                attn_impl=attn_impl, use_liger=use_liger)
        except ValueError:
            # Pure-MoE, no shared expert: no plain gate/up/down_proj to adapt.
            torch.manual_seed(0)
            net = NativeLlamaQLoRA(
                model, r=8, alpha=16.0, target_modules=["q_proj"],
                compute_dtype=dtype, gradient_checkpointing=True,
                attn_impl=attn_impl, use_liger=use_liger)
        net.train()
        return net

    def run(net):
        loss = net.compute_loss(ids, ids.clone())
        loss.backward()
        grads = {}
        probe = None
        for w in net._wrappers:
            if w.r <= 0:
                continue
            if probe is None:
                probe = w.lora_a.detach().clone()
            # First-step grads: B==0 makes grad_A exactly zero, so lora_b.grad
            # carries the signal; grab both (a compares as zero-vs-zero).
            for name, p in (("a", w.lora_a), ("b", w.lora_b)):
                g = p.grad
                grads[f"{w.key}.{name}"] = (
                    None if g is None else g.detach().float().cpu())
        return loss.item(), grads, probe

    def tier(label, dtype, cos_min, rel_max, loss_rel_max):
        net = build(False, dtype)
        loss_t, g_t, probe_t = run(net)
        del net
        net = build(True, dtype)
        loss_l, g_l, probe_l = run(net)
        del net
        print(f"  [{label}]")
        # Sanity: the seeded inits really are identical, else the compare is void.
        if not torch.equal(probe_t, probe_l):
            print("    FAIL -- seeded adapter inits differ; parity compare is void")
            return False

        loss_rel = abs(loss_t - loss_l) / max(abs(loss_t), 1e-9)
        ok = loss_rel < loss_rel_max
        stats = []                                    # (cos, rel, key)
        for key in g_t:
            gt, gl = g_t[key], g_l[key]
            if gt is None or gl is None:
                if gt is not gl:                      # grad on one side only
                    ok = False
                    print(f"    FAIL -- {key}: grad present on only one side")
                continue
            nt = gt.norm().item()
            nl = gl.norm().item()
            if nt == 0.0 and nl == 0.0:
                continue                              # e.g. all lora_a at step 1
            cos = torch.cosine_similarity(gt.flatten(), gl.flatten(), dim=0).item()
            rel = (gt - gl).norm().item() / max(nt, 1e-12)
            stats.append((cos, rel, key))
        fails = [(c, r, k) for c, r, k in stats
                 if not (c > cos_min and r < rel_max)]
        ok &= not fails
        by_cos = sorted(stats)
        print(f"    loss: torch {loss_t:.6f} vs liger {loss_l:.6f}  "
              f"(rel {loss_rel:.2e}, bound {loss_rel_max:.0e})")
        print(f"    {len(stats)} adapter grads compared, {len(fails)} outside "
              f"tolerance (cos > {cos_min}, rel < {rel_max})")
        if by_cos:
            med = by_cos[len(by_cos) // 2]
            print(f"    median cosine: {med[0]:.6f}   rel {med[1]:.2e}")
            # The distribution separates the two failure classes: a numerics
            # gap shows deep-layer outliers over a tight median; a corrupted
            # backward blows out most of the list.
            for c, r, k in by_cos[:5]:
                print(f"      {c:.6f}  rel {r:.2e}  {k}")
        print(f"    {label}:", "PASS" if ok else "FAIL")
        return ok

    # Tier 1: fp32 -- near-exact or the liger backward math is wrong. The
    # bounds leave room for kernel reassociation only.
    all_ok = tier("fp32 math gate", torch.float32,
                  cos_min=0.9999, rel_max=5e-3, loss_rel_max=1e-4)

    # Tier 2: the actual training dtype, calibrated to the measured benign
    # spread (see docstring); skipped when the run is already fp32.
    if cdt in (torch.float16, torch.bfloat16):
        all_ok &= tier(f"{str(cdt).split('.')[-1]} noise-band gate", cdt,
                       cos_min=0.95, rel_max=0.35, loss_rel_max=2e-2)

    print("  liger parity:", "PASS" if all_ok else
          "FAIL -- liger backward diverges from torch; do NOT train with --use-liger")
    return all_ok


def check_init_lora(model, tokenizer, prompt, device, cdt, mode,
                    ref_model_dir=None, svd_niter=16, attn_impl="auto"):
    """
    Step-0 gate for the SVD adapter inits (--init-lora pissa/qerr): before any
    training run trusts them, verify the model they produce at step 0 is the
    one the math promises. The whole class of bookkeeping bugs (offset sign,
    scale folding, orientation, padding) shows up here as a hard FAIL instead
    of a mysteriously worse training run.

    pissa: function-preserving by construction -- the trainable adapter starts
    equal to the frozen offset, so the step-0 loss must match the base model's.
    Gated near-exactly at fp32 compute (tier-1 style: only reassociation of
    the subtract-then-add-back may differ); at a half training dtype the
    cancellation of the large principal component is inherently noisier, so
    that tier gets a loose calibrated bound and a printed delta.

    qerr: NOT function-preserving by design -- step 0 is the closest rank-r
    repair of the ORIGINAL (unquantized) model, so the loss should move a
    little, typically toward the bf16 model's. Reported with a wide sanity
    bound only; the exact factor math is covered by the CPU unit tests
    (tests/test_lora_init.py).

    eva: B stays zero, so the step-0 delta is exactly zero in EVERY dtype;
    the gate proves the activation pre-pass (hooks, shared-input sites,
    sketch SVD) runs on the real model and leaves the function untouched,
    with the gate prompt itself as the pre-pass data. Tight bound at both
    tiers -- any deviation means the init path corrupted something.
    """
    print("\n" + "-" * 78)
    print(f"init-lora gate ({mode}): step-0 model vs frozen base")
    # This gate certifies the INIT's bookkeeping (offset sign, scale folding,
    # orientation), so it runs on the legacy dequant path, where the forward
    # is fp32-linear and the near-exact fp32 bound is meaningful. The fast
    # path rounds activations through fp16 at every base matmul, which
    # amplifies the (correct) cancellation residue to fp16-noise scale --
    # box-measured 1.7e-4 vs the same init's 5.5e-7 under legacy on the 1B.
    # The fast path's own offset handling is certified fast-vs-legacy by the
    # dequant-parity gate's pissa round.
    print("  (runs on the legacy dequant path; fast-path offset parity is "
          "covered by the dequant-parity gate)")
    from exllamav3.training import backbone as _bb

    ids = tokenizer.encode(prompt, add_bos=True).to(device)

    def loss_of(dtype, with_init):
        torch.manual_seed(0)
        net = NativeLlamaQLoRA(
            model, r=8, alpha=8.0,
            target_modules=["q_proj", "v_proj", "down_proj"],
            compute_dtype=dtype, gradient_checkpointing=True,
            attn_impl=attn_impl)
        net.train()
        if with_init:
            net.apply_init_lora(
                mode, ref_model_dir=ref_model_dir, svd_niter=svd_niter,
                eva_batches=([{"input_ids": ids}] if mode == "eva" else None))
        prev = _bb.dequant_mode()
        _bb.set_dequant_mode("legacy")
        try:
            with torch.no_grad():
                loss = net.compute_loss(ids, ids.clone()).item()
        finally:
            _bb.set_dequant_mode(prev)
        del net
        return loss

    def tier(label, dtype, rel_bound, hard):
        base = loss_of(dtype, False)       # B=0 default init == exact base model
        init = loss_of(dtype, True)
        rel = abs(init - base) / max(abs(base), 1e-9)
        ok = rel < rel_bound
        print(f"  [{label}] base loss {base:.6f} vs {mode}-init {init:.6f}  "
              f"(rel {rel:.2e}, bound {rel_bound:.0e})"
              f"{'' if hard else '  [informational]'}")
        print(f"    {label}:", "PASS" if ok else
              ("FAIL" if hard else "OUTSIDE BOUND (informational)"))
        return ok if hard else True

    if mode == "pissa":
        # fp32: the offset must cancel the init adapter near-exactly.
        all_ok = tier("fp32 function-preservation gate", torch.float32,
                      rel_bound=1e-4, hard=True)
        if cdt in (torch.float16, torch.bfloat16):
            all_ok &= tier(f"{str(cdt).split('.')[-1]} noise band", cdt,
                           rel_bound=2e-2, hard=True)
    elif mode == "eva":
        # x@A@B with B=0 adds exactly zero, so both tiers are hard and tight:
        # any deviation at all means the pre-pass corrupted the model/adapter.
        all_ok = tier("fp32 function-preservation gate", torch.float32,
                      rel_bound=1e-6, hard=True)
        if cdt in (torch.float16, torch.bfloat16):
            all_ok &= tier(f"{str(cdt).split('.')[-1]} function-preservation "
                           f"gate", cdt, rel_bound=1e-6, hard=True)
    else:  # qerr
        # The shift toward the bf16 model is expected and small; a blown
        # scale/orientation shows up as a loss excursion orders bigger.
        all_ok = tier("fp32 step-0 sanity", torch.float32,
                      rel_bound=0.5, hard=True)
        if cdt in (torch.float16, torch.bfloat16):
            all_ok &= tier(f"{str(cdt).split('.')[-1]} step-0 sanity", cdt,
                           rel_bound=0.5, hard=False)

    print(f"  init-lora {mode} gate:", "PASS" if all_ok else
          f"FAIL -- do NOT train with --init-lora {mode}")
    return all_ok


def check_dequant_parity(model, tokenizer, prompt, device, cdt, attn_impl="auto"):
    """
    Gate the fast dequant path (audit A1): EXL3LoRAHadFunction reconstructs
    only the inner trellis weight and applies the Hadamard/sign transforms to
    the ACTIVATIONS (the inference reconstruct_hgemm math, fp16 base matmul),
    where the legacy path materializes the fully-transformed weight per call.
    Same function, different arithmetic path -- so one seeded adapter net,
    forwarded and backwarded under each mode, must agree on logits and adapter
    gradients to within half-precision reassociation noise. A formula error
    (transform order, adjoint, pissa-offset sign) lands orders of magnitude
    outside these bounds. The fast backward also runs under the recompute->
    backward weight cache, so a cache-corruption bug would surface here too.
    """
    print("\n" + "-" * 78)
    print("dequant parity: fast (activation-side transforms) vs legacy closures")
    from exllamav3.training import backbone as _bb

    ids = tokenizer.encode(prompt, add_bos=True).to(device)

    def build(pissa):
        torch.manual_seed(0)
        try:
            net = NativeLlamaQLoRA(
                model, r=8, alpha=16.0,
                target_modules=["q_proj", "gate_proj", "up_proj", "down_proj"],
                compute_dtype=cdt, gradient_checkpointing=True,
                attn_impl=attn_impl)
        except ValueError:
            torch.manual_seed(0)
            net = NativeLlamaQLoRA(
                model, r=8, alpha=16.0, target_modules=["q_proj"],
                compute_dtype=cdt, gradient_checkpointing=True,
                attn_impl=attn_impl)
        if pissa:
            # Same seed both modes -> identical randomized-SVD factors (the
            # init reads weights through the mode-independent legacy closure).
            torch.manual_seed(1)
            net.apply_init_lora("pissa")
        return net

    def run(mode, cache, pissa):
        _bb.set_dequant_mode(mode)
        net = build(pissa)
        net.train()
        with torch.no_grad():
            logits = net.logits(ids).float().cpu()
        loss = net.compute_loss(ids, ids.clone())
        with _bb.backward_dequant_cache(enable=cache):
            loss.backward()
        grads = {f"{w.key}.{n}": p.grad.detach().float().cpu()
                 for w in net._wrappers if w.r > 0
                 for n, p in (("a", w.lora_a), ("b", w.lora_b))
                 if p.grad is not None}
        del net
        return logits, loss.item(), grads

    def compare(label, pissa, agree_min, cos_min, rel_max, med_rel_max,
                is_moe=False):
        prev_mode = _bb.dequant_mode()
        try:
            logits_f, loss_f, g_f = run("fast", cache=True, pissa=pissa)
            logits_l, loss_l, g_l = run("legacy", cache=False, pissa=pissa)
        finally:
            _bb.set_dequant_mode(prev_mode)

        agree = (logits_f.argmax(-1) == logits_l.argmax(-1)).float().mean().item()
        cos_last = torch.cosine_similarity(
            logits_f[0, -1], logits_l[0, -1], dim=0).item()
        loss_rel = abs(loss_f - loss_l) / max(abs(loss_f), 1e-9)
        ok = agree >= agree_min and cos_last >= 0.999 and loss_rel < 2e-2

        stats = []
        for key in sorted(set(g_f) & set(g_l)):
            gf, gl = g_f[key], g_l[key]
            nf, nl = gf.norm().item(), gl.norm().item()
            if nf == 0.0 and nl == 0.0:
                continue                   # lora_a at step 1 (B==0, no pissa)
            cos = torch.cosine_similarity(
                gf.flatten(), gl.flatten(), dim=0).item()
            rel = (gf - gl).norm().item() / max(nf, 1e-12)
            stats.append((cos, rel, key))
        fails = [s for s in stats if not (s[0] > cos_min and s[1] < rel_max)]
        med_rel = sorted(s[1] for s in stats)[len(stats) // 2] if stats else 1.0
        ok &= not fails and len(stats) > 0 and med_rel < med_rel_max

        print(f"  [{label}]")
        print(f"    logits: per-position argmax agreement {agree*100:.1f}%, "
              f"last-token cos {cos_last:.6f}")
        print(f"    loss: fast {loss_f:.6f} vs legacy {loss_l:.6f} "
              f"(rel {loss_rel:.2e})")
        by_cos = sorted(stats)
        print(f"    {len(stats)} adapter grads compared, {len(fails)} outside "
              f"tolerance (cos > {cos_min}, rel < {rel_max}); "
              f"median rel {med_rel:.3f} (bound {med_rel_max})")
        if by_cos:
            med = by_cos[len(by_cos) // 2]
            print(f"    median cosine: {med[0]:.6f}   rel {med[1]:.2e}")
            for c, r, k in by_cos[:3]:
                print(f"      {c:.6f}  rel {r:.2e}  {k}")
        print(f"    {label}:", "PASS" if ok else "FAIL")
        if not ok and is_moe:
            print("    NOTE (MoE): fast-vs-legacy fp noise can flip top-k "
                  "EXPERT SELECTION per token, so the two arms genuinely "
                  "compute different downstream values and grad parity is "
                  "expected to fail here even with correct math (the same "
                  "routing-tie phenomenon as the argmax gate; worst on "
                  "many-expert sigmoid routers). Read the forward gate + a "
                  "short fast-vs-legacy training A/B (loss + |dB| curves) "
                  "as the real gate on MoE models.")
        return ok

    # Round 1: default-init adapters (B=0 -- the pure base + LoRA-grad path).
    # Round 2: pissa-initialized adapters -- exercises the fast path's
    # activation-side offset term (and nonzero-A/B grads) against the legacy
    # folded-into-the-weight offset. Its bounds are LOOSER and shaped
    # direction-strict / amplitude-tolerant, calibrated on Gemma4-12B (48
    # layers, fp32, Session 30): the pissa cancellation residue is re-rounded
    # by every fp16 base matmul, and in an amplitude-sensitive band of layers
    # (10-14, softcapped attention) that shifts grad MAGNITUDE up to ~1.5x
    # while direction stays intact (offenders all cos >= 0.94) -- the same
    # shift legacy-bf16 shows vs legacy-fp32, i.e. inside the noise band bf16
    # training already accepts (Adam's per-param normalization absorbs
    # amplitude noise; an 8-step 12B training A/B at the recipe matched loss
    # to 2-3 decimals and |dB| to 3 decimals every step). So: every grad must
    # AGREE IN DIRECTION (cos) and the median amplitude error stays small,
    # while individual amplitude outliers pass. A formula bug (offset sign /
    # scale / orientation) breaks cos and the medians by orders of magnitude.
    _is_moe = any(_bb.is_block_sparse_mlp(getattr(m, "mlp", None))
                  for m in model.modules)
    all_ok = compare("default init", pissa=False,
                     agree_min=0.98, cos_min=0.98, rel_max=0.25,
                     med_rel_max=0.10, is_moe=_is_moe)
    all_ok &= compare("pissa init + offset", pissa=True,
                      agree_min=0.93, cos_min=0.90, rel_max=1.0,
                      med_rel_max=0.25, is_moe=_is_moe)

    print("  dequant parity:", "PASS" if all_ok else
          "FAIL -- do NOT train with --dequant-mode fast")
    return all_ok


def check_packing(net, tokenizer, prompts, device):
    """
    Prove sample packing isolates documents: the logits for each document inside a
    packed block must match running that document ALONE. This exercises both
    halves of correct packing -- the block-diagonal attention (no token attends
    across a document boundary) and the per-document RoPE position reset. A
    mismatch means packed training would silently mix documents.

    Runs on whatever dtype/attn the net was built with, so it covers the fp32
    eager reference and (under --compute-dtype bfloat16) the flash-varlen path.
    """
    print("\n" + "-" * 78)
    print("packing check: packed-block logits == per-document logits")
    docs = [tokenizer.encode(p, add_bos=True)[0].tolist() for p in prompts]

    # Per-document reference: each document forwarded on its own.
    ref = []
    with torch.no_grad():
        for d in docs:
            ref.append(net.logits(torch.tensor([d], device=device))[0].float().cpu())

    # Pack the documents into one sequence with seg ids + per-document position
    # resets (exactly what pack_examples/collate produce for training).
    input_ids, seg_ids, position_ids = [], [], []
    for s, d in enumerate(docs):
        input_ids += d
        seg_ids += [s] * len(d)
        position_ids += list(range(len(d)))
    ii = torch.tensor([input_ids], device=device)
    sg = torch.tensor([seg_ids], device=device)
    pp = torch.tensor([position_ids], device=device)
    with torch.no_grad():
        packed = net.logits(ii, position_ids=pp, seg_ids=sg)[0].float().cpu()

    ok, off = True, 0
    for s, (d, r) in enumerate(zip(docs, ref)):
        sl = packed[off: off + len(d)]
        off += len(d)
        agree = (sl.argmax(-1) == r.argmax(-1)).float().mean().item()
        max_abs = (sl - r).abs().max().item()
        cos = torch.cosine_similarity(sl[-1], r[-1], dim=0).item()
        good = agree > 0.999
        ok &= good
        print(f"  doc {s} ({len(d):>3} tok): per-position argmax {agree*100:5.1f}% | "
              f"max|Δ|={max_abs:.4f} cos={cos:.6f}  {'OK' if good else 'MISMATCH'}")
    print("  packing check:", "PASS" if ok else "FAIL")
    return ok


def _fp16_ulp_key(t):
    """Map fp16 bit patterns to a monotonically ordered int32 key (sign-magnitude
    -> lexicographic), so adjacent representable values differ by exactly 1."""
    u = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)


def check_head_slice(net):
    """
    Gate the chunked-vocab head (--head-vocab-chunk): assert the column-sliced head
    reconstruction equals the matching slice of the full reconstruction to within
    1 fp16 ulp. Not bit-for-bit: the Hadamard pre-applies are fp32 GEMMs whose
    width differs between the paths (full vocab vs one chunk), so cuBLAS kernel
    selection can shift the accumulation order and flip the final fp16 rounding by
    one ulp. A real slicing bug (wrong had_n block, misindexed sv column) is on
    the order of the weights themselves -- thousands of ulps -- so the 1-ulp gate
    still catches it loudly. This is the GPU-only half the CPU gradcheck
    (tests/test_fused_ce.py) can't cover. Uses the same backbone seam the trainer
    uses. SKIPs when the head can't slice (e.g. an unquantized head).
    """
    print("\n" + "-" * 78)
    print("head-slice check: get_weight_tensor_slice == full reconstruction")
    inner = net.lm_head.inner
    if getattr(inner, "get_weight_tensor_slice", None) is None:
        print("  SKIP -- this head does not support sliced reconstruction")
        return True
    sl = backbone.head_weight_slice_closure(net.lm_head)
    slice_fn, vocab, gran = sl
    full = backbone.head_weight_closure(net.lm_head)()        # [d, V]
    chunk = max(gran, (min(32768, vocab) // gran) * gran)
    # First, a middle, and the last aligned chunk -- exercise n_start=0, an interior
    # offset, and the trailing slice.
    starts = sorted({0,
                     max(0, ((vocab // 2) // gran) * gran),
                     max(0, vocab - chunk)})
    max_d, max_ulp = 0.0, 0
    for a in starts:
        a = min(a, vocab - chunk)
        b = a + chunk
        s = slice_fn(a, b - a).to(full.dtype).to(full.device)
        ref = full[:, a:b]
        max_d = max(max_d, (ref - s).abs().max().item())
        if full.dtype == torch.float16:
            ulp = (_fp16_ulp_key(ref) - _fp16_ulp_key(s)).abs().max().item()
        else:
            # Non-fp16 heads reach here via the generic fallback, which indexes
            # the one resident weight tensor -- exact equality is fair there.
            ulp = 0 if torch.equal(ref, s) else 2
        max_ulp = max(max_ulp, ulp)
    del full
    ok = max_ulp <= 1
    print(f"  vocab={vocab}, granularity={gran}, chunk={chunk}, "
          f"slices@{starts}")
    print(f"  max|full[:, a:b] - slice(a, b)| = {max_d:.3e}  "
          f"(max {max_ulp} ulp, tolerance 1)")
    print("  head-slice check:",
          "PASS" if ok else "FAIL (sliced head != full head beyond 1 fp16 ulp)")
    return ok


def check_mtp(model, tokenizer, config, prompts, device, cdt, attn_impl="auto"):
    """MTP head parity: the native head forward (``NativeLlamaQLoRA._forward_mtp``
    -- the shifted trunk state + token embedding through the pre-fc norms, fc,
    the head's block(s) and final norm, then the borrowed LM head) against the
    inference draft path (the trunk's exported post-final-norm state, shifted
    exactly as the generator's prefill wiring does, through the MTP component
    model's own forward). Position ``i`` of both is the head's draft for token
    ``i+1``; the same top-1 / argmax-agreement / cosine gate as the trunk check.

    Run this before spending a run on --mtp-targets: the head's forward is the
    one piece of the MTP path the trunk gate does not cover."""
    from exllamav3.model.model import Model as _Model
    print("\n" + "=" * 78)
    print("MTP head parity (native draft forward vs inference draft path)")
    print("=" * 78)
    draft = _Model.from_config(config, component="mtp")
    draft.load(device=str(model.output_device) if getattr(model, "output_device", None)
               is not None else device, progressbar=True)
    draft.attach_to(model)      # borrows embed / lm_head, sets the export key
    # Adapter-free head: pure frozen forward for parity (mtp_targets=[] wraps
    # every head linear at r=0, like target_modules=[] for the trunk).
    net = NativeLlamaQLoRA(model, target_modules=[], compute_dtype=cdt,
                           gradient_checkpointing=False, attn_impl=attn_impl,
                           mtp_model=draft, mtp_targets=[])
    net.eval()
    lm = model.modules[-1]
    ok_all = True
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True).to(device)
        with torch.inference_mode():
            # Trunk forward with the final norm's output exported -- the very
            # tensor the generator hands the head as target_hidden.
            params = dict(draft.draft_verifier_params)
            model.forward(ids, params)
            state = params["export_states"][-1]                    # [1, t, d] half
            # The generator's prefill shift: position i sees the state that
            # produced token i (i-1), zero carry at the first position.
            shifted = torch.cat((torch.zeros_like(state[:, :1, :]), state[:, :-1, :]), dim=1)
            mtp_state = draft.forward(ids, {"target_hidden": shifted})   # [1, t, d]
            logits_native = lm.forward(lm.prepare_for_device(mtp_state, {}), {}).float()
        with torch.no_grad():
            logits_diff = net.mtp_logits(ids).float()
        logits_diff = logits_diff.to(logits_native.device)
        V = min(logits_native.shape[-1], logits_diff.shape[-1])    # padded vocab
        ln, ld = logits_native[0, :, :V], logits_diff[0, :, :V]
        top1_native, top1_diff = int(ln[-1].argmax()), int(ld[-1].argmax())
        match = top1_native == top1_diff
        max_abs = (ln[-1] - ld[-1]).abs().max().item()
        cos = torch.cosine_similarity(ln[-1], ld[-1], dim=0).item()
        agree = (ln.argmax(-1) == ld.argmax(-1)).float().mean().item()
        is_lowp = cdt in (torch.float16, torch.bfloat16)
        ok = match or (is_lowp and agree >= 0.8 and cos >= 0.999)
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {prompt!r}: draft top-1 native="
              f"{tokenizer.decode(torch.tensor([top1_native]))!r} diff="
              f"{tokenizer.decode(torch.tensor([top1_diff]))!r} | argmax agree "
              f"{agree:.3f} | cos {cos:.5f} | max|dlogit| {max_abs:.3f}")
    print(f"[mtp] head parity {'PASSED' if ok_all else 'FAILED'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0",
                    help="single-device load target (ignored when --parallel split)")
    ap.add_argument("--parallel", choices=["single", "split"], default="single",
                    help="single: load to --device; split: layer-autosplit across visible GPUs")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB to reserve per device; negative excludes a device")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB budget per device; caps a card to force a split on a small model")
    ap.add_argument("--compute-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="dtype for the differentiable linears (float32 = closest to true math)")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="auto/eager/flash. NOTE flash needs CUDA fp16/bf16, so the "
                         "default float32 validate runs eager regardless; pass "
                         "--compute-dtype bfloat16 to exercise/validate the flash path.")
    ap.add_argument("--use-liger", action="store_true",
                    help="Validate the Liger RMSNorm/SwiGLU path (needs liger-kernel). "
                         "Runs the two-tier torch-vs-liger parity gate automatically: "
                         "an fp32 math gate (near-exact or the backward formula is "
                         "wrong) plus, under --compute-dtype bfloat16/float16, a "
                         "noise-band gate at the training dtype. REQUIRED green "
                         "before any --use-liger training run.")
    ap.add_argument("--init-lora", choices=["pissa", "qerr", "eva"], default=None,
                    help="Run the step-0 gate for an SVD adapter init: pissa "
                         "must be function-preserving vs the base model "
                         "(fp32 near-exact), qerr must land within a sane "
                         "step-0 loss shift, eva must be EXACTLY function-"
                         "preserving (B=0) after its activation pre-pass. "
                         "REQUIRED green before any --init-lora training run.")
    ap.add_argument("--init-ref-model", default=None,
                    help="Original (unquantized) HF model dir for --init-lora qerr.")
    ap.add_argument("--init-svd-niter", type=int, default=16,
                    help="Randomized-SVD iterations for the init gate (0 = exact SVD).")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--image", default=None, metavar="PATH",
                    help="Image+text gate for --vision training: run the model's "
                         "vision component on this image and compare the "
                         "differentiable forward (features spliced through the "
                         "training path, 3-D mRoPE / deepstack / bidirectional "
                         "spans as the arch needs) against the native multimodal "
                         "forward. REQUIRED green before any --vision run on a base.")
    ap.add_argument("--image-prompt", default="Describe the image.",
                    help="text that follows the image in the --image gate prompt")
    ap.add_argument("--check-backward", action="store_true",
                    help="also smoke-test cross-device gradient flow (tiny adapter + backward)")
    ap.add_argument("--check-mtp", action="store_true",
                    help="Also load the model's MTP head (Qwen3.5/3.6) and gate "
                         "the native draft forward against the inference draft "
                         "path. Run before any --mtp-targets training run.")
    ap.add_argument("--check-packing", action="store_true",
                    help="also verify sample packing: a packed block's per-document "
                         "logits must match running each document alone (block-"
                         "diagonal attention + per-document RoPE reset). Run with "
                         "--compute-dtype bfloat16 to exercise the flash-varlen path.")
    ap.add_argument("--skip-head-slice-check", action="store_true",
                    help="skip the chunked-vocab head equality check (it gates "
                         "--head-vocab-chunk; runs by default when the head can slice)")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast",
                    help="Dequant path for the main forward compare (matches the "
                         "trainer flag). The fast-vs-legacy parity gate below runs "
                         "regardless (skip with --skip-dequant-parity).")
    ap.add_argument("--skip-dequant-parity", action="store_true",
                    help="skip the fast-vs-legacy dequant parity gate (forward + "
                         "adapter-grad compare under both modes)")
    args = ap.parse_args()

    backbone.set_dequant_mode(args.dequant_mode)

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]
    prompts = args.prompts or DEFAULT_PROMPTS

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    if args.parallel == "split":
        load_kwargs = {}
        if args.reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = args.reserve_per_device
        if args.use_per_device is not None:
            load_kwargs["use_per_device"] = args.use_per_device
        model.load(progressbar=True, **load_kwargs)
        print(f" -- layer-autosplit: active devices {model.active_devices}, "
              f"output device {model.output_device}")
    else:
        model.load(device=args.device, progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    # No adapters: a pure frozen forward, directly comparable to native inference.
    net = NativeLlamaQLoRA(model, target_modules=[], compute_dtype=cdt,
                           gradient_checkpointing=False, attn_impl=args.attn_impl,
                           use_liger=args.use_liger)
    net.eval()
    print(f" -- {net.describe_attn()}")

    dist = Counter(str(d) for d in net._block_devices)
    print(f" -- decoder block devices: {dict(dist)}  (final norm + head on {net.device})")

    print("=" * 78)
    print(f"Validating differentiable forward (compute_dtype={args.compute_dtype}) "
          f"vs native exllamav3")
    print("=" * 78)

    all_ok = True
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True).to(args.device)

        # Native (correct) forward -- runs under inference_mode, fp16 kernels.
        with torch.inference_mode():
            logits_native = model.forward(ids).float()       # [1, t, V]

        # Differentiable forward.
        with torch.no_grad():
            logits_diff = net.logits(ids).float()            # [1, t, V]

        # Native output lands on the model's output device; co-locate for compare.
        logits_diff = logits_diff.to(logits_native.device)

        ln = logits_native[0, -1]
        ld = logits_diff[0, -1]
        top1_native = int(ln.argmax())
        top1_diff = int(ld.argmax())
        match = top1_native == top1_diff

        # Cross-entropy of the *native* next-token prediction under each model,
        # and agreement metrics on the final-token logits.
        max_abs = (ln - ld).abs().max().item()
        cos = torch.cosine_similarity(ln, ld, dim=0).item()
        # How often do the two forwards agree on the argmax across all positions?
        agree = (logits_native[0].argmax(-1) == logits_diff[0].argmax(-1)).float().mean().item()

        # Pass criterion: fp32 is the strict correctness gate (top-1 must match). In
        # fp16/bf16 the forward is inherently looser -- a borderline top-1 can flip even
        # at cos ~0.9999 (rounding; the fp32-math big-head SDPA path on Gemma flips both
        # ways vs native, independent of Liger). So in low precision, accept a flip when
        # the per-position argmax agreement AND last-token cosine stay high; this stops a
        # single noise-flip reading as FAIL while still catching real drift (a genuinely
        # broken forward shows low agreement / low cosine and still fails).
        is_lowp = cdt in (torch.float16, torch.bfloat16)
        ok = match or (is_lowp and agree >= 0.8 and cos >= 0.999)
        all_ok &= ok

        tok_native = repr(tokenizer.decode(torch.tensor([[top1_native]]),
                                           decode_special_tokens=True)[0])
        tok_diff = repr(tokenizer.decode(torch.tensor([[top1_diff]]),
                                         decode_special_tokens=True)[0])

        status = "OK" if match else (
            "MISMATCH (tolerated: low-precision noise, cos/agree high)" if ok
            else "MISMATCH")
        print(f"\nprompt: {prompt!r}")
        print(f"  native next-token : {tok_native}")
        print(f"  diff   next-token : {tok_diff}   {status}")
        print(f"  per-position argmax agreement: {agree*100:.1f}%")
        print(f"  last-token logits: max|Δ|={max_abs:.4f}  cos={cos:.6f}")

    if args.image:
        all_ok &= check_vision(model, net, tokenizer, config, args.image,
                               args.image_prompt, args.device, cdt)

    if not args.skip_head_slice_check:
        all_ok &= check_head_slice(net)

    if not args.skip_dequant_parity:
        all_ok &= check_dequant_parity(model, tokenizer, " ".join(prompts),
                                       args.device, cdt,
                                       attn_impl=args.attn_impl)

    if args.check_backward:
        all_ok &= check_backward(model, tokenizer, prompts[0], args.device, cdt,
                                 attn_impl=args.attn_impl, use_liger=args.use_liger)

    if args.use_liger:
        # The grad-parity gate (Session 10 #3): always runs with --use-liger --
        # a smoke test alone cannot catch a wrong-value gradient (#119).
        all_ok &= check_liger_parity(model, tokenizer, " ".join(prompts),
                                     args.device, cdt, attn_impl=args.attn_impl)

    if args.init_lora:
        all_ok &= check_init_lora(model, tokenizer, " ".join(prompts),
                                  args.device, cdt, args.init_lora,
                                  ref_model_dir=args.init_ref_model,
                                  svd_niter=args.init_svd_niter,
                                  attn_impl=args.attn_impl)

    if args.check_mtp:
        if "mtp" not in config.model_classes:
            print(f"\n -- --check-mtp: {config.architecture} defines no MTP head "
                  f"(or the checkpoint carries none); FAIL")
            all_ok = False
        else:
            all_ok &= check_mtp(model, tokenizer, config, prompts, args.device, cdt,
                                attn_impl=args.attn_impl)

    if args.check_packing:
        if getattr(net, "has_gdn", False) or getattr(net, "has_shortconv", False):
            print("\n -- skipping packing check: sample packing is not supported "
                  "on GatedDeltaNet / ShortConv models (train unpacked).")
        else:
            all_ok &= check_packing(net, tokenizer, prompts, args.device)

    print("\n" + "=" * 78)
    print("RESULT:", "PASS -- differentiable forward matches native"
          if all_ok else "FAIL -- see above")
    print("=" * 78)
    # Exit non-zero on failure so a `validate && train` kickoff aborts the run
    # instead of training against a broken forward (e.g. a new architecture).
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
