"""
Energy-Based Fine-Tuning (EBFT) of an EXL3 model, native path -- no
HuggingFace Transformers in the loop.

Trains LoRA adapters on a frozen EXL3 base with the feature-matching
policy-gradient objective from "Matching Features, Not Tokens: Energy-Based
Fine-Tuning of Language Models" (Jelassi et al., arXiv:2603.12248), using the
same transformers-free differentiable forward as qlora_train_native.py. The
frozen FEATURE NETWORK phi is the base model itself, obtained by running the
same net under ``adapters_disabled()`` (the DPO/KTO reference trick) -- with
--init-lora default/pissa/eva the step-0 policy IS phi exactly, which is the
paper's setup (phi = generator at init), at zero extra VRAM.

One step, per micro-batch of --batch sequences:
  1. pick --anchors positions per sequence inside the supervised span; the G
     ground-truth tokens after each anchor are that group's target window;
  2. sample --n-samples on-policy rollouts of G tokens per anchor with the
     EXACT sampler (no-grad differentiable forward -- no KV cache, but zero
     sampling/scoring mismatch; the paper's G is 8, so this stays cheap);
  3. embed rollout and ground-truth windows with phi: residual stream at
     ~25/50/75% depth blocks, last window token, concat + joint L2 norm
     (reference ``hidden_state_method=concat``, ``embed_method=last_token``);
  4. whitened feature-matching reward + corrected RLOO baseline
     (exllamav3/training/ebft.py; reference-faithful, see its docstring);
  5. REINFORCE on the per-token-MEAN completion logps times the advantage,
     plus --ce-coef * standard CE on the ground-truth sequences:
         loss = rl_coef * mean_rows(-adv * logp_mean) + ce_coef * CE.

Anchors-as-rows replaces the reference's strided block-parallel rollouts
(Quiet-STaR custom attention mask): same math per (context, window) group,
less prefix amortization. Fine at paper scale (1-8B, G=8); the strided mask
is the planned throughput upgrade if this validates.

Data modes:
  --mode qa        instruction data via the chat template (build_sft_examples;
                   same keys as the SFT trainer). Anchors live in the RESPONSE
                   span only (the reference's qa_masking).
  --mode pretrain  raw text packed into --seq-len blocks (build_lm_examples,
                   --text-key). Anchors live anywhere past --min-context.

Usage (validation-scale run, Qwen2.5-1.5B / Llama-3.2-1B class):
    python training/qlora_train_ebft.py \
        --model /path/to/exl3_model --out out/exl3_ebft_adapter \
        --dataset <hf-id-or-path> --mode qa \
        --gen-len 8 --n-samples 4 --anchors 4 --temperature 0.6 \
        --lr 1e-5 --epochs 1 --eval-every 50 --val-frac 0.02

Notes vs the SFT trainer:
  * Single-process only (--parallel single|split); frozen LM head only (the
    logps path), no --train-head/--train-embeddings/packing.
  * The paper's full-FT actor LR is 1e-6; LoRA wants more -- 1e-5..1e-4 is
    the expected range, unvalidated (nobody has run EBFT+LoRA before).
  * MoE: routed-expert LoRA trains fine here (rollouts use the differentiable
    forward, not the fused inference kernels), but keep --targets non-expert
    for first validation runs.

Verify afterwards on the native inference path:
    python training/qlora_infer_native.py --model <model> --adapter <out>
"""

import argparse
import datetime
import os
import random
import sys
import time
import torch
import torch.nn.functional as F

# Reuse the SFT trainer's helpers (same dir on sys.path when run as a script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import (  # noqa: E402
    ThroughputMeter, StepTimer, append_run_log,
    checkpoint_dir, prune_checkpoints,
    save_trainer_state, load_trainer_state, restore_optimizer_state,
    snapshot_datasets, resolve_resumed_datasets,
    build_sft_examples, build_lm_examples,
    build_optimizer, make_lr_scheduler, resolve_steps_and_warmup,
    compute_even_split_budgets,
    format_prompt_and_eot, sample,
    _FAIL_CTX, _log_failure, _REPORT, _finish_report,
)
from run_report import (RunLogger, decode_example_docs,  # noqa: E402
                        start_live_monitor)

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer  # noqa: E402
from exllamav3.generator.sampler import ComboSampler  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402
from exllamav3.training.ebft import ebft_rewards, sample_rollouts  # noqa: E402


# ---------------------------------------------------------------------------
# Anchors and batch assembly
# ---------------------------------------------------------------------------

def sup_start(labels):
    """Index of the first supervised (label != -100) position, or None."""
    for i, l in enumerate(labels):
        if l != -100:
            return i
    return None


def pick_anchors(example, gen_len, n_anchors, min_context, rng):
    """Anchor positions for one example: generation starts AT the anchor, the
    ground-truth window is input_ids[a : a+G]. Valid anchors keep the window
    inside the supervised span (qa: response-only; pretrain: everything) and
    at least ``min_context`` context tokens. Sampled with replacement when the
    valid span is shorter than requested (duplicate groups are harmless --
    their rollouts differ), never yields fewer than 1 anchor unless the
    example has no room at all."""
    ids, labels = example["input_ids"], example["labels"]
    s = sup_start(labels)
    if s is None:
        return []
    lo = max(s, min_context, 1)
    hi = len(ids) - gen_len          # anchor a uses window [a, a+G)
    if hi < lo:
        return []
    span = hi - lo + 1
    if span >= n_anchors:
        return sorted(rng.sample(range(lo, hi + 1), n_anchors))
    return sorted(rng.choices(range(lo, hi + 1), k=n_anchors))


def pad_rows(rows, pad_id):
    """List of 1-D long lists/tensors -> right-padded [R, L] + lens [R]."""
    lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
    maxlen = int(lens.max())
    out = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        t = torch.as_tensor(r, dtype=torch.long)
        out[i, :len(t)] = t
    return out, lens


def extract_features(net, input_ids, attn, positions, blocks):
    """phi features at ``positions`` ([b, P]): residual stream at the tap
    blocks, gathered, concatenated across taps, jointly L2-normalized.
    Caller wraps in no_grad + adapters_disabled. Returns [b, P, 3d] fp32 on
    net.device."""
    with net.collect_hidden(blocks, positions=positions):
        net.forward(input_ids, attention_mask=attn)
        taps = [net.collected[b].to(net.device) for b in blocks]
    return F.normalize(torch.cat(taps, dim=-1), p=2, dim=-1)


def sample_rollouts_native(gen, ctx_rows, gen_len, args, pad_id, seed):
    """KV-cached on-policy rollouts through the native inference path.

    The caller must have pushed the current adapter into the runtime LoRA slots
    (``net.apply_to_native()``) first, so the generator samples the CURRENT
    policy -- modulo inference-kernel numerics (fused fp16 kernels vs the
    training forward's dequant matmul): the classic RLHF sampler/trainer split,
    minus the weight staleness. Scoring stays on the differentiable path
    (``compute_logps``), same as the exact sampler.

    Speed vs ``sample_rollouts``: one paged prefill + gen_len single-token
    decode steps instead of gen_len full-prefix re-forwards -- and the paged
    cache dedups the shared context across each anchor's n sibling rollouts,
    so the context prefills once per ANCHOR, not once per row.

    NOT valid for routed-expert LoRA (fused MoE kernels bypass the runtime
    LoRA slots -> rollouts would come from the base experts, off-policy); the
    trainer refuses that combination at startup.

    Returns the same ``(rows [R, L'], lens [R])`` contract as the exact
    sampler: right-padded context + exactly gen_len sampled tokens per row.
    """
    assert args.temperature > 0.0, \
        "greedy rollouts collapse the RLOO baseline; use temperature > 0"
    # Same stack order as the exact sampler: temperature -> top_k -> top_p ->
    # sample (min_p 0 is a no-op step). No stop conditions: a sampled EOS is
    # just a token, and every row must return exactly gen_len tokens.
    sampler = ComboSampler(temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p)
    ctx_lists = [torch.as_tensor(c, dtype=torch.long).tolist()
                 for c in ctx_rows]
    job_idx = {}
    for i, ctx in enumerate(ctx_lists):
        # Job counts the sample taken at the prefill position toward its
        # budget (job.py: max_new_tokens - 1), so N yields N-1 sampled tokens;
        # ask for gen_len + 1. The exact-count assert below guards this.
        job = Job(input_ids=torch.tensor([ctx], dtype=torch.long),
                  max_new_tokens=gen_len + 1, sampler=sampler, seed=seed + i)
        job_idx[job] = i
        gen.enqueue(job)
    out = [[] for _ in ctx_lists]
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            i = job_idx.get(r["job"])
            if i is None:
                continue
            # Tokens stream across events (and the eos event may carry a held
            # remainder); collect from both spots.
            for blob in (r, r.get("held") or {}):
                t = blob.get("token_ids")
                if t is not None:
                    out[i] += t.flatten().tolist()
    for i, toks in enumerate(out):
        assert len(toks) == gen_len, \
            f"native rollout {i}: {len(toks)} tokens, wanted {gen_len}"
    return pad_rows([ctx + out[i] for i, ctx in enumerate(ctx_lists)], pad_id)


def ebft_batch(net, batch, pad_id, args, rng, blocks, want_rollouts=False,
               gen=None):
    """Loss + metrics for one micro-batch of examples.

    Returns (loss, metrics) like the other trainers' batch fns; loss is
    rl_coef * REINFORCE + ce_coef * CE (CE always computed -- it shares no
    forward with the RL arm here, but it is the cheap part). Metrics carry the
    reward/calibration diagnostics. ``want_rollouts`` additionally returns
    decoded-friendly rollout tokens for --inspect-style debugging."""
    G, n, A = args.gen_len, args.n_samples, args.anchors

    # 1. Anchors. Drop examples with no room (short completions).
    groups = []                      # (ex_idx, anchor)
    per_ex_anchors = []
    for bi, ex in enumerate(batch):
        anchors = pick_anchors(ex, G, A, args.min_context, rng)
        per_ex_anchors.append(anchors)
        groups += [(bi, a) for a in anchors]
    if not groups:
        return None, {}

    # 2. On-policy rollouts: n rows per group, context = ids[:anchor].
    ctx_rows = []
    for bi, a in groups:
        ctx_rows += [batch[bi]["input_ids"][:a]] * n
    ctx_ids, ctx_lens = pad_rows(ctx_rows, pad_id)
    if gen is not None:
        # Native KV-cached sampler: install the current adapter in the runtime
        # LoRA slots for the duration of the sampling call. try/finally so a
        # sampler error can't leave stale weights in the inference path.
        net.apply_to_native()
        try:
            rows, row_lens = sample_rollouts_native(
                gen, ctx_rows, G, args, pad_id, seed=rng.randrange(1 << 31))
        finally:
            net.remove_from_native()
    else:
        rows, row_lens = sample_rollouts(
            net, ctx_ids, ctx_lens, G, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p)

    # 3. phi features (frozen base = feature network).
    net.eval()
    with torch.no_grad(), net.adapters_disabled():
        # Ground-truth windows: ONE forward over the original sequences gives
        # every anchor's window-end feature (full causal context, exactly the
        # reference's gt path).
        seq_ids, seq_lens = pad_rows([ex["input_ids"] for ex in batch], pad_id)
        seq_attn = (torch.arange(seq_ids.shape[1]).unsqueeze(0)
                    < seq_lens.unsqueeze(1)).long()
        # positions per ORIGINAL example row: pad ragged anchor lists with 0
        # (gathered but discarded below via the group list).
        maxA = max(len(a) for a in per_ex_anchors)
        pos = torch.zeros(len(batch), maxA, dtype=torch.long)
        for bi, anchors in enumerate(per_ex_anchors):
            for j, a in enumerate(anchors):
                pos[bi, j] = a + G - 1
        gt_all = extract_features(net, seq_ids, seq_attn, pos, blocks)
        gt_feats = torch.stack([
            gt_all[bi, per_ex_anchors[bi].index(a)] for bi, a in groups])

        # Rollout windows: feature at each row's last sampled token.
        roll_attn = (torch.arange(rows.shape[1]).unsqueeze(0)
                     < row_lens.unsqueeze(1)).long()
        roll_pos = (row_lens - 1).unsqueeze(1)
        gen_feats = extract_features(net, rows, roll_attn, roll_pos, blocks)
        gen_feats = gen_feats.squeeze(1).view(len(groups), n, -1)

        rw = ebft_rewards(gen_feats, gt_feats, whiten=not args.no_whiten,
                          align_coef=args.align_coef, div_coef=args.div_coef,
                          whiten_tol=args.whiten_tol)
        adv = rw["advantage"].reshape(-1)          # [R], detached

        # Duplicate-rollout diagnostic (the whitened diversity term exists to
        # punish exactly these).
        samp = torch.stack([rows[i, ctx_lens[i]:ctx_lens[i] + G]
                            for i in range(rows.shape[0])])
        samp_g = samp.view(len(groups), n, G)
        dup = (samp_g.unsqueeze(1) == samp_g.unsqueeze(2)).all(-1)
        dup_frac = ((dup.sum(-1) > 1).float().mean().item())
    net.train()

    # 4. REINFORCE surrogate on the policy (grad path): per-token-MEAN logps
    # of the sampled window, weighted by the (detached) advantage. Matches the
    # reference EBFTPolicyLoss (masked_mean over generated tokens, then batch
    # mean; PPO ratio == 1 single-update surrogate).
    R = rows.shape[0]
    labels = torch.full_like(rows, -100)
    for i in range(R):
        labels[i, ctx_lens[i]:ctx_lens[i] + G] = rows[i, ctx_lens[i]:ctx_lens[i] + G]
    logps, counts = net.compute_logps(rows, labels, attention_mask=roll_attn,
                                      chunk=args.ce_chunk)
    logp_mean = logps / counts.clamp_min(1)
    rl_loss = -(adv.to(logp_mean.device) * logp_mean).mean()

    # 5. CE regularizer on the ground-truth sequences (the gamma term).
    seq_labels = torch.full_like(seq_ids, -100)
    for bi, ex in enumerate(batch):
        lab = torch.as_tensor(ex["labels"], dtype=torch.long)
        seq_labels[bi, :len(lab)] = lab
    ce_loss = net.compute_loss(seq_ids, seq_labels, attention_mask=seq_attn,
                               chunk=args.ce_chunk)

    loss = args.rl_coef * rl_loss + args.ce_coef * ce_loss
    metrics = {
        "rl": rl_loss.item(), "ce": ce_loss.item(),
        "reward": rw["reward"].mean().item(),
        "align": rw["alignment"].mean().item(),
        "div": rw["diversity"].mean().item(),
        "cfm": rw["cfm"].mean().item(),
        "fmw": rw["fm_whiten"].mean().item(),
        "dup": dup_frac,
        "adv_std": rw["advantage"].std().item(),
        "sup": int(counts.sum()),
        "tot": int(roll_attn.sum()) + int(seq_attn.sum()),
        "n": len(groups),
    }
    if want_rollouts:
        metrics["rollouts"] = (groups, samp_g)
    return loss, metrics


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def reward_self_test():
    """CPU checks of the reward math against hand-derivable cases."""
    torch.manual_seed(0)
    B, n, D = 3, 4, 64
    # (a) Distinct near-orthogonal rollouts: whitened Gram ~ I => diversity ~ 0.
    X = F.normalize(torch.randn(B, n, D), dim=-1)
    y = F.normalize(torch.randn(B, D), dim=-1)
    rw = ebft_rewards(X, y)
    assert rw["diversity"].abs().max() < 1e-3, \
        f"distinct rollouts should have ~0 whitened diversity, got {rw['diversity'].abs().max()}"
    # (b) Exact duplicates: whitened self-similarity fires on the duplicated
    # pair only (Lemma B.2: <phi_j, phi_j'> = 1/n_k for duplicates, 0 else;
    # at the reference CODE's scaling a duplicate pair with n=4 gives
    # DT = 2*(1/2)/(n-1) = 1/3 -- the appendix's eq. 48 values are n x this).
    Xd = X.clone()
    Xd[:, 1] = Xd[:, 0]
    rwd = ebft_rewards(Xd, y)
    assert (rwd["diversity"][:, :2] > 0.3).all(), "duplicates must be penalized"
    assert rwd["diversity"][:, 2:].abs().max() < 1e-3, "non-duplicates must not be"
    # (c) RLOO baseline: alignment part is the leave-one-out mean.
    align, div = rw["alignment"], rw["diversity"]
    b0 = rw["reward"] - rw["advantage"]
    exp_align = (align.sum(1, keepdim=True) - align) / (n - 1)
    exp_div = (div.sum(1, keepdim=True) - 2 * div) / (n - 2)
    exp_b = 1.0 * exp_align - 0.5 * exp_div
    assert torch.allclose(b0, exp_b, atol=1e-5), "RLOO baseline mismatch"
    # (d) Whitening invariant: whitened rollout rows are orthonormal-ish for a
    # full-rank group => alignment magnitude bounded by 2.
    assert (rw["alignment"].abs() <= 2 + 1e-4).all()
    print(" -- reward self-test: OK (diversity gating, RLOO baseline, bounds)")


def model_self_test(net, batch, pad_id, args, blocks, gen=None):
    """Model-dependent checks: tap gather correctness, logp agreement between
    compute_logps and materialized logits, grad isolation. With ``gen`` (a
    native Generator, --rollout-sampler native), additionally checks that the
    inference forward agrees with the training forward and that a full EBFT
    step through the native sampler is finite."""
    ex = batch[0]
    ids = torch.tensor([ex["input_ids"][:min(len(ex["input_ids"]), 64)]])
    attn = torch.ones_like(ids)
    # (a) positions gather == full-stream slice.
    P = torch.tensor([[3, ids.shape[1] - 1]])
    with torch.no_grad(), net.adapters_disabled():
        with net.collect_hidden(blocks, positions=P):
            net.forward(ids, attention_mask=attn)
            got = {b: net.collected[b] for b in blocks}
        with net.collect_hidden(blocks):
            net.forward(ids, attention_mask=attn)
            full = {b: net.collected[b] for b in blocks}
    for b in blocks:
        want = full[b][:, P[0]]
        assert torch.allclose(got[b], want.to(got[b].device), atol=1e-5), \
            f"tap position-gather mismatch at block {b}"
    # (b) compute_logps == gather(log_softmax(logits)) on a short row.
    labels = ids.clone()
    labels[:, :ids.shape[1] // 2] = -100
    lp, cnt = net.compute_logps(ids, labels, attention_mask=attn)
    logits = net.logits(ids, attention_mask=attn).float()
    lsm = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = labels[:, 1:].to(lsm.device)
    mask = tgt != -100
    ref = lsm.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ref = (ref * mask).sum()
    assert torch.allclose(lp.sum().to(ref.device), ref, atol=0.05), \
        f"logp mismatch: compute_logps {lp.sum().item():.4f} vs logits {ref.item():.4f}"
    # (c) grad isolation: one EBFT loss backward touches ONLY adapter params.
    rng = random.Random(0)
    loss, m = ebft_batch(net, batch, pad_id, args, rng, blocks)
    assert loss is not None and torch.isfinite(loss), "EBFT loss not finite"
    loss.backward()
    grads = [p for p in net.trainable_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no adapter gradients after EBFT backward"
    net.zero_grad(set_to_none=True)
    print(f" -- model self-test: OK (taps, logps, grads on {len(grads)} adapter tensors; "
          f"step metrics: reward {m['reward']:+.4f}, cfm {m['cfm']:.4f}, "
          f"dup {m['dup']:.2f})")
    if gen is None:
        return
    # (d) native-vs-training forward agreement at the next-token position: the
    # policy the native sampler draws from should match the policy the
    # REINFORCE gradient is computed under, up to inference-kernel numerics
    # (fused fp16 kernels vs dequant matmul). Reported, and loosely asserted:
    # either the argmax tokens agree or the top-20 logprob gap is small.
    ctx = torch.as_tensor(ex["input_ids"][:48], dtype=torch.long).tolist()
    net.apply_to_native()
    try:
        job = Job(input_ids=torch.tensor([ctx], dtype=torch.long),
                  max_new_tokens=1, return_logits=True,
                  sampler=ComboSampler(temperature=args.temperature))
        gen.enqueue(job)
        nat_logits = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                for blob in (r, r.get("held") or {}):
                    if blob.get("logits") is not None:
                        nat_logits = blob["logits"]
    finally:
        net.remove_from_native()
    assert nat_logits is not None, "native generator returned no logits"
    with torch.no_grad():
        tr_logits = net.logits(torch.tensor([ctx], dtype=torch.long),
                               attention_mask=torch.ones(1, len(ctx),
                                                         dtype=torch.long))
    nat = torch.log_softmax(
        nat_logits.reshape(-1, nat_logits.shape[-1])[-1].float(), dim=-1).cpu()
    tr = torch.log_softmax(tr_logits[0, -1].float(), dim=-1).cpu()
    top = tr.topk(20).indices
    dmax = (nat[top] - tr[top]).abs().max().item()
    agree = int(nat.argmax()) == int(tr.argmax())
    assert agree or dmax < 0.25, \
        (f"native/training forward disagree: argmax {int(nat.argmax())} vs "
         f"{int(tr.argmax())}, max |dlogp| over training top-20 = {dmax:.4f}")
    # (e) full EBFT step through the native sampler: finite loss, same metrics
    # surface as the exact path.
    loss2, m2 = ebft_batch(net, batch, pad_id, args, random.Random(0), blocks,
                           gen=gen)
    assert loss2 is not None and torch.isfinite(loss2), \
        "EBFT loss (native sampler) not finite"
    net.zero_grad(set_to_none=True)
    print(f" -- native-sampler self-test: OK (argmax "
          f"{'match' if agree else 'MISMATCH (within tol)'}, max |dlogp| "
          f"top-20 = {dmax:.4f}; native-path step: reward {m2['reward']:+.4f}, "
          f"cfm {m2['cfm']:.4f}, dup {m2['dup']:.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        _run_main()
    except KeyboardInterrupt as e:
        _log_failure("interrupted", e)
        raise SystemExit(130)
    except SystemExit as e:
        if e.code not in (0, None):
            _log_failure("failed", e)
        raise
    except BaseException as e:
        _log_failure("failed", e)
        raise


def _run_main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        raise SystemExit("qlora_train_ebft.py is single-process (--parallel single|split).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_ebft_adapter")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parallel", choices=["single", "split"], default="single")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB")
    ap.add_argument("--split-even", nargs="?", type=float, const=2.0, default=None,
                    metavar="MARGIN_GB",
                    help="(split) Balance the layer split across GPUs instead of the "
                         "default greedy fill-device-0-first: each device is capped to "
                         "its even share of the model's weight bytes (measured from the "
                         "safetensors headers) plus any pre-attached KV-cache pages "
                         "(the rollout sampler's cache allocates during load), "
                         "plus MARGIN_GB of load-time slack "
                         "(default 2.0), leaving every card similar training headroom. "
                         "Raise the margin if loading fails with 'Insufficient VRAM in "
                         "split'; shrink it if the split comes out lopsided. Mutually "
                         "exclusive with --reserve-per-device/--use-per-device.")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB")
    # LoRA
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--expert-r", type=int, default=None)
    ap.add_argument("--init-lora", choices=["default", "pissa", "qerr", "eva"],
                    default="default",
                    help="default/pissa/eva keep phi == step-0 policy (the "
                         "paper's setup); qerr does not (phi is the raw base).")
    ap.add_argument("--init-svd-niter", type=int, default=16)
    ap.add_argument("--init-ref-model", default=None)
    ap.add_argument("--init-eva-tokens", type=int, default=65536)
    # EBFT objective
    ap.add_argument("--gen-len", type=int, default=8,
                    help="Rollout window G (paper: 8 code, 4 translation).")
    ap.add_argument("--n-samples", type=int, default=4,
                    help="Rollouts per anchor (paper: 4; RLOO div term needs >2).")
    ap.add_argument("--anchors", type=int, default=4,
                    help="Anchor positions sampled per sequence per visit "
                         "(the reference strides ALL positions via a custom "
                         "attention mask; this is the anchors-as-rows analog).")
    ap.add_argument("--min-context", type=int, default=8,
                    help="Minimum context tokens before an anchor.")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Rollout sampling temperature (paper Table 2: 0.6).")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--rollout-sampler", choices=["exact", "native"],
                    default="exact",
                    help="exact: gen_len full differentiable re-forwards per "
                         "row, zero sampling/scoring mismatch (v1 default). "
                         "native: KV-cached generator through the runtime-LoRA "
                         "slots (apply_to_native) with paged prefix dedup "
                         "across sibling rollouts -- much faster, on-policy "
                         "modulo inference-kernel numerics; refused for "
                         "routed-expert LoRA (fused MoE kernels bypass the "
                         "runtime slots). Scoring stays differentiable either "
                         "way.")
    ap.add_argument("--sampler-cache-tokens", type=int, default=32768,
                    help="Paged KV cache size (tokens) for --rollout-sampler "
                         "native. Rows that don't fit at once run in waves "
                         "(correct, just less parallel).")
    ap.add_argument("--align-coef", type=float, default=1.0,
                    help="Alignment reward coefficient (reference: 1.0).")
    ap.add_argument("--div-coef", type=float, default=0.5,
                    help="Diversity reward coefficient (reference example: "
                         "0.5, the paper's alpha=0.5 alignment bias; 1.0 is "
                         "the pure proper-scoring-rule objective).")
    ap.add_argument("--rl-coef", type=float, default=1.0)
    ap.add_argument("--ce-coef", type=float, default=0.03,
                    help="CE (gamma) weight on ground-truth sequences "
                         "(reference example: 0.03; paper sweeps 0/0.03/0.1 "
                         "and recommends >0 for stability).")
    ap.add_argument("--no-whiten", action="store_true",
                    help="Disable feature whitening (ablation only -- the "
                         "paper reports clear degradation without it).")
    ap.add_argument("--whiten-tol", type=float, default=1e-5)
    ap.add_argument("--feature-fracs", default="0.25,0.5,0.75",
                    help="Fractional depths of the phi taps.")
    # Optimization
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Paper full-FT actor LR is 1e-6; LoRA wants higher "
                         "(1e-5..1e-4, unvalidated territory).")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                    default="adamw")
    ap.add_argument("--adam-betas", default="0.9,0.95",
                    help="Paper Table 2 uses (0.9, 0.95).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2,
                    help="SEQUENCES per micro-batch. Rollout rows per micro-"
                         "batch = batch * anchors * n_samples (the real "
                         "memory knob) -- default 2*4*4 = 32 rows.")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # Data
    ap.add_argument("--mode", choices=["qa", "pretrain"], default="qa")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="context")
    ap.add_argument("--response-key", default="response")
    ap.add_argument("--messages-key", default=None)
    ap.add_argument("--text-key", default="text", help="(pretrain mode)")
    ap.add_argument("--prompt-format",
                    choices=["auto", "mistral", "metharme", "gemma4-nothink",
                             "llama3", "qwen3.5", "qwen3.5-nothink", "chatml",
                             "hy3", "hy3-nothink", "jinja"],
                    default="auto",
                    help="Chat format (see the SFT trainer's help; jinja = the "
                         "model directory's own Jinja chat template).")
    ap.add_argument("--chat-template-file", default=None,
                    help="(jinja) Jinja template file overriding the model "
                         "directory's template.")
    ap.add_argument("--template-vars", default=None,
                    help="(jinja) JSON object of extra template variables, "
                         "e.g. '{\"enable_thinking\": false}'.")
    ap.add_argument("--strip-sys-prompt-extras", action="store_true",
                    help="(jinja) Delete template-injected 'Reasoning "
                         "strength:', 'Knowledge cutoff:' and 'Current "
                         "date:' lines from every rendered prompt (OFF by "
                         "default). Harmony-style templates synthesize "
                         "these into the system block for rows with no "
                         "system message; the date comes from "
                         "strftime_now(), so without this the rendered "
                         "corpus changes from day to day.")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--clean-text", action="store_true")
    ap.add_argument("--min-response-words", type=int, default=3)
    # Eval / saving
    ap.add_argument("--eval-split", default=None)
    ap.add_argument("--eval-dataset", default=None)
    ap.add_argument("--eval-max-samples", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-cfm", action="store_true", default=True,
                    help="Eval also runs rollouts to report held-out CFM / "
                         "reward calibration (the paper's headline metric). "
                         "On by default; --no-eval-cfm for CE-only eval.")
    ap.add_argument("--no-eval-cfm", dest="eval_cfm", action="store_false")
    ap.add_argument("--eval-cfm-samples", type=int, default=0,
                    help="Cap the CFM eval pass (the expensive rollout part) "
                         "at a fixed-seed subset of N val examples; 0 = full "
                         "val set. CE eval always uses the full val set. The "
                         "full-set CFM pass measured ~half the wall clock of "
                         "the Session-40 quarter-epoch A/B (116 examples, "
                         "~3 min/eval).")
    ap.add_argument("--save-best", action="store_true",
                    help="Keep the checkpoint with the best held-out CE.")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--keep-checkpoints", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-optimizer", action="store_true")
    ap.add_argument("--run-log", default="qlora_runs.csv")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--self-test", action="store_true",
                    help="Run reward-math + model correctness checks on one "
                         "micro-batch, then exit without training.")
    # Runtime knobs shared with the SFT trainer
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--head-vocab-chunk", type=int, default=0)
    ap.add_argument("--offload-activations", action="store_true")
    ap.add_argument("--offload-mode", choices=["async", "sync"], default="async")
    ap.add_argument("--use-liger", action="store_true")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast")
    # Live sample generations (same knobs / native path as the SFT trainer).
    ap.add_argument("--sample-every", type=int, default=0,
                    help="Generate a sample completion every N steps (0 to "
                         "disable). Uses the native inference path with the "
                         "current adapter installed (apply_to_native).")
    ap.add_argument("--sample-prompt", default="Tell me about your day.",
                    help="Prompt for the live sample generations.")
    ap.add_argument("--no-report", action="store_true",
                    help="Disable the local run report (default: on when --out is "
                         "set, written to <out>/run_report/report.html).")
    ap.add_argument("--run-name", default="",
                    help="Name for the local run report (report title / compare "
                         "legend). Default: basename of --out.")
    ap.add_argument("--live-report", action="store_true",
                    help="Serve a LIVE run monitor (localhost http thread, opens "
                         "the browser at run start): charts redraw as metrics "
                         "log, plus a step viewer decoding the exact examples "
                         "any step trains on, past or future (deterministic "
                         "order; computed from memory, never written to disk -- "
                         "report.html stays the dataset-free shareable file). "
                         "Anchor positions/rollouts are sampled at train time "
                         "and are NOT shown; the viewer shows the example text.")
    ap.add_argument("--live-report-port", type=int, default=0,
                    help="Port for --live-report (default 0 = pick a free one).")
    args = ap.parse_args()

    # Process-wide, before any renderer is built (see chat_jinja).
    from chat_jinja import set_strip_sys_prompt_extras
    set_strip_sys_prompt_extras(args.strip_sys_prompt_extras)

    from exllamav3.training import backbone as _backbone
    _backbone.set_dequant_mode(args.dequant_mode)

    if args.n_samples < 2:
        raise SystemExit("--n-samples must be >= 2 (RLOO baseline); paper uses 4.")
    if args.n_samples <= 2 and args.div_coef:
        print(" -- note: n_samples <= 2 zeroes the diversity RLOO correction "
              "(reference behavior); rewards still include the diversity term.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    reward_self_test()

    # Dataset-mix continuity (see the SFT trainer): fresh runs snapshot their
    # data files into <out>/dataset/; resumes verify against that record and
    # may substitute the snapshot. Before the failure-log record below so the
    # RESOLVED paths get logged.
    if args.out:
        data_roles = {"dataset": args.dataset,
                      "eval_dataset": args.eval_dataset}
        if args.resume:
            resolved = resolve_resumed_datasets(args.out, args.resume, data_roles)
            args.dataset = resolved["dataset"]
            args.eval_dataset = resolved["eval_dataset"]
        else:
            snapshot_datasets(args.out, data_roles)

    _FAIL_CTX["run_log"] = args.run_log
    _FAIL_CTX["phase"] = "startup"
    _FAIL_CTX["record"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "arm": "exl3-ebft", "model": args.model, "out": args.out,
        "dataset": args.dataset, "eval_split": args.eval_split or "",
        "eval_dataset": args.eval_dataset or "",
        "r": args.r, "alpha": args.alpha,
        "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
        "lr": args.lr, "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "batch": args.batch, "grad_accum": args.grad_accum, "world_size": 1,
        "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
        "steps_planned": args.steps, "seq_len": args.seq_len,
        "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
        "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
        "max_samples": args.max_samples, "prompt_format": args.prompt_format,
        "notes": _hparam_note(args),
    }

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Load the native model + tokenizer.
    _FAIL_CTX["phase"] = "load_model"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    # The KV cache must be created BEFORE model.load() so each attention layer
    # allocates its paged cache during loading (same constraint as the SFT
    # trainer's live-sample generator); otherwise generation asserts on a
    # missing k_cache. Only needed for --rollout-sampler native.
    # A KV cache is needed for the native rollout sampler AND/OR live samples;
    # either way it must exist before model.load(). One cache serves both (the
    # rollout sampler sizes it; a live-sample-only run needs only a small one).
    sampler_cache = None
    if args.rollout_sampler == "native":
        sampler_cache = Cache(model, max_num_tokens=args.sampler_cache_tokens)
    elif args.sample_every:
        sampler_cache = Cache(model, max_num_tokens=4096)
    if args.split_even is not None and args.parallel != "split":
        raise SystemExit("--split-even only applies to --parallel split.")
    if args.parallel == "split":
        load_kwargs = {}
        if args.split_even is not None:
            if args.reserve_per_device is not None or args.use_per_device is not None:
                raise SystemExit("--split-even computes its own per-device budgets; "
                                 "drop --reserve-per-device/--use-per-device.")
            budgets = compute_even_split_budgets(model, config, args.split_even)
            print(f" -- split-even: use_per_device = "
                  f"[{', '.join(f'{b:.1f}' for b in budgets)}] GB "
                  f"(even weight shares + {args.split_even:g} GB load margin)")
            load_kwargs["use_per_device"] = budgets
        if args.reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = args.reserve_per_device
        if args.use_per_device is not None:
            load_kwargs["use_per_device"] = args.use_per_device
        model.load(progressbar=True, **load_kwargs)
        active_devices = list(model.active_devices)
        print(f" -- layer-autosplit: active devices {active_devices}, "
              f"output device {model.output_device}")
    else:
        model.load(device=args.device, progressbar=True)
        active_devices = [torch.device(args.device).index]
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 2. Differentiable QLoRA net (frozen head only: the fused logps path).
    _FAIL_CTX["phase"] = "build_net"
    _FAIL_CTX["record"]["arch"] = getattr(config, "architecture", "")
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        use_rslora=args.use_rslora, compute_dtype=cdt,
        gradient_checkpointing=not args.no_grad_ckpt,
        attn_impl=args.attn_impl, head_vocab_chunk=args.head_vocab_chunk,
        offload_activations=args.offload_activations,
        offload_mode=args.offload_mode, use_liger=args.use_liger,
        expert_r=args.expert_r,
    )
    net.train()
    if args.init_lora in ("pissa", "qerr"):
        _FAIL_CTX["phase"] = "init_lora"
        net.apply_init_lora(args.init_lora, ref_model_dir=args.init_ref_model,
                            svd_niter=args.init_svd_niter)
        if args.init_lora == "qerr":
            print(" -- note: with qerr the step-0 policy != phi (phi is the raw "
                  "quantized base); the paper's phi == init-policy identity "
                  "does not hold.")
    if args.resume:
        net.load_adapter(args.resume)

    fracs = tuple(float(x) for x in args.feature_fracs.split(","))
    blocks = net.feature_block_indices(fracs)
    print(f" -- trainable params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules})")
    print(f" -- {net.describe_attn()}")
    print(f" -- EBFT: G={args.gen_len} n={args.n_samples} anchors={args.anchors} "
          f"temp={args.temperature} | align {args.align_coef} / div {args.div_coef} "
          f"| rl {args.rl_coef} / ce {args.ce_coef} | "
          f"whiten={'off' if args.no_whiten else 'on'} | "
          f"phi taps at blocks {blocks} (of {len(net.blocks)})")

    # Optional KV-cached rollout sampler on the native inference path.
    is_moe_lora = any(w.r > 0 and ".experts." in w.key for w in net._wrappers)
    gen = None
    if args.rollout_sampler == "native":
        if is_moe_lora:
            raise SystemExit(
                "--rollout-sampler native is off-policy for routed-expert "
                "LoRA: the fused MoE inference kernels bypass the runtime "
                "LoRA slots, so rollouts would come from the BASE experts. "
                "Use --rollout-sampler exact (or non-expert --targets).")
        gen = Generator(model=model, cache=sampler_cache, tokenizer=tokenizer)
        print(f" -- rollout sampler: native generator (KV-cached, paged prefix "
              f"dedup across sibling rollouts, {args.sampler_cache_tokens} "
              f"cache tokens); scoring stays on the differentiable path")

    # Optional live sample generations (native inference path with the current
    # adapter installed, exactly like the SFT trainer -- so the two A/B arms
    # print comparable previews). Reuses the rollout generator/cache when
    # present, else builds one over the small cache allocated before load().
    sample_gen = None
    build_prompt = None
    from chat_jinja import parse_template_vars
    template_vars = parse_template_vars(args.template_vars)
    if args.sample_every:
        build_prompt, _ = format_prompt_and_eot(
            model, tokenizer, args.prompt_format,
            chat_template_file=args.chat_template_file,
            template_vars=template_vars)
        sample_gen = gen or Generator(model=model, cache=sampler_cache,
                                      tokenizer=tokenizer)
        if is_moe_lora:
            print(" -- note: live samples on routed-expert LoRA reflect the "
                  "BASE experts (fused MoE kernels bypass the runtime LoRA "
                  "slots); the preview won't show the adapter's expert deltas.")
        print(f" -- live samples every {args.sample_every} steps: "
              f"{args.sample_prompt!r}")

    def live_sample(tag):
        """Generate one completion from the CURRENT policy (adapter installed
        in the runtime slots), print it, and restore train state."""
        net.eval()
        net.apply_to_native()
        try:
            with torch.inference_mode():
                txt = sample(model, sampler_cache, tokenizer, sample_gen,
                             build_prompt, args.sample_prompt)
        finally:
            net.remove_from_native()
            net.train()
        print(f"\n  \U0001f3ad  {tag} {args.sample_prompt}\n     -> {txt}\n")

    # 3. Data.
    _FAIL_CTX["phase"] = "build_dataset"
    if args.mode == "qa":
        examples = build_sft_examples(
            model, tokenizer, args.dataset, args.max_samples, args.seq_len,
            instruction_key=args.instruction_key, context_key=args.context_key,
            response_key=args.response_key, split=args.dataset_split,
            clean_text=args.clean_text,
            min_response_words=args.min_response_words,
            messages_key=args.messages_key, prompt_format=args.prompt_format,
            shuffle=args.shuffle, shuffle_seed=args.shuffle_seed,
            config_name=args.dataset_config,
            chat_template_file=args.chat_template_file,
            template_vars=template_vars)
    else:
        examples = build_lm_examples(
            tokenizer, args.dataset, args.dataset_split, args.seq_len,
            text_key=args.text_key, max_samples=args.max_samples,
            config_name=args.dataset_config)
    # Drop rows that can't host a single anchor.
    usable = [ex for ex in examples
              if pick_anchors(ex, args.gen_len, 1, args.min_context, random.Random(0))]
    if len(usable) < len(examples):
        print(f" -- dropped {len(examples) - len(usable)} rows with no anchor room "
              f"(supervised span < {args.gen_len} tokens past min-context)")
    examples = usable
    print(f" -- {len(examples)} {args.mode} examples"
          f"{' (shuffled)' if args.shuffle else ''}")
    assert examples, "no usable training examples"

    val_examples = []
    if args.eval_split:
        if args.mode == "qa":
            val_examples = build_sft_examples(
                model, tokenizer, args.eval_dataset or args.dataset,
                args.eval_max_samples, args.seq_len,
                instruction_key=args.instruction_key,
                context_key=args.context_key, response_key=args.response_key,
                split=args.eval_split, clean_text=args.clean_text,
                min_response_words=args.min_response_words,
                messages_key=args.messages_key,
                prompt_format=args.prompt_format,
                config_name=args.dataset_config,
                chat_template_file=args.chat_template_file,
                template_vars=template_vars)
        else:
            val_examples = build_lm_examples(
                tokenizer, args.eval_dataset or args.dataset, args.eval_split,
                args.seq_len, text_key=args.text_key,
                max_samples=args.eval_max_samples,
                config_name=args.dataset_config)
        print(f" -- held-out eval: {len(val_examples)} examples from "
              f"'{args.eval_split}'")
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val examples; "
              f"{len(examples)} for training")
        assert examples, "val_frac too large; no training examples left"

    if args.init_lora == "eva" and not args.resume:
        _FAIL_CTX["phase"] = "init_lora"

        def eva_prepass():
            used, i = 0, 0
            while used < args.init_eva_tokens and i < len(examples):
                b = examples[i:i + args.batch]
                i += len(b)
                ids, lens = pad_rows([ex["input_ids"] for ex in b], pad_id)
                attn = (torch.arange(ids.shape[1]).unsqueeze(0)
                        < lens.unsqueeze(1)).long()
                used += int(attn.sum())
                yield dict(input_ids=ids, attention_mask=attn)

        net.apply_init_lora("eva", svd_niter=args.init_svd_niter,
                            eva_batches=eva_prepass())
    elif args.init_lora == "eva":
        print(" -- eva init skipped on --resume")

    if args.self_test:
        _FAIL_CTX["phase"] = "self_test"
        model_self_test(net, examples[:args.batch], pad_id, args, blocks,
                        gen=gen)
        print(" -- self-test complete; exiting (drop --self-test to train).")
        return

    args.steps, warmup_steps = resolve_steps_and_warmup(
        args, len(examples), args.batch * args.grad_accum)
    print(f" -- {args.steps} steps, scheduler={args.scheduler}, "
          f"warmup={warmup_steps}, weight_decay={args.weight_decay}")

    # 4. Optimizer + schedule.
    betas = tuple(float(b) for b in args.adam_betas.split(","))
    opt = build_optimizer(net.param_groups(args.weight_decay), args.lr, args.optim)
    for g in opt.param_groups:
        g["betas"] = betas
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)
    resume_step, resume_state = 0, None
    yaml_base_lrs = list(sched.base_lrs)  # per-group base LRs from the CONFIG
    if args.resume and not args.reset_optimizer:
        resume_state = load_trainer_state(args.resume)
        if resume_state is not None:
            restore_optimizer_state(opt, resume_state["optimizer"])
            if resume_state.get("scheduler") is not None:
                sched.load_state_dict(resume_state["scheduler"])
            # Config lr wins on resume (see the SFT trainer): re-base the
            # restored schedule at the config value; moments/position kept.
            if list(sched.base_lrs) != yaml_base_lrs:
                old_lr = sched.base_lrs[0]
                sched.base_lrs = list(yaml_base_lrs)
                for g, base, fn in zip(opt.param_groups, sched.base_lrs,
                                       sched.lr_lambdas):
                    g["initial_lr"] = base
                    g["lr"] = base * fn(sched.last_epoch)
                sched._last_lr = [g["lr"] for g in opt.param_groups]
                print(f" -- resume LR override: base lr {old_lr:.2e} "
                      f"(checkpoint) -> {yaml_base_lrs[0]:.2e} (config); "
                      f"optimizer moments and schedule position kept")
            resume_step = int(resume_state["step"])
            print(f" -- resumed trainer state: continuing at step "
                  f"{resume_step + 1}/{args.steps}")

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(args.shuffle_seed).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    b0_refs = []
    with torch.no_grad():
        for w in net._wrappers:
            if w.r <= 0:
                continue
            if w.init_b0_master is not None:
                b0_refs.append((w, w.init_b0_master))
            elif args.init_lora != "default" and w.lora_b.abs().max().item() > 0:
                b0_refs.append((w, w.lora_b.detach().float().cpu().clone()))
            else:
                b0_refs.append((w, None))

    def adapter_b_norm():
        with torch.no_grad():
            tot = 0.0
            for w, b0 in b0_refs:
                if b0 is None:
                    tot += w.lora_b.float().pow(2).sum().item()
                else:
                    tot += (w.lora_b.detach().float().cpu() - b0).pow(2).sum().item()
            return tot ** 0.5

    def save(tag):
        net.save_adapter(args.out, base_model_name_or_path=args.model)
        save_trainer_state(args.out, step=step, opt=opt, sched=sched,
                           best_val=best_val, best_val_step=best_val_step, ema=ema)
        print(f"{tag} Adapter written to {args.out}")

    # CFM eval subset (--eval-cfm-samples): the rollout pass is the expensive
    # part of eval; a fixed-seed subset keeps the CFM trend comparable across
    # evals AND across runs while cutting its cost proportionally. CE eval
    # always covers the full val set (cheap).
    cfm_examples = val_examples
    if (args.eval_cfm and args.eval_cfm_samples
            and len(val_examples) > args.eval_cfm_samples):
        keep = sorted(random.Random(12345).sample(
            range(len(val_examples)), args.eval_cfm_samples))
        cfm_examples = [val_examples[i] for i in keep]
        print(f" -- eval CFM capped at {len(cfm_examples)}/{len(val_examples)} "
              f"val examples (--eval-cfm-samples, fixed-seed subset)")

    def evaluate():
        """Held-out CE (nats/token over supervised positions) and, with
        --eval-cfm, rollout-based CFM / reward calibration on the val set."""
        if not val_examples:
            return None, {}
        ev_t0 = time.time()
        net.eval()
        vrng = random.Random(12345)   # fixed anchors/rollouts across evals
        # Fix the rollout RNG for comparable evals, WITHOUT perturbing the
        # training stream: save/restore global torch RNG state around eval.
        cpu_state = torch.random.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        torch.manual_seed(12345)
        ce_num, ce_den = 0.0, 0
        agg = {}
        with torch.no_grad():
            for i in range(0, len(val_examples), args.batch):
                vb = val_examples[i:i + args.batch]
                ids, lens = pad_rows([ex["input_ids"] for ex in vb], pad_id)
                attn = (torch.arange(ids.shape[1]).unsqueeze(0)
                        < lens.unsqueeze(1)).long()
                labels = torch.full_like(ids, -100)
                for bi, ex in enumerate(vb):
                    lab = torch.as_tensor(ex["labels"], dtype=torch.long)
                    labels[bi, :len(lab)] = lab
                l = net.compute_loss(ids, labels, attention_mask=attn,
                                     chunk=args.ce_chunk)
                ntok = int((labels[:, 1:] != -100).sum())
                ce_num += l.item() * ntok
                ce_den += ntok
            if args.eval_cfm:
                for i in range(0, len(cfm_examples), args.batch):
                    vb = cfm_examples[i:i + args.batch]
                    _, m = ebft_batch(net, vb, pad_id, args, vrng, blocks,
                                      gen=gen)
                    if not m:
                        continue
                    for k in ("cfm", "fmw", "reward", "dup"):
                        s, c = agg.get(k, (0.0, 0))
                        agg[k] = (s + m[k] * m["n"], c + m["n"])
        net.train()
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        ce = ce_num / max(ce_den, 1)
        em = {k: s / c for k, (s, c) in agg.items()}
        em["eval_s"] = time.time() - ev_t0
        return ce, em

    def fmt_eval(vl, em):
        parts = [f"ce {vl:.4f}"]
        if "cfm" in em:
            parts.append(f"cfm {em['cfm']:.4f}")
        if "fmw" in em:
            parts.append(f"fm-w {em['fmw']:+.4f}")
        if "reward" in em:
            parts.append(f"reward {em['reward']:+.4f}")
        if "dup" in em:
            parts.append(f"dup {em['dup']:.2f}")
        if "eval_s" in em:
            parts.append(f"{em['eval_s']:.0f}s")
        return " | ".join(parts)

    bgen = batches()
    if resume_step:
        # Fast-forward the data stream to where the original process stopped
        # (see the SFT trainer; doc/bug_resume_data_order.md). Deterministic
        # from Random(shuffle_seed), so this reproduces the exact order an
        # uninterrupted run would have used.
        for _ in range(resume_step * args.grad_accum):
            next(bgen)
        print(f" -- data stream fast-forwarded {resume_step * args.grad_accum} "
              f"micro-batches to continue the run's batch order at step "
              f"{resume_step + 1}")
    opt.zero_grad(set_to_none=True)
    ema = resume_state["ema"] if resume_state else None
    step = resume_step
    best_val = resume_state["best_val"] if resume_state else float("inf")
    best_val_step = resume_state["best_val_step"] if resume_state else 0
    start_loss = end_loss = None
    start_val = None
    last_eval_step, last_val = -1, None
    tok_seen, tot_seen = 0, 0
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    meter = ThroughputMeter()

    # Local run report -- the default logging path (same as the SFT trainer, so
    # EBFT and SFT runs render comparable dashboards). On by default when there's
    # an --out; --no-report opts out. Reuses native's module-level _REPORT state
    # so native's failure logger renders it on a crash. Config mirrors the CSV row.
    report = None
    if args.out and not args.no_report:
        run_config = dict(_FAIL_CTX["record"])
        run_config.update(
            warmup_steps=warmup_steps, targets=" ".join(net.target_modules),
            trainable_params=net.num_trainable(), n_train=len(examples),
            n_val=len(val_examples))
        report = RunLogger(
            args.out,
            args.run_name or os.path.basename(os.path.normpath(args.out)),
            config=run_config)
        _REPORT["rep"] = report

    # Live monitor (--live-report). Mirrors batches()/the loop: optimizer step
    # s consumes micro-batches (s-1)*ga .. (s-1)*ga+ga-1 -- an ABSOLUTE
    # mapping, valid on resume too now that bgen is fast-forwarded to the
    # resume step. batches() reseeds
    # Random(shuffle_seed) every pass but shuffles the SAME list in place, so
    # epoch e's permutation is that seeded shuffle applied e+1 times
    # cumulatively -- replayed lazily as steps are browsed. Rollout anchors are
    # sampled at train time (stateful rng) and not shown.
    if args.live_report and report is None:
        print(" -- --live-report needs the local report; ignoring "
              "(--no-report set or no --out)")
    elif args.live_report:
        live_orders = {"order": list(range(len(examples))), "perms": []}
        live_mpe = max(1, len(examples) // args.batch)  # micro-batches per pass

        def live_perm(e):
            while len(live_orders["perms"]) <= e:
                random.Random(args.shuffle_seed).shuffle(live_orders["order"])
                live_orders["perms"].append(live_orders["order"][:])
            return live_orders["perms"][e]

        def live_batch_fn(s):
            if not (0 < s <= args.steps):
                raise ValueError(f"step must be in [1, {args.steps}]")
            mbs = []
            for g in range(args.grad_accum):
                m = (s - 1) * args.grad_accum + g
                e, w = divmod(m, live_mpe)
                seqs = [{"index": j,
                         "docs": decode_example_docs(tokenizer, examples[j])}
                        for j in live_perm(e)[w * args.batch:(w + 1) * args.batch]]
                mbs.append({"micro": g, "epoch": e, "sequences": seqs})
            return {"step": s, "micro_batches": mbs}

        start_live_monitor(
            args.out, batch_fn=live_batch_fn, port=args.live_report_port,
            live_info={"total_steps": args.steps, "first_step": resume_step + 1,
                       "run_name": args.run_name
                       or os.path.basename(os.path.normpath(args.out))})

    _FAIL_CTX["phase"] = "baseline_eval"
    if val_examples:
        start_val, em = evaluate()
        if start_val is not None:
            print(f"    [eval] step 0 (baseline): {fmt_eval(start_val, em)}")
            if report is not None:
                report.log({k: v for k, v in (("eval/held_out", start_val),
                                              ("eval/cfm", em.get("cfm")))
                            if v is not None}, step=0)
    if args.sample_every:
        _FAIL_CTX["phase"] = "baseline_sample"
        live_sample("baseline (step 0):")

    t0 = time.time()
    if torch.cuda.is_available():
        for d in active_devices:
            torch.cuda.reset_peak_memory_stats(d)

    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return max((torch.cuda.max_memory_allocated(d) / 1e9 for d in active_devices),
                   default=0.0)

    def log_run(status, dt, final_val):
        _FAIL_CTX["logged"] = True
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "exl3-ebft", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "",
            "eval_dataset": args.eval_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "lr": args.lr, "scheduler": args.scheduler,
            "warmup_steps": warmup_steps, "weight_decay": args.weight_decay,
            "batch": args.batch, "grad_accum": args.grad_accum, "world_size": 1,
            "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(net.target_modules),
            "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
            "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
            "max_samples": args.max_samples, "prompt_format": args.prompt_format,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "start_val": rnd(start_val), "final_val": rnd(final_val),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(tok_seen / dt) if dt else "",
            "tot_tok_s": round(tot_seen / dt) if dt else "",
            "peak_vram_gb": rnd(peak_vram_gb(), 3),
            "t_data_s": rnd(timer.total["data"], 1),
            "t_fwd_s": rnd(timer.total["fwd"], 1),
            "t_bwd_s": rnd(timer.total["bwd"], 1),
            "t_opt_s": rnd(timer.total["opt"], 1),
            "phase": "", "error": "", "notes": _hparam_note(args),
        })

    timer = StepTimer(devices=active_devices if torch.cuda.is_available() else None)

    # Last COMPLETED optimizer step (see qlora_train_native.py): `step` holds the
    # in-flight iteration mid-loop, and saving that on Ctrl-C desyncs the step
    # counter from the scheduler, skipping an iteration's batches on resume.
    done_step = resume_step
    try:
        for step in range(resume_step + 1, args.steps + 1):
            _FAIL_CTX["phase"] = f"train step {step}"
            step_t0 = time.time()
            accum_loss = 0.0
            step_sup = step_tot = 0
            step_m = {}
            timer.begin_step()
            window = [next(bgen) for _ in range(args.grad_accum)]
            timer.mark("data")
            n_micro = 0
            for batch in window:
                loss, m = ebft_batch(net, batch, pad_id, args, rng, blocks,
                                     gen=gen)
                if loss is None:
                    continue
                n_micro += 1
                loss_val = loss.item()
                timer.mark("fwd")
                (loss / len(window)).backward()
                timer.mark("bwd")
                accum_loss += loss_val / len(window)
                step_sup += m["sup"]
                step_tot += m["tot"]
                for k in ("rl", "ce", "reward", "align", "div", "cfm",
                          "fmw", "dup", "adv_std"):
                    s, c = step_m.get(k, (0.0, 0))
                    step_m[k] = (s + m[k] * m["n"], c + m["n"])
            if not n_micro:
                print(f"  step {step}: no usable anchors in window, skipped")
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(
                net.trainable_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            timer.mark("opt")
            timer.end_step()

            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            _, tot_tps = meter.rates()

            if start_loss is None:
                start_loss = accum_loss
            end_loss = accum_loss
            ema = accum_loss if ema is None else 0.9 * ema + 0.1 * accum_loss
            mm = {k: s / c for k, (s, c) in step_m.items()}
            print(f"  step {step:>5}/{args.steps} | loss {accum_loss:7.4f} | "
                  f"rl {mm.get('rl', 0):+7.4f} ce {mm.get('ce', 0):6.4f} | "
                  f"reward {mm.get('reward', 0):+6.3f} "
                  f"(al {mm.get('align', 0):+5.3f} dv {mm.get('div', 0):5.3f} "
                  f"dup {mm.get('dup', 0):.2f}) | "
                  f"cfm {mm.get('cfm', 0):6.4f} | grad {gnorm:7.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"|dB| {adapter_b_norm():7.3f} | {tot_tps:,.0f} tok/s | "
                  f"{timer.step_line()}")

            if report is not None:
                report.log({
                    "train/loss": accum_loss, "train/ema": ema,
                    "train/rl": mm.get("rl", 0.0), "train/ce": mm.get("ce", 0.0),
                    "train/reward": mm.get("reward", 0.0),
                    "train/align": mm.get("align", 0.0),
                    "train/div": mm.get("div", 0.0), "train/dup": mm.get("dup", 0.0),
                    "train/cfm": mm.get("cfm", 0.0), "train/grad_norm": gnorm,
                    "train/lr": sched.get_last_lr()[0],
                    "train/adapter_b_dist": adapter_b_norm(),
                    "perf/tot_tok_s": tot_tps,
                }, step=step)

            _FAIL_CTX["record"].update(
                steps_done=step, end_loss=round(accum_loss, 6),
                peak_vram_gb=round(peak_vram_gb(), 3))
            if start_loss is not None and "start_loss" not in _FAIL_CTX["record"]:
                _FAIL_CTX["record"]["start_loss"] = round(start_loss, 6)

            if args.eval_every and step % args.eval_every == 0 and val_examples:
                vl, em = evaluate()
                last_eval_step, last_val = step, vl
                if vl is not None:
                    print(f"    [eval] step {step}: {fmt_eval(vl, em)}")
                    if report is not None:
                        emetrics = {"eval/held_out": vl}
                        if "cfm" in em:
                            emetrics["eval/cfm"] = em["cfm"]
                        if vl < best_val:
                            emetrics["eval/best_val"] = vl
                        else:
                            emetrics["eval/best_val"] = best_val
                        report.log(emetrics, step=step)
                    if vl < best_val:
                        best_val = vl
                        best_val_step = step
                        if args.save_best:
                            save(f"[best step {step}, val {vl:.4f}]")

            if args.sample_every and step % args.sample_every == 0:
                live_sample(f"[step {step}]")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                cdir = checkpoint_dir(args.out, step)
                net.save_adapter(cdir, base_model_name_or_path=args.model)
                save_trainer_state(cdir, step=step, opt=opt, sched=sched,
                                   best_val=best_val, best_val_step=best_val_step,
                                   ema=ema)
                print(f"  [checkpoint] step {step} -> {cdir} (resumable)")
                prune_checkpoints(args.out, args.keep_checkpoints)

            # Reached only if the whole iteration ran: this step is now durable.
            done_step = step
    except KeyboardInterrupt:
        # Roll back to the last completed step so the saved counter, the
        # scheduler and the data stream all agree.
        step = done_step
        if args.save_best and val_examples:
            print(f"\nInterrupted at step {step}; keeping best-val adapter.")
        else:
            print(f"\nInterrupted at step {step}; saving adapter before exit.")
            if step > 0:
                save("[interrupted]")
        log_run("interrupted", time.time() - t0, None)
        if report is not None:
            report.update_summary({
                "end_loss": end_loss, "best_val": best_val if best_val != float("inf") else None,
                "best_val_step": best_val_step or None,
                "peak_vram_gb": peak_vram_gb(), "steps_done": step})
        _finish_report(exit_code=0, status="interrupted")
        raise SystemExit(0)

    dt = time.time() - t0
    _FAIL_CTX["phase"] = "final_eval"
    if last_eval_step == step:
        val_loss = last_val
    elif val_examples:
        print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss, em = evaluate()
        if val_loss is not None:
            print(f"    [eval] final: {fmt_eval(val_loss, em)}")
    else:
        val_loss = None
    if not (args.save_best and val_examples):
        save("Done.")
    if torch.cuda.is_available():
        peak_str = " / ".join(
            f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1e9:.2f}GB"
            for d in active_devices)
    else:
        peak_str = "n/a"
    print(f"[PERF] {tok_seen / dt if dt else 0:,.0f} sup tok/s, "
          f"{tot_seen / dt if dt else 0:,.0f} tot tok/s | peak VRAM {peak_str} | "
          f"{dt:.0f}s for {step} steps | step time: {timer.summary()}")
    log_run("completed", dt, val_loss)
    if report is not None:
        report.update_summary({k: v for k, v in {
            "end_loss": end_loss, "final_val": val_loss,
            "best_val": best_val if best_val != float("inf") else None,
            "best_val_step": best_val_step or None,
            "peak_vram_gb": peak_vram_gb(),
            "sup_tok_s": tok_seen / dt if dt else 0,
            "tot_tok_s": tot_seen / dt if dt else 0,
            "total_s": dt, "steps_done": step,
        }.items() if v is not None})
        _finish_report()
    print("Verify with: python training/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}")


def _hparam_note(args):
    return (f"method=ebft G={args.gen_len} n={args.n_samples} "
            f"anchors={args.anchors} temp={args.temperature} "
            f"align={args.align_coef} div={args.div_coef} "
            f"rl={args.rl_coef} ce={args.ce_coef} "
            f"whiten={0 if args.no_whiten else 1} mode={args.mode} "
            f"sampler={args.rollout_sampler}")


if __name__ == "__main__":
    main()
