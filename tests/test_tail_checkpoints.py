"""
Integration test for completion tail checkpoints on hybrid recurrent models.

Loads a real model with SWA layers in recurrent mode and verifies, with greedy sampling:

1. Continuation matrix: a completed request leaves a page-aligned recurrent tail checkpoint,
   and a follow-up request extending the same conversation resumes within one page of its end
   instead of at the last interval checkpoint. Each case compares the resumed output
   token-for-token against a cold full-prefill reference from a separate page table, which
   catches a state stored under the right page hash at the wrong ring alignment. The sweep
   covers rewind distances 1/40/254 across the cursor alignment band, plus the exact-boundary
   case whose final sampled token is never forwarded (rewind 255, previous-boundary fallback).
2. Enqueue-triggered cleanup: a caller that stops driving iterate() at EOS leaves the
   completed job deferred; enqueue() must run the cleanup (under inference mode) without
   crashing.
3. Requeue coinciding with EOS: a job whose max_rq_tokens round-trip lands exactly on its
   max_new_tokens EOS must complete (requeue suppressed), reach the deferred-cleanup path,
   and leave a usable tail checkpoint.

Requires a GPU and an exl3 model directory of a hybrid-attention model whose recurrent state
supports in-place rollback (e.g. Gemma-4 with SWA layers in recurrent mode):

    python tests/test_tail_checkpoints.py --model <exl3 model dir>

Note for linear-attention models (e.g. GDN): states without guaranteed_rollback keep default
checkpoint behavior by design, and their scan kernels are not run-to-run deterministic, so
this test skips the matrix for them.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job, GreedySampler
from exllamav3.constants import PAGE_SIZE

checks_passed = 0
checks_failed = 0

def check(name, cond, detail = ""):
    global checks_passed, checks_failed
    if cond:
        checks_passed += 1
        print(f"  PASS  {name}")
    else:
        checks_failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def make_long_prompt(tokenizer, seed_text, target_pages):
    # Tokenize while growing: character-count heuristics undershoot on tokenizers that compress
    # repetitive filler well, and the test needs a guaranteed number of full pages
    text = seed_text
    filler = (
        " The archive records further details, cross-referenced against seasonal observations"
        " and the incidental notes of several field assistants over many years of study."
    )
    while tokenizer.encode(text).shape[-1] < target_pages * PAGE_SIZE + 16:
        text += filler
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True,
                    help = "exl3 model directory of a hybrid-attention model with recurrent SWA layers "
                           "(e.g. Gemma-4)")
    ap.add_argument("--cache_size", type = int, default = 8192)
    ap.add_argument("--max_batch_size", type = int, default = 16,
                    help = "Recurrent state slots per cache; the test itself is batch-1, so this can be "
                           "lowered to fit large hybrid models on smaller GPUs")
    args = ap.parse_args()

    print(f" -- Loading model: {args.model}")
    config = Config.from_directory(args.model)
    model = Model.from_config(config, swa_full = False)
    cache_main = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    cache_ctrl = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    sampler = GreedySampler()

    generator = Generator(model = model, cache = cache_main, tokenizer = tokenizer)
    if generator.recurrent_cache is None:
        print(" -- Model has no recurrent states; tail checkpoints do not apply. Skipping.")
        sys.exit(0)

    prompt_a = make_long_prompt(
        tokenizer,
        "The migration of songbirds across continents follows magnetic fields and stellar cues.", 6)
    a_ids = tokenizer.encode(prompt_a)
    print(f" -- prompt A: {a_ids.shape[-1]} tokens ({a_ids.shape[-1] // PAGE_SIZE} full pages)")

    def run_job_ids(gen, ids, max_new, cleanup_tail = True, max_rq_tokens = None):
        # Token-level Job API: continuation prompts are built from exact ids, avoiding tokenization-seam
        # ambiguity at the prompt/response join
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = sampler,
                  max_rq_tokens = max_rq_tokens)
        gen.enqueue(job)
        out, eos_r = [], None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                t = r.get("token_ids")
                if t is not None:
                    out.append(t)
                if r.get("eos"):
                    eos_r = r
            if eos_r is not None and not cleanup_tail:
                # A cold correctness reference does not need to populate the recurrent cache. Release its
                # deferred state without allocating another model-sized pinned tail-staging buffer.
                gen.clear_queue()
                break
        out_ids = torch.cat(out, dim = -1) if out else torch.empty((1, 0), dtype = torch.long)
        return out_ids, eos_r

    # Tail checkpoints engage only for recurrent state types with guaranteed in-place rollback
    rollback_ok = getattr(generator.cache.recurrent_state_cls, "guaranteed_rollback", 0) > 0
    if not rollback_ok:
        print(" -- Recurrent state has no guaranteed rollback (e.g. GDN); tail checkpoints keep default")
        print("    behavior on this architecture by design. Skipping matrix.")
        sys.exit(0)

    mid_case_ids = torch.cat([
        a_ids,
        tokenizer.encode(" A separate continuation-check branch follows.", add_bos = False),
    ], dim = -1)
    extra_ids = tokenizer.encode(" Now summarize the above in one sentence.", add_bos = False)

    def check_continuation_case(name, base_ids, tail_offset, exact_boundary, expected_rewind,
                                extra_pages = 1):
        prompt_boundary = base_ids.shape[-1] // PAGE_SIZE * PAGE_SIZE
        target_boundary = prompt_boundary + extra_pages * PAGE_SIZE
        # Job's token-level limit includes the already-pending first decode position, hence the +1.
        generated_tokens = target_boundary - base_ids.shape[-1] + 1 + tail_offset
        resp_ids, _ = run_job_ids(generator, base_ids, generated_tokens)
        prev_ids = torch.cat([base_ids, resp_ids], dim = -1)
        prev_boundary = prev_ids.shape[-1] // PAGE_SIZE * PAGE_SIZE
        check(f"{name}: setup crosses a new page boundary", prev_boundary > prompt_boundary,
              f"prev len = {prev_ids.shape[-1]}, boundary = {prev_boundary}, "
              f"prompt pages end = {prompt_boundary}")
        if exact_boundary:
            check(f"{name}: completion ends exactly on the boundary",
                  prev_ids.shape[-1] == prev_boundary,
                  f"prev len = {prev_ids.shape[-1]}, boundary = {prev_boundary}")
        # Recurrent position at completion is prev len - 1 (the final sampled token is never forwarded)
        stash_target = prev_boundary - PAGE_SIZE if exact_boundary else prev_boundary
        actual_rewind = prev_ids.shape[-1] - 1 - stash_target
        check(f"{name}: exercises the intended rewind distance", actual_rewind == expected_rewind,
              f"actual rewind = {actual_rewind}, intended = {expected_rewind}")

        cont_ids = torch.cat([prev_ids, extra_ids], dim = -1)
        # A separate page table has no matching KV pages or recurrent checkpoints, providing a full-prefill
        # reference. This catches a state stored under the right page hash at the wrong ring alignment.
        cold_gen = Generator(model = model, cache = cache_ctrl, tokenizer = tokenizer)
        cold_out, _ = run_job_ids(cold_gen, cont_ids, 16, cleanup_tail = False)

        warm_out, eos_r = run_job_ids(generator, cont_ids, 16)
        cached_cont = eos_r["cached_tokens"]
        # The final sampled token's K/V is never forwarded, so a completion landing exactly on a page
        # boundary stashes at the previous boundary instead
        resume_floor = prev_boundary - PAGE_SIZE if exact_boundary else prev_boundary
        check(f"{name}: resumes within a page of the previous request's end",
              cached_cont >= resume_floor,
              f"cached_tokens = {cached_cont}, expected >= {resume_floor} "
              f"(a cap at {prompt_boundary} means no post-generation resume point exists)")
        check(f"{name}: resumed output matches cold full-prefill",
              torch.equal(warm_out, cold_out),
              f"warm ids = {warm_out.tolist()}\ncold ids = {cold_out.tolist()}")

    print(" -- Conversation continuation: hybrid resume granularity after a completed generation")
    # The exact case proves the previous-boundary fallback for a final sampled token that was never
    # forwarded: it generates through TWO new pages so the fallback target is itself a post-generation
    # boundary that only completion cleanup can have stashed (the prompt-tail checkpoint cannot satisfy
    # it). The mid-page cases prove decode-time stashes are ring-cursor-correct: tail_offset = N yields a
    # rewind of N - 1, and the sweep covers both edges of the alignment band. Distinct histories per case
    # so none can silently reuse another case's tail checkpoint.
    sweep_lo_ids = torch.cat([
        a_ids, tokenizer.encode(" Cursor sweep low-edge case follows.", add_bos = False)], dim = -1)
    sweep_hi_ids = torch.cat([
        a_ids, tokenizer.encode(" Cursor sweep high-edge case instead.", add_bos = False)], dim = -1)
    check_continuation_case("exact-boundary continuation", a_ids, 0, True, 255, extra_pages = 2)
    check_continuation_case("mid-page continuation", mid_case_ids, 41, False, 40)
    check_continuation_case("mid-page continuation (rewind 1)", sweep_lo_ids, 2, False, 1)
    check_continuation_case("mid-page continuation (rewind 254)", sweep_hi_ids, 255, False, 254)

    # Deferred cleanup must also work when triggered from enqueue() rather than iterate(): a caller that
    # stops driving iterate() at EOS leaves the completed job deferred, and enqueue() has no inference-mode
    # decorator of its own. Regression for an unstash inference-tensor crash on that path.
    print(" -- Enqueue-triggered deferred cleanup")
    job = Job(input_ids = a_ids, max_new_tokens = 40, sampler = sampler)
    generator.enqueue(job)
    eos_seen = False
    while generator.num_remaining_jobs() and not eos_seen:
        for r in generator.iterate():
            if r.get("eos"):
                eos_seen = True
    try:
        follow_up, _ = run_job_ids(generator, a_ids, 8)   # enqueue() triggers the deferred cleanup
        check("enqueue-triggered deferred cleanup does not crash", follow_up.shape[-1] > 0)
    except RuntimeError as e:
        check("enqueue-triggered deferred cleanup does not crash", False, str(e))

    # A requeue that coincides with EOS is suppressed and the job completes; it must then flow through the
    # deferred-cleanup path like any completion and leave a usable tail checkpoint. On recurrent models
    # max_rq_tokens snaps up to the next recurrent_checkpoint_interval boundary, so a dedicated generator
    # pins the interval to PAGE_SIZE, the snapped threshold is recomputed with the Job's own formula, and
    # max_new_tokens = snapped + 1 makes the EOS step provably the first requeue-trigger step.
    print(" -- Requeue coinciding with EOS")
    rq_case_ids = torch.cat([
        a_ids, tokenizer.encode(" Requeue coincidence case follows here.", add_bos = False)], dim = -1)
    x = rq_case_ids.shape[-1]
    # Snap target two boundaries out, so the exact-boundary fallback target (y - PAGE_SIZE) is itself a
    # post-generation boundary that only completion cleanup can have stashed
    y = (x - 1 + 8 + PAGE_SIZE + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    snapped = y - x
    rq_gen = Generator(model = model, cache = cache_main, tokenizer = tokenizer,
                       recurrent_checkpoint_interval = PAGE_SIZE)
    # max_new_tokens includes the already-pending first decode position: the job emits snapped tokens and its
    # completion lands exactly on the boundary y, while new_tokens reaches snapped + 1 > max_rq_tokens on the
    # EOS step, making that step the first (and only) requeue trigger
    resp_ids, eos_r = run_job_ids(rq_gen, rq_case_ids, snapped + 1, max_rq_tokens = 8 + PAGE_SIZE)
    check("requeue-at-EOS: job completes in one round with EOS",
          eos_r is not None and resp_ids.shape[-1] == snapped,
          f"generated = {resp_ids.shape[-1]}, expected {snapped}")
    # Only the suppression path can have delivered this EOS; the tail checkpoint proves the job then went
    # through deferred cleanup like any completion (previous-boundary fallback, since the completion ends
    # exactly on y)
    cont_ids = torch.cat([rq_case_ids, resp_ids, extra_ids], dim = -1)
    cold_gen = Generator(model = model, cache = cache_ctrl, tokenizer = tokenizer)
    cold_out, _ = run_job_ids(cold_gen, cont_ids, 16, cleanup_tail = False)
    warm_out, w_eos = run_job_ids(rq_gen, cont_ids, 16)
    check("requeue-at-EOS: suppressed-requeue completion leaves a usable tail checkpoint",
          w_eos["cached_tokens"] >= y - PAGE_SIZE,
          f"cached_tokens = {w_eos['cached_tokens']}, expected >= {y - PAGE_SIZE} "
          f"(prompt pages end at {x // PAGE_SIZE * PAGE_SIZE})")
    check("requeue-at-EOS: resumed output matches cold full-prefill",
          torch.equal(warm_out, cold_out),
          f"warm ids = {warm_out.tolist()}\ncold ids = {cold_out.tolist()}")

    print(f"\n -- {checks_passed} passed, {checks_failed} failed")
    sys.exit(1 if checks_failed else 0)


if __name__ == "__main__":
    main()
