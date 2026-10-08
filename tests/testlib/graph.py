"""
Model-level helpers for CUDA-graph (BC_*) path tests: greedy generation that keeps each decode step's logits, and
the teacher-forced check of those logits against the plain (no-cache) model forward over the same tokens.

    results = generate_with_logits(model, cache, tokenizer, prompts, max_new_tokens = 12)
    for prompt, (tokens, logits) in zip(prompts, results):
        assert_decode_matches_forward(model, tokenizer.encode(prompt, add_bos = True), tokens, logits, tol = 2e-2)

Cache-mode decode differs from the no-cache path only at kernel precision; a graph replayed against the wrong state
(stale strides, unpatched inputs) is off by orders of magnitude, so the relative Frobenius error per step separates
the two cleanly. On models whose MoE routing amplifies row-count (GEMM tiling) differences, kernel precision itself
is far coarser, so the tolerance scales with a measured noise floor: the same no-cache reference against forwards
of exactly prompt + k tokens.
"""

import gc

import torch


def load_with_caches(model_dir: str, device, cache_kwargs: list[dict] = ({},), config_hook = None):
    """(config, model, caches, tokenizer) with the model loaded on one device, below model_init: for tests that need
    several caches on one model or a config edit before construction (testlib.e2e.load_model covers everything a
    command line can express). Caches are created before the load
    (one per entry of cache_kwargs, max_num_tokens 4096 unless given); config_hook(config) may edit the config
    (e.g. its infer_params) before the model is built. MoE layers are switched to the deterministic eager
    accumulation (fused_mode_buffers = None)"""
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    config = Config.from_directory(model_dir)
    if config_hook is not None:
        config_hook(config)
    model = Model.from_config(config)
    caches = [Cache(model, **{"max_num_tokens": 4096, **kw}) for kw in cache_kwargs]
    model.load(device = str(device))
    tokenizer = Tokenizer.from_config(config)
    for m in model:
        if isinstance(m, BlockSparseMLP):
            m.fused_mode_buffers = None
    return config, model, caches, tokenizer


def unload_model(model):
    model.unload()
    gc.collect()
    torch.cuda.empty_cache()


def generate_with_logits(model, cache, tokenizer, prompts: list[str], max_new_tokens: int,
                         max_batch_size: int = 4) -> list[tuple[list[int], list[torch.Tensor]]]:
    """Greedy decode of all prompts as concurrent jobs; per job (tokens, [per-step logits before sampling]), at
    most max_new_tokens each (fewer if the job stopped on EOS)"""
    from exllamav3 import Generator, Job, ArgmaxSampler
    with torch.inference_mode():
        gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = max_batch_size)
        jobs = [
            Job(input_ids = tokenizer.encode(p, add_bos = True), max_new_tokens = max_new_tokens,
                sampler = ArgmaxSampler(), return_logits = True)
            for p in prompts
        ]
        toks = {j: [] for j in jobs}
        logits = {j: [] for j in jobs}
        for j in jobs:
            gen.enqueue(j)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    toks[r["job"]] += r["token_ids"].view(-1).tolist()
                    logits[r["job"]].append(r["logits"].view(-1, r["logits"].shape[-1])[0].clone())
    out = []
    for j in jobs:
        n = min(len(toks[j]), len(logits[j]), max_new_tokens)
        out.append((toks[j][:n], logits[j][:n]))
    return out


def _rfn(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error of logits a against b over the entries finite in both (masked vocab padding)"""
    a = a.float().to(b.device).view(-1)
    b = b.float().view(-1)
    m = torch.isfinite(a) & torch.isfinite(b)
    return ((a[m] - b[m]).norm() / b[m].norm()).item()


def decode_vs_forward_errors(model, input_ids: torch.Tensor, tokens: list[int],
                             step_logits: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    """Per-step relative Frobenius errors (decode, floor). decode: the decode logits against the no-cache forward
    of prompt + tokens (teacher-forced: step k is scored on the prompt and the first k generated tokens). floor:
    the same reference against a no-cache forward of exactly prompt + k tokens, i.e. what a different GEMM row
    count alone does to this model's logits (MoE routing near-ties amplify it far above kernel precision on some
    models)"""
    n = len(tokens)
    assert n == len(step_logits), f"{n} tokens but {len(step_logits)} logit rows"
    ids = torch.cat((input_ids, torch.tensor([tokens[:-1]], dtype = input_ids.dtype)), dim = 1)
    prompt_len = input_ids.shape[1]
    with torch.inference_mode():
        ref = model.forward(ids, params = {"last_tokens_only": n})[0].float()
        floor = [_rfn(model.forward(ids[:, :prompt_len + k], params = {"last_tokens_only": 1})[0, 0], ref[k])
                 for k in range(n)]
    errs = [_rfn(step_logits[k], ref[k]) for k in range(n)]
    return errs, floor


def assert_decode_matches_forward(model, input_ids: torch.Tensor, tokens: list[int], step_logits: list[torch.Tensor],
                                  tol: float, label: str = "", min_steps: int = 1, floor_factor: float = 2.0) -> float:
    """Assert every decode step's logits are within tolerance (relative Frobenius) of the teacher-forced no-cache
    forward; returns the worst step's error. The tolerance is tol, or floor_factor times the model's measured
    row-count noise floor where that is higher: a graph replayed against the wrong state is off by orders of
    magnitude more than either"""
    assert len(tokens) >= min_steps, f"{label}: too few decode steps to check ({len(tokens)})"
    errs, floor = decode_vs_forward_errors(model, input_ids, tokens, step_logits)
    worst = max(errs)
    limit = max(tol, floor_factor * max(floor))
    assert worst < limit, (
        f"{label}: decode logits deviate from the no-cache reference (per-step rfn "
        f"{[round(e, 4) for e in errs]}, tol {limit:.4f}; row-count noise floor {[round(e, 4) for e in floor]})"
    )
    return worst
