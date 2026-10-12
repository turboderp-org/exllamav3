"""
Model-level helpers for CUDA-graph (BC_*) path tests: greedy generation that keeps each decode step's logits, and
the comparison of a scenario's decode run with graphs captured against the same run with capture disabled.

    def scenario(model_dir):                        # module-level, runs in a child process
        _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
        return [("label", *generate_with_logits(model, cache, tok, prompts, 12)[0])]

    graphed, eager = run_with_and_without_graphs(scenario, model_dir, device = device)
    assert_matches_eager(graphed, eager)

EXL3_GRAPHS=0 runs every graphed site's C++ launch sequence eagerly (the same kernels in the same order), so a
correctly replayed graph reproduces the eager run's logits bit for bit, while a graph replayed against the wrong
state (stale strides, unpatched inputs) is off by orders of magnitude. The eager run is a sharper reference than the
no-cache forward: decode and prefill differ by the decode path's own precision (int8-activation GEMVs, decode
attention), which MoE routing near-ties amplify into isolated large deviations. Both runs are child processes with
EXL3_GRAPHS set explicitly, since the default differs per backend (off on ROCm).
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


def run_with_and_without_graphs(fn, *args, device, timeout: float = 900.0):
    """(graphed, eager): fn(*args) in two child processes pinned to `device`, with graph capture enabled and
    disabled. fn is a module-level function of the test module that loads on get_test_device() and returns its
    generations as [(label, tokens, step logits)], logits on the CPU"""
    from testlib.isolated import device_env, run_isolated
    graphed = run_isolated(fn, *args, env = {**device_env(device), "EXL3_GRAPHS": "1"}, timeout = timeout)
    eager = run_isolated(fn, *args, env = {**device_env(device), "EXL3_GRAPHS": "0"}, timeout = timeout)
    return graphed, eager


def assert_matches_eager(graphed, eager, tol: float = 1e-3, min_steps: int = 1):
    """Every generation of the graphed run has the eager run's tokens, and per-step logits within tol (relative
    Frobenius; expected exact, the tolerance only absorbs atomics-order noise where a kernel has it)"""
    assert len(graphed) == len(eager), f"{len(graphed)} graphed generations, {len(eager)} eager"
    for (label, tok_g, lg_g), (label_e, tok_e, lg_e) in zip(graphed, eager):
        assert label == label_e
        assert len(tok_g) >= min_steps, f"{label}: too few decode steps to check ({len(tok_g)})"
        assert tok_g == tok_e, f"{label}: graphed tokens {tok_g} differ from eager {tok_e}"
        errs = [_rfn(a, b) for a, b in zip(lg_g, lg_e)]
        assert max(errs) <= tol, f"{label}: graphed decode logits deviate from eager (per-step rfn " \
                                 f"{[round(e, 6) for e in errs]}, tol {tol})"
