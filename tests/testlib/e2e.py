"""
Whole-model helpers for end-to-end tests: loading a registry model under a given configuration, and the
teacher-forced logit runs the feature tests compare.

    with load_model(model_dir, device, "-mcs", "8") as lm:
        logits = prefill_decode_logits(lm, ids, decode_steps = 16)
"""

import argparse
import contextlib
import gc
from dataclasses import dataclass

import torch

# Fixed English text for teacher-forced comparisons: long enough for a few hundred tokens, plain enough that
# every model's tokenizer handles it without special tokens
REFERENCE_TEXT = (
    "The history of the Roman Empire spans many centuries, beginning with the founding of the city and ending "
    "with the fall of Constantinople. During this time, the empire grew from a small settlement on the banks of "
    "the Tiber into a state that controlled the entire Mediterranean basin. Its roads, aqueducts and laws shaped "
    "the development of Europe for more than a thousand years. Historians have long debated the causes of its "
    "decline: economic troubles, military defeats, political instability and the pressure of migrating peoples "
    "all played a part. Meanwhile, in the east, the empire continued as a distinct civilization with its own "
    "language, art and religious traditions. Scholars in Constantinople preserved many works of classical "
    "literature, and their libraries later contributed to the revival of learning in western Europe. Trade "
    "routes connected the capital with distant markets, carrying silk, spices and precious metals. The city's "
    "walls withstood numerous sieges until gunpowder artillery finally breached them in the fifteenth century. "
)


@dataclass
class LoadedModel:
    config: object
    model: object
    cache: object
    tokenizer: object
    device: torch.device | None
    draft_model: object = None
    draft_cache: object = None
    args: object = None

    def encode(self, text: str = REFERENCE_TEXT, max_tokens: int | None = None) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_bos = True)
        return ids[:, :max_tokens] if max_tokens else ids

    def generator(self, **kwargs):
        """Generator over the loaded model, configured from the model_init arguments the way the example scripts do
        it (draft model/cache, drafting options, cache tiers); kwargs override"""
        from exllamav3 import Generator
        a = self.args
        opts = dict(
            draft_model = self.draft_model,
            draft_cache = self.draft_cache,
            num_draft_tokens = a.num_draft_tokens,
            ngram_match_min = a.ngram_match_min,
            ngram_corpus = a.ngram_corpus,
            dynamic_draft_tokens = a.dynamic_draft,
            draft_confidence = a.draft_confidence,
            cpu_cache_size = int(a.cpu_cache_size * 1024 ** 3),
            recurrent_cache_size = int(a.recurrent_cache_size * 1024 ** 3),
        )
        opts.update(kwargs)
        return Generator(model = self.model, cache = self.cache, tokenizer = self.tokenizer, **opts)


def gpu_split(devices: list[torch.device], total: int | None = None) -> str:
    """--gpu_split value that places a load on exactly these devices"""
    from testlib.env import num_devices
    n = total or num_devices()
    split = ["0"] * n
    for d in devices:
        split[d.index] = "1000"
    return ",".join(split)


@contextlib.contextmanager
def load_model(model_dir: str, device: torch.device | None, *args: str, cache_tokens: int = 4096, warmup: bool = False):
    """Load a model the way the example scripts do, through model_init: args are model_init command-line arguments
    (e.g. "-mcs", "8" for split expert offload, "-cq", "8" for an 8-bit cache, "-tp" plus "-gs" for tensor
    parallelism, "-dm", dir or "-mtp" for drafting). With a device, the whole load (and any draft model) goes
    there; with device = None the arguments place it. The post-load warmup is skipped unless warmup = True.
    Everything is unloaded on exit"""
    from exllamav3 import model_init

    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True)
    argv = ["-m", model_dir, "-cs", str(cache_tokens)] + list(args)
    if not warmup:
        argv.append("-nw")
    parsed = parser.parse_args(argv)
    kwargs = {"device": device} if device is not None else {}
    res = model_init.init(parsed, quiet = True, progress = False, **kwargs)
    model, config, cache, tokenizer = res[:4]
    draft_model, draft_cache = (res[4], res[6]) if len(res) > 4 else (None, None)
    try:
        yield LoadedModel(config, model, cache, tokenizer, torch.device(device) if device is not None else None,
                          draft_model, draft_cache, parsed)
    finally:
        if draft_model is not None:
            draft_model.unload()
        model.unload()
        del model, cache, draft_model, draft_cache
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def forward_logits(lm: LoadedModel, ids: torch.Tensor) -> torch.Tensor:
    """Cache-less full-sequence logits (1, T, vocab) fp32 on the CPU"""
    out = lm.model.forward(input_ids = ids, params = {"attn_mode": "flash_attn_nc"})
    return out[..., :lm.tokenizer.actual_vocab_size].float().cpu()


@torch.inference_mode()
def prefill_decode_logits(lm: LoadedModel, ids: torch.Tensor | list[torch.Tensor], decode_steps: int):
    """Teacher-forced logits through the generator's cached path: prefill ids[:, :-decode_steps], then feed the
    remaining tokens one at a time (single-token decode). Returns (decode_steps + 1, vocab) fp32 logits on the
    CPU: the prefill's last position followed by each decode step. A list of sequences runs as concurrent jobs
    (batched decode) and returns a list"""
    from exllamav3 import Job
    from exllamav3.generator.sampler import ArgmaxSampler

    seqs = ids if isinstance(ids, list) else [ids]
    gen = lm.generator()
    jobs = []
    for i, seq in enumerate(seqs):
        prompt = seq[:, :seq.shape[-1] - decode_steps]
        forced = seq[0, seq.shape[-1] - decode_steps:]
        job = Job(input_ids = prompt, max_new_tokens = decode_steps + 1, sampler = ArgmaxSampler(),
                  return_logits = True, stop_conditions = [], identifier = i)
        gen.enqueue(job)
        job.constrain_output_now(forced.view(1, -1))
        jobs.append(job)
    logits = [[] for _ in seqs]
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and r.get("logits") is not None:
                logits[r["identifier"]].append(r["logits"].view(-1, r["logits"].shape[-1]).float().cpu())
    out = [torch.cat(l, dim = 0)[:decode_steps + 1, :lm.tokenizer.actual_vocab_size] for l in logits]
    return out if isinstance(ids, list) else out[0]


@torch.inference_mode()
def greedy(gen, ids: torch.Tensor, new_tokens: int, return_logits: bool = False, **job_kwargs):
    """Greedy generation of exactly new_tokens through gen. Returns (tokens, logits or None, events) where events
    are the job's "started" and final ("eos") result dicts"""
    from exllamav3 import Job
    from exllamav3.generator.sampler import ArgmaxSampler
    job = Job(input_ids = ids, max_new_tokens = new_tokens, sampler = ArgmaxSampler(), stop_conditions = [],
              return_logits = return_logits, **job_kwargs)
    gen.enqueue(job)
    tokens, logits, events = [], [], {}
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "started":
                events["started"] = r
            if r["stage"] != "streaming":
                continue
            if r.get("token_ids") is not None:
                tokens += r["token_ids"].view(-1).tolist()
            if return_logits and r.get("logits") is not None:
                logits.append(r["logits"].view(-1, r["logits"].shape[-1]).float().cpu())
            if r["eos"]:
                events["eos"] = r
    return tokens, (torch.cat(logits) if logits else None), events


def assert_greedy_equivalent(ref_tokens: list[int], ref_logits: torch.Tensor, tokens: list[int],
                             near_tie: float = 0.25, tag: str = ""):
    """Two greedy runs of one model through different paths must agree token for token, except that they may
    part ways at a genuine near-tie: at the first differing position, the reference's top choice may lead the other
    run's token by less than near_tie logits. Positions after a permitted divergence are not compared"""
    n = min(len(ref_tokens), len(tokens))
    div = next((i for i in range(n) if ref_tokens[i] != tokens[i]), None)
    if div is None:
        assert len(tokens) == len(ref_tokens), f"{tag}: {len(tokens)} tokens vs {len(ref_tokens)}"
        return
    row = ref_logits[div]
    gap = (row[ref_tokens[div]] - row[tokens[div]]).item()
    assert gap < near_tie, (f"{tag}: output diverges at token {div} of {n} where the reference's choice leads by "
                            f"{gap:.3f} logits (not a near-tie)")


@torch.inference_mode()
def noise_floor(lm: LoadedModel, ids: torch.Tensor, positions: int, full_logits: torch.Tensor | None = None):
    """Per-position KL between two cache-less forward passes that differ only in length (ids and ids minus its last
    `positions` tokens), over the `positions + 1` positions before the cut. Measures how much the model's output
    moves when nothing but the batch geometry changes: near zero for most models, large for MoE models whose
    routing flips on near-ties. full_logits: forward_logits(lm, ids)[0], if already computed"""
    from testlib.compare import kl_divergence
    full = forward_logits(lm, ids)[0] if full_logits is None else full_logits
    shorter = forward_logits(lm, ids[:, :-positions])[0]
    n = positions + 1
    return kl_divergence(full[-positions - n:-positions], shorter[-n:])


def assert_logits_agree(ref: torch.Tensor, got: torch.Tensor, floor: torch.Tensor | None = None, tag: str = "",
                        abs_median_kl: float = 2e-3, abs_mean_kl: float = 0.05, floor_factor: float = 4.0,
                        min_confident_top1: float = 0.97, margin: float = 1.0):
    """The tolerance policy for comparing two paths of one model (position-wise logits, (T, vocab)):
    - median and mean KL below an absolute bound, or floor_factor times the model's own noise floor
      (testlib.e2e.noise_floor) where that is higher: routing-sensitive MoE models move on any arithmetic change
    - top-1 agreement at the confident positions (reference lead > margin logits), where only a real error flips
      the choice; near-ties are free to go either way"""
    from testlib.compare import confident_top1_agreement, kl_divergence
    assert torch.isfinite(got).all(), f"{tag}: non-finite logits"
    kl = kl_divergence(ref, got)
    f_med = floor.median().item() if floor is not None else 0.0
    f_mean = floor.mean().item() if floor is not None else 0.0
    summary = (f"{tag}: KL median {kl.median().item():.2e} mean {kl.mean().item():.2e} max {kl.max().item():.2e}"
               + (f"; noise floor median {f_med:.2e} mean {f_mean:.2e}" if floor is not None else ""))
    assert kl.median().item() < max(abs_median_kl, floor_factor * f_med), summary
    assert kl.mean().item() < max(abs_mean_kl, floor_factor * f_mean), summary
    agree, n = confident_top1_agreement(ref, got, margin)
    assert agree >= min_confident_top1, f"{tag}: top-1 agreement {agree:.3f} over {n} confident positions; {summary}"
