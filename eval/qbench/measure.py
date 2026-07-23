"""
KLD/ppl measurement against the cached reference logits.
"""

import math
import os

import torch

from exllamav3.util.measures import compute_kl_div, compute_target_log_probs

from .data import load_tensor, save_tensors

# One bf16 unit roundoff of multiplicative noise per element per layer. The measured floor is
# insensitive to the exact scale (it plateaus), so this needs no careful calibration.
BF16_ROUNDING_EPS = 2 ** -9

CONF_BUCKETS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.95), (0.95, 1.01)]

# Bumped when the per-model measurement set changes; invalidates cached KLD results without
# invalidating the (expensive) cached reference logits. v3: bpw convention unified across
# engines (biases/router gates excluded, tied-head fallback).
METRICS_VERSION = 3


class DiffStats:
    """Accumulates ppl over the model's own logits, and (if a reference store is given) KLD vs
    the reference, per-token, bucketed by the reference's top-token probability."""

    def __init__(self, ids: torch.Tensor, prefix_len: int, vocab_size: int, ref_store: str | None):
        self.ids = ids
        self.prefix_len = prefix_len
        self.vocab_size = vocab_size
        self.ref_store = ref_store
        self.logprob_sum = 0.0
        self.logprob_count = 0
        self.total_count = 0
        self.kl_toks = []
        self.conf_toks = []

    def __call__(self, r: int, logits: torch.Tensor):
        logits = logits[:, self.prefix_len:, :].float()
        logits.clamp_(min = -200.0)

        # ppl on own logits. Non-finite positions (a model NaN-ing on some input, e.g. fp16
        # overflow in a backend) are excluded and counted rather than poisoning the aggregate
        targets = self.ids[r, self.prefix_len + 1:].view(1, -1).to(logits.device)
        lp = compute_target_log_probs(logits[:, :-1, :], targets, min(self.vocab_size, logits.shape[-1]))
        finite = torch.isfinite(lp)
        self.logprob_sum += lp[finite].sum().item()
        self.logprob_count += finite.sum().item()
        self.total_count += targets.numel()

        # kld vs reference
        if self.ref_store:
            ref = load_tensor(os.path.join(self.ref_store, f"row_{r:06d}.safetensors"), "logits")
            ref = ref.to(logits.device).float()
            vs = min(self.vocab_size, logits.shape[-1], ref.shape[-1])
            kl = compute_kl_div(logits.squeeze(0), ref.squeeze(0), vs)
            self.kl_toks.append(kl.flatten().float().cpu())
            del ref

    def load_conf(self):
        conf = load_tensor(os.path.join(self.ref_store, "conf.safetensors"), "conf")
        self.conf_toks = [conf.flatten().float()]

    def results(self) -> dict:
        res = {}
        if self.total_count:
            nf_share = 1.0 - self.logprob_count / self.total_count
            if nf_share > 0.0:
                res["nonfinite_share"] = nf_share
        if self.logprob_count:
            res["ppl"] = math.exp(-self.logprob_sum / self.logprob_count)
        if self.kl_toks:
            self.load_conf()
            kl = torch.cat(self.kl_toks)
            conf = torch.cat(self.conf_toks)
            assert kl.numel() == conf.numel()
            finite = torch.isfinite(kl)
            if finite.any():
                klf = kl[finite]
                res.update({
                    "kld": klf.mean().item(),
                    "kld_median": klf.median().item(),
                    "kld_p10": klf.quantile(0.1).item(),
                    "kld_p25": klf.quantile(0.25).item(),
                    "kld_p75": klf.quantile(0.75).item(),
                    "kld_p90": klf.quantile(0.9).item(),
                    "kld_buckets": [
                        {
                            "conf_range": [lo, min(hi, 1.0)],
                            "share": ((conf >= lo) & (conf < hi)).float().mean().item(),
                            "mean": kl[finite & (conf >= lo) & (conf < hi)].mean().item()
                                if (finite & (conf >= lo) & (conf < hi)).any() else None,
                            "median": kl[finite & (conf >= lo) & (conf < hi)].median().item()
                                if (finite & (conf >= lo) & (conf < hi)).any() else None,
                        }
                        for lo, hi in CONF_BUCKETS
                    ],
                })
        return res


def save_reference_row(store_dir: str, r: int, logits: torch.Tensor, prefix_len: int, conf_rows: list):
    logits = logits[:, prefix_len:, :].float()
    logits.clamp_(min = -200.0)
    # reference top-token probability per position, for the confidence buckets
    l2 = logits.squeeze(0)
    confs = []
    for a in range(0, l2.shape[0], 256):
        c = l2[a:a + 256]
        confs.append((c.max(dim = -1).values - c.logsumexp(dim = -1)).exp().cpu())
    conf_rows.append(torch.cat(confs))
    save_tensors(
        os.path.join(store_dir, f"row_{r:06d}.safetensors"),
        {"logits": logits.half().cpu()},
    )


def print_stats(label: str, res: dict):
    if "ppl" not in res:
        print(f" -- {label:24}   INVALID (all logits non-finite; model broken under this backend)")
        return
    line = f" -- {label:24}   ppl {res['ppl']:9.4f}"
    if "kld" in res:
        line += f"   kld {res['kld']:.6f}   median {res['kld_median']:.6f}   p90 {res['kld_p90']:.6f}"
    if res.get("nonfinite_share"):
        line += f"   !! {100 * res['nonfinite_share']:.1f}% non-finite tokens excluded"
    print(line)
    if "kld_buckets" in res:
        for b in res["kld_buckets"]:
            if b["mean"] is None:
                continue
            lo, hi = b["conf_range"]
            print(
                f"      ref conf [{lo:4.2f}, {hi:4.2f}): {100 * b['share']:5.1f}% of tokens"
                f"   mean {b['mean']:.6f}   median {b['median']:.6f}"
            )
