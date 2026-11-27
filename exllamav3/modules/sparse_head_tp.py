"""
Sparse draft head for a tensor-parallel target (EXL3_TP_SPARSE_DRAFT).

Under TP the target's lm_head is split by vocabulary over the ranks, so every MTP draft token costs a full
TP round trip: the state goes through the shared arena, every rank runs its head slice, the partial maxima
are gathered to the output device, and the next draft step then dispatches the embedding lookup to the
output device as well. On 3x R9700 that is ~1 ms per draft token, several ms per verify round.

TPSparseHead keeps the subset columns (and their embedding rows) on the output device, which runs in the
main process, so drafting needs no dispatch at all. The columns are reconstructed by the rank that owns
them, only when the subset changes (rebuild cadence of SparseHead), and shipped back over the worker pipe
as plain arrays. Verification still runs the full sharded head, so the output does not change; the subset
only decides which tokens the drafter can propose. Probe rounds and the cost model work as in SparseHead,
with t_full measured as the real TP path (argmax dispatch + gather + embedding dispatch).
"""
from __future__ import annotations
import os
import time
import numpy as np
import torch
from .sparse_head import SparseHead
from ..util.device_copy import to_device

_CHUNK = 1024           # columns per worker request
_LOG = int(os.environ.get("EXL3_TP_SPARSE_DRAFT_LOG", "0") or 0)
# EXL3_TP_SPARSE_REBENCH=<rounds>: re-time the subset and the full TP head after this many drafting rounds and
# hold the sparse/full mode decision until then (the load-time numbers come from the start transient); 0 = off
_REBENCH = int(os.environ.get("EXL3_TP_SPARSE_REBENCH", "64") or 0)


# Worker side (runs in every TP rank's process, including the in-process output device)

def mp_sparse_head_info(local_context: dict):
    lin = local_context.get("logits_module")
    if lin is None or lin.inner is None:
        return None
    bias = lin.inner.get_bias_tensor() if hasattr(lin.inner, "get_bias_tensor") else None
    return {
        "in_features": int(lin.in_features),
        "has_bias": bias is not None,
        "pre_scale": float(lin.pre_scale),
        "softcap": float(lin.softcap),
        "post_scale": float(lin.post_scale),
        "can_reconstruct": hasattr(lin.inner, "reconstruct_columns"),
    }


def _reconstruct_local(lin, ids: torch.Tensor) -> torch.Tensor:
    w = torch.empty((lin.in_features, ids.numel()), dtype = torch.half, device = ids.device)
    try:
        # Small groups: the fp32 transient is (k, 128 * max_blocks), and these cards are nearly full
        lin.inner.reconstruct_columns(ids, w, max_blocks = 16)
    except TypeError:
        lin.inner.reconstruct_columns(ids, w)
    return w


def mp_sparse_head_columns(local_context: dict, local_ids):
    lin = local_context["logits_module"]
    dev = torch.device("cuda", local_context["device"])
    ids = torch.from_numpy(np.ascontiguousarray(local_ids, dtype = np.int64)).to(dev)
    w = _reconstruct_local(lin, ids)
    return w.cpu().numpy()


# Main-process side

class _TPHeadProxy:
    """
    Stand-in for the Linear a SparseHead mirrors: the sharded head as a whole. reconstruct_columns routes
    each requested vocabulary id to the rank whose slice holds it
    """

    def __init__(self, model, info: dict, ranges: list[tuple[int, int, int]]):
        self.model = model
        self.device = torch.device("cuda", model.tp_output_device)
        self.in_features = info["in_features"]
        self.pre_scale = info["pre_scale"]
        self.softcap = info["softcap"]
        self.post_scale = info["post_scale"]
        self.config = model.config
        self.ranges = ranges
        self.inner = self
        self.reconstructed = 0
        self.reconstruct_time = 0.0

    def get_bias_tensor(self):
        return None

    def reconstruct_columns(self, cols: torch.Tensor, out: torch.Tensor):
        t0 = time.perf_counter()
        m = self.model
        out_dev = m.tp_output_device
        lc = m.mp_parent_conn[out_dev].local_context
        cols_cpu = cols.reshape(-1).cpu()
        for dev, a, b in self.ranges:
            mask = (cols_cpu >= a) & (cols_cpu < b)
            if not bool(mask.any()):
                continue
            pos = mask.nonzero().squeeze(1)
            local = cols_cpu[pos] - a
            for c in range(0, local.numel(), _CHUNK):
                lp = pos[c:c + _CHUNK].to(self.device)
                li = local[c:c + _CHUNK]
                if dev == out_dev:
                    w = _reconstruct_local(lc["logits_module"], li.to(self.device))
                else:
                    arr = m.tp_worker_dispatch_single(dev, mp_sparse_head_columns, (li.numpy(),))
                    w = torch.from_numpy(arr).to(self.device)
                out[:, lp] = w
        self.reconstructed += cols_cpu.numel()
        self.reconstruct_time += time.perf_counter() - t0


class TPSparseHead(SparseHead):

    def __init__(self, model, capacity: int, vocab_size: int, **kwargs):
        self.model = model
        head_key = model.modules[model.logit_layer_idx].key
        ranges, info = [], None
        for dev in model.active_devices:
            a, b, _ = model.plan[dev][head_key]
            if b > a:
                ranges.append((dev, a, min(b, vocab_size)))
                if info is None:
                    info = model.tp_worker_dispatch_single(dev, mp_sparse_head_info, ())
        if info is None:
            raise RuntimeError("no rank holds an lm_head slice")
        if info["has_bias"]:
            raise NotImplementedError("lm_head with bias")
        if not info["can_reconstruct"]:
            raise NotImplementedError("lm_head quant type cannot reconstruct columns")
        proxy = _TPHeadProxy(model, info, ranges)
        # The output device's own embedding module (it serves mp_model_forward_embedding in this process)
        from .embedding import Embedding
        lc = model.mp_parent_conn[model.tp_output_device].local_context
        # modules[0], as mp_model_forward_embedding uses it
        mods = lc["modules"]
        emb = mods[0] if mods and isinstance(mods[0], Embedding) else None
        if emb is None:
            print(" !! TP sparse draft head: no embedding on the output device, drafting keeps the lookup dispatch")
        super().__init__(proxy, capacity, vocab_size, embedding = emb, **kwargs)


    def benchmark(self, iters: int = 20):
        """t_full = one TP draft step without the subset: argmax dispatch + gather + embedding dispatch"""
        from ..model.model_tp_fn import mp_model_forward_embedding
        m = self.model
        x = torch.randn((1, 1, self.k), dtype = torch.half, device = self.device)

        def full():
            s = m.tp_producer.send(x)
            ids, _ = m.tp_dispatch_lm_head_argmax((s, {}), return_max = True)
            e = m.tp_producer.send(ids.view(1, 1))
            m.tp_dispatch_master(mp_model_forward_embedding, (e, {}))
            return ids

        def timed(fn):
            for _ in range(3):
                fn()
            best = None
            for _ in range(3):      # minimum of three repetitions: one stall must not decide the mode
                torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize(self.device)
                t = (time.perf_counter() - t0) / iters
                best = t if best is None else min(best, t)
            return best

        n_active = self.n_active
        self.n_active = max(n_active, 1)
        try:
            self.t_full = timed(full)
            self.t_sparse = timed(lambda: self.argmax(x, embedding = True))
        finally:
            self.n_active = n_active


    def argmax_state(self, state: torch.Tensor, embedding: bool = False):
        return self.argmax(to_device(state, self.device).half(), embedding = embedding)


    def _decide_mode(self):
        # Load-time timings are not trusted for the mode decision until the re-benchmark has run
        if _REBENCH and not getattr(self, "_rebenched", False):
            return
        # A non-positive break-even rate only comes from implausible timings: never switch the subset off on it
        m_star = self.breakeven_miss_rate
        if m_star is not None and m_star <= 0:
            return
        super()._decide_mode()


    def note_round(self, seconds: float, drafted: int, head_calls: int):
        super().note_round(seconds, drafted, head_calls)
        if _REBENCH and not getattr(self, "_rebenched", False) and self.rounds >= getattr(self, "_rebench_at", _REBENCH):
            self._rebenched = True
            old = (self.t_full, self.t_sparse, self.fused)
            try:
                self._pick_path()
                self.benchmark()
            except Exception as e:                                   # noqa: BLE001
                print(f" !! TP sparse draft head: re-benchmark failed ({e!r}), keeping the load-time numbers", flush = True)
                self.t_full, self.t_sparse, self.fused = old
            if self.t_sparse < self.t_full and not self.mode_sparse:
                self.mode_sparse = True
            print(
                f" -- TP sparse draft head re-benchmark after {self.rounds} rounds: draft step "
                f"{self.t_full * 1e3:.2f} ms (TP head) vs {self.t_sparse * 1e3:.3f} ms (subset), "
                f"{'fused' if self.fused else 'torch'} path (load time: {old[0] * 1e3:.2f} vs {old[1] * 1e3:.3f} ms, "
                f"{'fused' if old[2] else 'torch'}), mode {'sparse' if self.mode_sparse else 'full'}",
                flush = True
            )
            if self.t_sparse >= self.t_full and getattr(self, "_rebench_tries", 0) < 3:
                # Implausible (the subset is a fraction of the full head's work): measure again, later
                self._rebench_tries = getattr(self, "_rebench_tries", 0) + 1
                self._rebench_at = self.rounds * 4
                self._rebenched = False
        if _LOG and self.rounds % _LOG == 0:
            self.log()


    def log(self):
        s = self.stats()
        be = s["breakeven_miss_rate"]
        be_txt = f" (break-even {be * 100:.1f} %)" if be is not None else ""
        print(
            f" -- tp sparse draft: rounds {s['rounds']}, active {s['active']}/{s['capacity']}, "
            f"mode {'sparse' if s['mode_sparse'] else 'full'}, miss {s['miss_rate'] * 100:.1f} %{be_txt}, "
            f"head {s['t_full'] * 1e3:.2f} -> {s['t_sparse'] * 1e3:.3f} ms, "
            f"probe disagree inside {s['probe_disagree_inside']}/{s['probes']}, "
            f"columns rebuilt {self.linear.reconstructed} in {self.linear.reconstruct_time:.2f} s",
            flush = True
        )


def make_tp_sparse_head(model, capacity: int, **kwargs):
    sh = TPSparseHead(model, capacity, model.config.vocab_size, **kwargs)
    sh.benchmark()
    model.tp_sparse_head = sh
    print(
        f" -- TP sparse draft head: capacity {sh.capacity}, active {sh.n_active}, "
        f"draft step {sh.t_full * 1e3:.2f} ms (TP head + embedding) vs {sh.t_sparse * 1e3:.3f} ms (subset), "
        f"{'fused' if sh.fused else 'torch'} path",
        flush = True
    )
    return sh
