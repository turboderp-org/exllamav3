"""
Dynamic sparse LM head for drafting.

A drafter that borrows the target's lm_head spends most of its step streaming the full vocabulary
projection, while the tokens a workload actually produces come from a small, slowly drifting subset of the
vocabulary. SparseHead keeps a dense fp16 copy of the unembedding columns of the most frequent token ids,
selected from running frequency counts of the tokens the generator sees, and answers the drafter's argmax
from that subset. Verification always runs the full head, so the subset only decides which tokens the
drafter can propose and never changes the output distribution.

The subset is rebuilt from the counts on a token cadence with hysteresis: an incumbent column is only
replaced by a challenger whose count exceeds it by a margin, so near-ties don't churn columns in and out
on every rebuild. Columns are extracted straight from the quantized weights in bulk, in the original basis,
into a buffer that grows with the number of distinct ids seen, so the draft GEMM reads only what is filled.

Whether the subset is used at all is a cost decision, not an acceptance guardrail: a target token outside
the subset is a draft miss the sparse head could never have avoided, and a rejection of a token inside it
is the drafter's own error that the full head would not have fixed either. The attributable miss rate is
measured on every target token for free; it pays off against the per-draft-token saving and the cost of a
lost window position until the break-even rate 2 * saving / round_time, with hysteresis around it. Probe
rounds through the full head at a low cadence check the one thing the miss count cannot see: whether the
sparse pick disagrees with the full pick while the full winner is in the subset.

Counts persist across runs beside the expert frequency profiles of the CPU offload path, keyed on the model
directory and vocabulary size, so a new session starts from the previous one's subset.
"""
from __future__ import annotations
import os
import hashlib
import math
import time
import torch
from ..ext import exllamav3_ext as ext

# Directory of the saved token frequency profiles; None = <user cache dir>/exllamav3/token_stats
# (the test suite points it at a scratch directory so runs neither read nor leave profiles)
TOKEN_STATS_DIR = None

_ALLOC_STEP = 1024          # column buffer growth granularity
_SAVE_INTERVAL = 4096       # observed generated tokens between profile writes
_MIN_SAVE_TOKENS = 64       # a session shorter than this leaves the saved profile alone
_MISS_EMA_TOKENS = 512      # time constant of the attributable miss rate, in target tokens
_ROUND_EMA = 0.05           # weight of a new round in the round-time average


def token_stats_path(model_dir: str, vocab_size: int) -> str:
    """
    Token frequencies belong to the tokenizer and the workload, not to one quantization of a model, so the
    profile is keyed on the tokenizer definition when the directory has one: every quant of a model (and every
    model sharing its tokenizer) reads and extends the same file. Directories without one fall back to the path
    """
    tok = os.path.join(model_dir, "tokenizer.json")
    try:
        with open(tok, "rb") as f:
            ident = hashlib.sha1(f.read()).hexdigest()
    except OSError:
        ident = os.path.abspath(model_dir)
    key = hashlib.sha1(f"{ident}|{vocab_size}".encode()).hexdigest()[:16]
    root = TOKEN_STATS_DIR
    if root is None:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        root = os.path.join(cache, "exllamav3", "token_stats")
    return os.path.join(root, key + ".pt")


class SparseHead:

    def __init__(
        self,
        linear,
        capacity: int,
        vocab_size: int,
        rebuild_interval: int = 128,
        decay_tokens: int = 8192,
        swap_margin: float = 0.25,
        prompt_weight: float = 0.2,
        min_evidence: float = 256.0,
        probe_interval: int = 128,
        profile_path: str | None = None,
        embedding = None,
    ):
        """
        :param linear:
            The Linear module whose columns are mirrored (its inner must implement reconstruct_columns)
        :param capacity:
            Maximum number of vocabulary columns held
        :param vocab_size:
            Unpadded vocabulary size; counts and candidates never cover padding columns
        :param rebuild_interval:
            Observed generated tokens between subset rebuilds
        :param decay_tokens:
            Observed generated tokens between halvings of the counts (exponential forgetting)
        :param swap_margin:
            A challenger replaces an incumbent only when its count exceeds the incumbent's by this fraction
        :param prompt_weight:
            Count weight of a prompt token relative to a generated token
        :param min_evidence:
            Weighted token count the counts must hold before the sparse argmax replaces the full head. Low on
            purpose: a thin subset costs misses for one rebuild interval at most, since the measured miss rate
            then switches the sparse path off until the counts have grown
        :param probe_interval:
            Drafting rounds between probe rounds through the full head (0 disables probing)
        :param profile_path:
            File the counts are restored from and saved to (None: no persistence)
        :param embedding:
            The Embedding module the drafting loop would look the next token up in (the target's). When given,
            the subset's embedding rows are mirrored on the head's device and the argmax hands back the winning
            token's embedding, so an MTP drafting loop never has to round-trip ids through the host for the
            next step's lookup. Its width may differ from the head's input width
        """
        self.linear = linear
        self.device = linear.device
        self.k = linear.in_features
        self.capacity = min(capacity, vocab_size)
        self.vocab_size = vocab_size
        self.rebuild_interval = rebuild_interval
        self.decay_tokens = decay_tokens
        self.swap_margin = swap_margin
        self.prompt_weight = prompt_weight
        self.min_evidence = min_evidence
        self.probe_interval = probe_interval
        self.profile_path = profile_path
        self.enabled = True

        dev = self.device
        self.counts = torch.zeros((vocab_size,), dtype = torch.float, device = dev)
        # Sized like the largest column buffer (capacity rounded to the kernel's 128-column tiles)
        self.cols = torch.full(((self.capacity + 127) // 128 * 128,), -1, dtype = torch.long, device = dev)
        self.member = torch.full((vocab_size,), -1, dtype = torch.long, device = dev)
        bias = linear.inner.get_bias_tensor() if hasattr(linear.inner, "get_bias_tensor") else None
        self.full_bias = bias
        self.embedding = embedding if embedding is not None and getattr(embedding, "hidden_size", None) else None
        self.emb_dim = self.embedding.hidden_size if self.embedding is not None else 0
        self.n_alloc = 0
        self.weight = None
        self.bias = None
        self.emb_rows = None
        self._scratch = None
        self._fused_available = hasattr(ext, "sparse_head_argmax") and self.device.type == "cuda"
        self.fused = self._fused_available
        self.n_active = 0
        self._ensure_alloc(min(_ALLOC_STEP, self.capacity))

        self.pending: list[tuple[torch.Tensor, float]] = []
        self.observed = 0           # generated tokens observed in total
        self.evidence = 0.0         # weighted count mass currently in the counts
        self.since_rebuild = 0
        self.since_decay = 0
        self.since_save = 0
        self.rebuilds = 0
        self.swapped = 0

        # Cost model: per-call head timings, round time, attributable miss rate
        self.t_full = None
        self.t_sparse = None
        self.round_time = None
        self.window = None          # draft tokens per round
        self.head_calls = None      # head invocations per round (MTP: one per draft token; DFlash: one)
        self.miss_rate = 0.0
        self.mode_sparse = True
        self.mode_switches = 0
        self.lookups = 0
        self.misses = 0
        self._interval_ids: list[torch.Tensor] = []     # target tokens seen since the last rebuild

        # Probe rounds
        self.rounds = 0
        self.probes = 0
        self.probe_disagree_inside = 0
        self.probe_disagree_outside = 0

        if profile_path:
            self.load_profile()
            import atexit, weakref
            ref = weakref.ref(self)
            def _save_at_exit():
                sh = ref()
                if sh is not None:
                    try:
                        sh.save_profile()
                    except Exception:
                        pass
            atexit.register(_save_at_exit)


    # Buffers

    def _ensure_alloc(self, n: int):
        """Grow the column buffer (and bias) to hold n columns, in _ALLOC_STEP multiples up to the capacity"""
        if n <= self.n_alloc:
            return
        # Rounded to the fused kernel's 128-column tiles (the buffer may run past the capacity; those
        # columns are never active)
        n_alloc = (min(self.capacity, (n + _ALLOC_STEP - 1) // _ALLOC_STEP * _ALLOC_STEP) + 127) // 128 * 128
        weight = torch.zeros((self.k, n_alloc), dtype = torch.half, device = self.device)
        if self.weight is not None:
            weight[:, :self.n_alloc] = self.weight
        self.weight = weight
        if self.full_bias is not None:
            bias = torch.zeros((n_alloc,), dtype = torch.float, device = self.device)
            if self.bias is not None:
                bias[:self.n_alloc] = self.bias
            self.bias = bias
        if self.embedding is not None:
            rows = torch.zeros((n_alloc, self.emb_dim), dtype = torch.half, device = self.device)
            if self.emb_rows is not None:
                rows[:self.n_alloc] = self.emb_rows
            self.emb_rows = rows
        # k-split partial sums of the fused argmax (up to 16 splits x 4 rows x columns, fp32)
        self._scratch = torch.empty((16 * 4 * n_alloc,), dtype = torch.float, device = self.device)
        self.n_alloc = n_alloc
        self._pick_path()


    def _pick_path(self):
        """
        Fused kernel or torch path for this buffer width, by timing both on the device: the fused GEMV wins
        while the buffer streams from L2, the tensor-core GEMM once it does not, and where that line falls
        depends on the GPU. Re-run whenever the buffer grows
        """
        self.fused = self._fused_available
        if not self.fused or not torch.cuda.is_available():
            return
        x = torch.randn((1, self.k), dtype = torch.half, device = self.device)
        n_active = self.n_active
        self.n_active = max(n_active, 1)
        try:
            def timed(fn, iters = 10):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize(self.device)
                return time.perf_counter() - t0
            self.fused = True
            t_fused = timed(lambda: self.argmax(x))
            self.fused = False
            t_torch = timed(lambda: self.argmax(x))
            self.fused = t_fused <= t_torch
        finally:
            self.n_active = n_active


    # Observation and maintenance

    @property
    def populated(self) -> bool:
        """The subset holds enough evidence to stand in for the full head (whether or not it currently does)"""
        return self.enabled and self.n_active > 0 and self.evidence >= self.min_evidence


    @property
    def ready(self) -> bool:
        return self.populated and self.mode_sparse


    def observe(self, ids: torch.Tensor, weight: float = 1.0):
        """
        Queue token ids (any shape, CPU or device) for the counts. Generated tokens carry weight 1 and feed the
        rebuild, decay and save cadence; prompt tokens are queued with prompt_weight and only shape the counts
        """
        ids = ids.reshape(-1)
        if ids.numel() == 0:
            return
        self.pending.append((ids, weight))
        self.evidence += weight * ids.numel()
        if weight >= 1.0:
            n = ids.numel()
            self.observed += n
            self.since_rebuild += n
            self.since_decay += n
            self.since_save += n


    def maintain(self):
        """
        Decay, rebuild and save on cadence; once per generator step. Queued observations stay on the host until
        the rebuild that reads them (the counts are consulted nowhere else), so the per-step cost is bookkeeping
        """
        if not self.pending:
            return
        if self.since_rebuild >= self.rebuild_interval or self.n_active == 0:
            self._flush()
        if self.since_decay >= self.decay_tokens:
            self._flush()
            self.counts *= 0.5
            self.evidence *= 0.5
            self.since_decay = 0
        if self.since_rebuild >= self.rebuild_interval or self.n_active == 0:
            self.rebuild()
            self._update_miss_rate()
            self._decide_mode()
            self.since_rebuild = 0
        if self.profile_path and self.since_save >= _SAVE_INTERVAL:
            self._flush()
            self.save_profile()


    def _flush(self):
        """Move the queued observations into the counts: one upload and one index_add per distinct weight"""
        if not self.pending:
            return
        populated = self.populated
        by_weight: dict[float, list[torch.Tensor]] = {}
        for ids, weight in self.pending:
            by_weight.setdefault(weight, []).append(ids.reshape(-1))
        self.pending.clear()
        for weight, parts in by_weight.items():
            ids = torch.cat(parts) if len(parts) > 1 else parts[0]
            ids = ids.to(self.device, non_blocking = False)
            # Only vocabulary ids count: a sequence also carries multimodal embedding indices (far past the
            # vocabulary) and may carry padding, neither of which the head can draft
            ids = ids[(ids >= 0) & (ids < self.vocab_size)]
            if ids.numel() == 0:
                continue
            if populated and weight >= 1.0:
                self._interval_ids.append(ids)
            self.counts.index_add_(0, ids, torch.full(ids.shape, weight, dtype = torch.float, device = self.device))


    def _update_miss_rate(self):
        """
        Fold the interval's target tokens into the miss-rate average, counted against the subset AFTER the rebuild
        that saw them: a token the rebuild just installed was a transient miss the subset has already fixed, so
        only tokens still outside count. Counted while the sparse head is switched off too, so it can come back
        (one device read per interval)
        """
        if not self._interval_ids:
            return
        ids = torch.cat(self._interval_ids)
        self._interval_ids.clear()
        n = ids.numel()
        misses = int((self.member[ids] < 0).sum().item())
        self.lookups += n
        self.misses += misses
        keep = math.exp(-n / _MISS_EMA_TOKENS)
        self.miss_rate = self.miss_rate * keep + (misses / n) * (1.0 - keep)


    def rebuild(self):
        """Move the subset toward the current top-capacity ids, with hysteresis against near-tie churn"""
        counts = self.counts
        cand_counts, cand = torch.topk(counts, self.capacity)
        valid = cand_counts > 0
        cand, cand_counts = cand[valid], cand_counts[valid]
        self.rebuilds += 1
        pos = self.member[cand]
        chal_mask = pos < 0
        chal, chal_counts = cand[chal_mask], cand_counts[chal_mask]       # strongest first
        if chal.numel() == 0:
            return
        n = self.n_active
        free = self.capacity - n
        if free > 0:
            take = min(free, chal.numel())
            self._install(torch.arange(n, n + take, device = self.device), chal[:take])
            chal, chal_counts = chal[take:], chal_counts[take:]
            n += take
        if chal.numel() == 0 or n == 0:
            return
        # Pair the weakest incumbents with the strongest remaining challengers
        inc_sorted, inc_order = torch.sort(counts[self.cols[:n]])
        m = min(chal.numel(), n)
        swap = chal_counts[:m] > inc_sorted[:m] * (1.0 + self.swap_margin)
        slots, new = inc_order[:m][swap], chal[:m][swap]
        if slots.numel():
            self._install(slots, new)


    def _install(self, slots: torch.Tensor, ids: torch.Tensor):
        top = int(slots.max().item()) + 1
        self._ensure_alloc(top)
        old = self.cols[slots]
        old = old[old >= 0]
        if old.numel():
            self.member[old] = -1
        self.cols[slots] = ids
        self.member[ids] = slots
        w = torch.empty((self.k, ids.numel()), dtype = torch.half, device = self.device)
        self.linear.inner.reconstruct_columns(ids, w)
        self.weight[:, slots] = w
        if self.bias is not None:
            self.bias[slots] = self.full_bias[ids].float()
        if self.emb_rows is not None:
            # The module's own forward, so scaling and normalization come along; ids go through the host as
            # the table lookup requires, once per installed column
            rows = self.embedding.forward(ids.cpu().view(1, -1), {}, out_dtype = torch.half)
            self.emb_rows[slots] = rows.view(-1, self.emb_dim).to(self.device)
        self.n_active = max(self.n_active, top)
        self.swapped += ids.numel()


    # Cost model

    def benchmark(self, iters: int = 20):
        """Time one full-head and one sparse argmax at batch size 1 on the head's device"""
        lin = self.linear
        x = torch.randn((1, 1, self.k), dtype = torch.half, device = self.device)
        def full():
            logits = lin.forward(x, {})
            return torch.argmax(logits, dim = -1)
        def timed(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize(self.device)
            return (time.perf_counter() - t0) / iters
        n_active = self.n_active
        self.n_active = max(n_active, 1)      # the sparse GEMM reads the allocated buffer whatever its content
        try:
            self.t_full = timed(full)
            self.t_sparse = timed(lambda: self.argmax(x))
        finally:
            self.n_active = n_active


    def note_round(self, seconds: float, drafted: int, head_calls: int):
        """One speculative round: wall time (draft and verification), draft tokens, head invocations"""
        if self.round_time is None:
            self.round_time, self.window, self.head_calls = seconds, float(drafted), float(head_calls)
        else:
            self.round_time += _ROUND_EMA * (seconds - self.round_time)
            self.window += _ROUND_EMA * (drafted - self.window)
            self.head_calls += _ROUND_EMA * (head_calls - self.head_calls)


    @property
    def breakeven_miss_rate(self) -> float | None:
        """
        Miss rate at which the sparse head stops paying: a miss at a random window position forfeits about
        half the round's accepted tokens, each worth a share of the round, while every head call saves
        (t_full - t_sparse). Per round: saving calls * S against cost m * W * T / 2, so m* = 2 calls S / (W T).
        An MTP drafter calls the head once per draft token (calls = W); DFlash drafters once per round
        """
        if self.t_full is None or self.round_time is None or self.round_time <= 0 or not self.window:
            return None
        return 2.0 * self.head_calls * (self.t_full - self.t_sparse) / (self.window * self.round_time)


    def _decide_mode(self):
        m_star = self.breakeven_miss_rate
        if m_star is None:
            return
        if self.mode_sparse and self.miss_rate > 1.25 * m_star:
            self.mode_sparse = False
            self.mode_switches += 1
        elif not self.mode_sparse and self.miss_rate < 0.8 * m_star:
            self.mode_sparse = True
            self.mode_switches += 1


    def round_flag(self):
        """
        Value of params["sparse_head"] for the next drafting round: True for the sparse path, "probe" for a
        probe round (full head, sparse pick recorded alongside), False when the sparse head is not in use.
        When the sparse head is off, probe rounds keep measuring the miss rate so it can come back
        """
        self.rounds += 1
        if not self.populated:
            return False
        if self.probe_interval and self.rounds % self.probe_interval == 0:
            return "probe"
        return self.mode_sparse


    def record_probe(self, full_ids: torch.Tensor, sparse_ids: torch.Tensor):
        inside = self.member[full_ids] >= 0
        differ = full_ids != sparse_ids
        self.probes += full_ids.numel()
        self.probe_disagree_inside += int((differ & inside).sum().item())
        self.probe_disagree_outside += int((~inside).sum().item())


    # Argmax

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        fp32 logits (rows, n_alloc) over the subset for x (rows, k), unfilled columns at -inf. fp32 like the
        full head's output: fp16 would turn its near-ties into exact ties
        """
        if x.shape[-1] < self.k:
            x = torch.nn.functional.pad(x, (0, self.k - x.shape[-1]))
        x = x.contiguous()
        n = self.n_active
        logits = torch.empty((x.shape[0], self.n_alloc), dtype = torch.float, device = x.device)
        ext.hgemm(x, self.weight, logits)
        if self.bias is not None:
            logits += self.bias
        if n < self.n_alloc:
            logits[:, n:] = float("-inf")
        return logits


    def _transform(self, v: torch.Tensor) -> torch.Tensor:
        """The Linear's output transforms (scales, softcap), all monotone: applied to scores, never needed for ranks"""
        lin = self.linear
        if lin.pre_scale != 1.0:
            v = v * lin.pre_scale
        if lin.softcap != 0.0:
            v = torch.tanh(v / lin.softcap) * lin.softcap
        if lin.post_scale != 1.0:
            v = v * lin.post_scale
        return v


    def argmax(self, x: torch.Tensor, embedding: bool = False):
        """
        Greedy token per position over the subset, with the winning logit as confidence; with embedding = True
        also the winners' embedding rows (None when no embedding is mirrored). One or two rows take the fused
        kernel (a scalar-FMA GEMV, memory-bound at that width); wider batches take the tensor-core torch path
        """
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        m = x.shape[0]
        want_emb = embedding and self.emb_rows is not None
        if self.fused and m <= 2 and x.dtype == torch.half and self.n_active > 0:
            if x.shape[-1] < self.k:
                x = torch.nn.functional.pad(x, (0, self.k - x.shape[-1]))
            x = x.contiguous()
            ids = torch.empty((m,), dtype = torch.long, device = self.device)
            conf = torch.empty((m,), dtype = torch.float, device = self.device)
            emb = torch.empty((m, self.emb_dim), dtype = torch.half, device = self.device) if want_emb else None
            ext.sparse_head_argmax(
                x, self.weight, self.bias, self.n_active, self.cols, ids, conf, self._scratch,
                self.emb_rows if want_emb else None, emb,
            )
        else:
            conf, idx = torch.max(self._logits(x), dim = -1)
            ids = self.cols[idx]
            emb = self.emb_rows[idx] if want_emb else None
        ids, conf = ids.view(shape), self._transform(conf).view(shape)
        if embedding:
            return ids, conf, emb.view(*shape, self.emb_dim) if emb is not None else None
        return ids, conf


    def topk(self, x: torch.Tensor, k: int, scale: float = 1.0, softcap: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k (scores, vocabulary ids) per position over the subset, scores scaled and softcapped the way the
        DFlash2 selector's walk_block prepares them, so they feed walk_candidates directly
        """
        shape = x.shape[:-1]
        logits = self._transform(self._logits(x.reshape(-1, x.shape[-1]))) * scale
        if softcap > 0.0:
            logits = torch.tanh(logits / softcap) * softcap
        unary, idx = torch.topk(logits, min(k, logits.shape[-1]), dim = -1, sorted = False)
        return unary.view(*shape, -1), self.cols[idx].view(*shape, -1)


    # Persistence

    def save_profile(self, path: str | None = None):
        """Write the decayed counts; a session too short to have seen a meaningful sample leaves the file alone"""
        path = path or self.profile_path
        if not path or self.since_save < _MIN_SAVE_TOKENS:
            return
        self._flush()
        import tempfile
        try:
            os.makedirs(os.path.dirname(path), exist_ok = True)
            fd, tmp = tempfile.mkstemp(dir = os.path.dirname(path), suffix = ".tmp")
            os.close(fd)
            torch.save({
                "model_dir": os.path.abspath(self.linear.config.directory) if getattr(self.linear.config, "directory", None) else None,
                "vocab_size": self.vocab_size,
                "counts": self.counts.cpu(),
                "evidence": self.evidence,
            }, tmp)
            os.replace(tmp, path)
            self.since_save = 0
        except OSError as e:
            print(f" !! token stats not saved ({path}): {e}")


    def load_profile(self, path: str | None = None) -> bool:
        """Restore counts saved by an earlier session and fill the subset from them; False if there are none"""
        path = path or self.profile_path
        if not path or not os.path.exists(path):
            return False
        try:
            d = torch.load(path, map_location = "cpu", weights_only = True)
        except Exception as e:
            print(f" !! token stats not loaded ({path}): {e}")
            return False
        counts = d.get("counts")
        if counts is None or counts.numel() != self.vocab_size:
            return False
        self.counts.copy_(counts.float())
        self.evidence = float(d.get("evidence", float(counts.sum())))
        self.rebuild()
        return True


    def stats(self) -> dict:
        self._flush()
        self._update_miss_rate()      # fold in the interval so far (one device read)
        return {
            "active": self.n_active,
            "allocated": self.n_alloc,
            "capacity": self.capacity,
            "observed": self.observed,
            "evidence": self.evidence,
            "rebuilds": self.rebuilds,
            "swapped": self.swapped,
            "lookups": self.lookups,
            "misses": self.misses,
            "miss_rate": self.miss_rate,
            "breakeven_miss_rate": self.breakeven_miss_rate,
            "t_full": self.t_full,
            "t_sparse": self.t_sparse,
            "round_time": self.round_time,
            "mode_sparse": self.mode_sparse,
            "mode_switches": self.mode_switches,
            "rounds": self.rounds,
            "probes": self.probes,
            "probe_disagree_inside": self.probe_disagree_inside,
            "probe_disagree_outside": self.probe_disagree_outside,
        }
