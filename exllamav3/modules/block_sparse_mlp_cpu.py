from __future__ import annotations
import os
import torch
from ..ext import exllamav3_ext as ext

# Kernel-fused issue/collect for decode-size split jobs (EXL3_MOE_SPLIT_FUSED=0 restores the
# cudaMemcpyAsync path for A/B testing)
_split_fused = os.environ.get("EXL3_MOE_SPLIT_FUSED", "1") != "0"

# EXL3_SPLIT_PROF=1: CUDA-event brackets around the fused issue enqueue and the collect
# wait+readback, reported as stream-time percentiles every ~2048 brackets. The collect
# bracket is the true exposed stall: it runs after the GPU's own expert work, so any time
# there is worker lateness the GPU could not hide.
_split_prof = bool(os.environ.get("EXL3_SPLIT_PROF"))
_sprof = {"issue": [], "wait": []}

def _sprof_wrap(kind, layer, fn):
    ev0 = torch.cuda.Event(enable_timing = True)
    ev1 = torch.cuda.Event(enable_timing = True)
    ev0.record()
    r = fn()
    ev1.record()
    recs = _sprof[kind]
    recs.append((layer, ev0, ev1))
    if kind == "wait" and len(recs) >= 2048:
        for k, rs in _sprof.items():
            ts = [(l, a.elapsed_time(b)) for l, a, b in rs if b.query()]
            if not ts:
                continue
            vals = sorted(t for _, t in ts)
            per_layer = {}
            for l, t in ts:
                s = per_layer.setdefault(l, [0.0, 0])
                s[0] += t; s[1] += 1
            worst = sorted(per_layer.items(), key = lambda kv: -kv[1][0] / kv[1][1])[:4]
            print(f" -- split prof [{k}] ({len(vals)} brackets, stream ms): "
                  f"med {vals[len(vals) // 2]:.3f} p90 {vals[int(len(vals) * 0.9)]:.3f} "
                  f"max {vals[-1]:.3f} | worst layers "
                  + " ".join(f"L{l}:{s / n:.3f}" for l, (s, n) in worst), flush = True)
            _sprof[k] = []
    return r

"""
Forward-path hooks are cpu_split_submit / cpu_offload_issue / cpu_split_combine, the
load-path hooks cpu_maybe_offload_load / cpu_maybe_split_load / cpu_post_load, plus
cpu_unload. The worker process itself lives in model/moe_cpu_host.py.
"""


# Directory of the saved expert frequency profiles; None = <user cache dir>/exllamav3/expert_stats
# (the test suite points it at a scratch directory so runs neither read nor leave profiles)
EXPERT_STATS_DIR = None


def _expert_stats_path(config, num_experts):
    """Per-model file for the routed-expert frequencies the dynamic placement collects, keyed on
    the model directory and the expert count (a stale or foreign profile only costs what the
    default identity placement costs anyway)"""
    import hashlib
    key = hashlib.sha1(f"{os.path.abspath(config.directory)}|{num_experts}".encode()).hexdigest()[:16]
    root = EXPERT_STATS_DIR
    if root is None:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        root = os.path.join(cache, "exllamav3", "expert_stats")
    return os.path.join(root, key + ".json")


def _expert_stats_load(config, num_experts):
    import json
    try:
        with open(_expert_stats_path(config, num_experts)) as f:
            d = json.load(f)
        return d.get("layers", {})
    except (OSError, ValueError):
        return {}


def save_expert_stats(infer_params):
    """Write every split module's decayed selection counts (in checkpoint expert order, undoing
    the load-time permutation) so the next load of this model starts from the hot/cold ranking
    instead of the arbitrary head/tail split. A session too short to have routed a meaningful
    sample keeps the file's previous entries (it must not overwrite a settled profile); the
    counts themselves are decayed, so the threshold is in decode steps, not hits"""
    import json, tempfile
    reg = getattr(infer_params, "moe_cpu_swap_modules", None)
    if not reg or reg[0]._split_step_count < _STATS_MIN_STEPS:
        return
    config, E = reg[0].config, reg[0].num_experts
    layers = _expert_stats_load(config, E)
    for m in reg:
        if m._split_hist is None or getattr(m, "tid2eid_key", None):
            continue
        hist = m._split_hist.tolist()
        if m.num_experts != E or sum(hist) <= 0.0:
            continue
        perm = m._split_perm or list(range(E))
        counts = [0.0] * E
        for r, c in enumerate(hist):
            counts[perm[r]] = c
        layers[m.key] = counts
    if not layers:
        return
    path = _expert_stats_path(config, E)
    try:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        fd, tmp = tempfile.mkstemp(dir = os.path.dirname(path), suffix = ".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"model_dir": os.path.abspath(config.directory), "layers": layers}, f)
        os.replace(tmp, path)
    except OSError as e:
        print(f" !! expert stats not saved ({path}): {e}")


@torch.inference_mode()
def run_pending_swap_sweeps(infer_params):
    """Run any pending dynamic-placement sweep (see BlockSparseMLP._split_swap_tick). Called
    by the generator when its job queue drains, so placement only ever changes between
    generations; safe no-op otherwise. The sweep updates the placement map, the hit
    histogram and the arena slots in place, and those are inference tensors (allocated under
    inference mode at load), so it must itself run under inference mode: the generator's
    cancel() and clear_queue() paths reach here outside any forward (issue #329)."""
    if not getattr(infer_params, "moe_cpu_swap_pending", False):
        return
    infer_params.moe_cpu_swap_pending = False
    reg = getattr(infer_params, "moe_cpu_swap_modules", None)
    if not reg:
        return
    reg[0]._split_sweep_layer_reset()
    # Quiesce every device holding a split layer, not just the current one: with a
    # multi-GPU layer split, worker jobs are issued from each device's stream, and their
    # completion is only guaranteed by the collect memop waits on those streams. A
    # single-device synchronize leaves other devices' jobs in flight while the sweep
    # rewrites arena slots. After the sync, every issued job's collect has completed, so
    # the ring is drained and the worker idle by construction.
    for d in sorted({m.device.index for m in reg if m.device is not None}):
        torch.cuda.synchronize(d)
    if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
        for h in {m.cpu_host for m in reg}:
            assert int(h.v_jobs_head[0]) == int(h.v_jobs_tail[0]), \
                "swap sweep: job ring not drained after all-device sync"
    budget = int(os.environ.get("EXL3_MOE_CPU_SWAP_MAX", 64))
    total = 0
    for m in reg:
        if budget <= 0:
            break
        n = m._split_sweep_layer(budget)
        budget -= n
        total += n
    if total and os.environ.get("EXL3_MOE_CPU_SWAP_DEBUG"):
        print(f" -- expert swap sweep: {total} swaps", flush = True)


_STATS_SAVE_INTERVAL = 4096   # decode steps between writes of the expert frequency profile
_STATS_MIN_STEPS = 64         # decode steps a session needs before its profile is worth keeping
_PROMOTE_BUDGET = 6         # promotions issued per step over all layers (and as many demotions)
_PROMOTE_RATIO = 1.5        # a miss is promoted when its decayed count is this many times the victim's
_PROMOTE_DECAY = 0.98       # per-step decay of the hit counts


class _PromoteState:
    """Promote-on-miss bookkeeping over every split module with spares, as [modules, experts] host
    tensors so one step is a few vector ops: the hit-count snapshots (pinned, one device-to-host copy
    per module per step, one event per device), the previous snapshot, the placement map mirror and
    the in-flight mask. Per module, the free slot lists and the transfers in flight stay Python lists"""

    def __init__(self, modules):
        self.modules = [m for m in modules if m._split_spares and m._split_pending is not None]
        n = len(self.modules)
        e = max(m.num_experts for m in self.modules)
        import numpy as np
        # The step's arithmetic is numpy over small arrays: a torch CPU op opens an OpenMP region, and
        # with the host confined to its reserved cores beside the worker's pinned threads
        # (EXL3_MOE_HOST_CORES) that region costs tens of milliseconds
        self.first = np.array([m.cpu_split_first for m in self.modules], dtype = np.int64).reshape(n, 1)
        self.mp = np.full((n, e), 1 << 30, dtype = np.int64)
        for i, m in enumerate(self.modules):
            self.mp[i, :m.num_experts] = m._split_mp
        self.snap = torch.zeros((n, e), dtype = torch.float).pin_memory()
        self.snap_np = self.snap.numpy()
        self.prev = np.zeros((n, e), dtype = np.float32)
        self.busy = np.zeros((n, e), dtype = bool)
        self.devices = sorted({torch.device(m.device).index or 0 for m in self.modules})
        self.snap_ev = {}
        self.stats = dict(routed = 0, missed = 0, steps = 0, ms = 0.0)
        for m in self.modules:
            m._split_snap = None          # the per-module snapshot is superseded by the shared one


def _split_promote_step(ip, modules, rows):
    """One decode step of promote-on-miss for every split module (called from the first module's
    submit, before any layer's selection translate of this step).

    (1) Transfers that landed take effect in the map, on the compute stream ahead of this step's
    translate kernels; the slot they vacated is free from this step on, since nothing maps to it any
    more. (2) The previous step's hit counts (snapshotted to pinned memory) name the routed experts;
    those the worker computed are the misses. (3) The misses, ranked by count over all layers, take
    free GPU slots of their layer when their count clears the layer's least-used unrouted resident's
    by _PROMOTE_RATIO, up to _PROMOTE_BUDGET per step: the block is copied from the arena on the
    promotion stream, ordered after everything the compute stream had enqueued (the slot's last
    reader was a previous step). (4) The worker slots promotions freed take the least-used unrouted
    residents, copied out on the promotion stream; the map flips once the copy has landed. Every
    transfer leaves the expert live where it was until its copy is complete, so a route always finds
    valid weights. Nothing is issued on a prefill-size step (its stager reads worker slots)."""
    import time
    t0 = time.perf_counter()
    ps = getattr(ip, "moe_cpu_promote_state", None)
    if ps is None or ps.modules[0] is not modules[0]:
        ps = ip.moe_cpu_promote_state = _PromoteState(modules)
    verify = os.environ.get("EXL3_MOE_CPU_PROMOTE_VERIFY")
    debug = os.environ.get("EXL3_MOE_CPU_SWAP_DEBUG")
    # (1) landed transfers
    for i, m in enumerate(ps.modules):
        if not m._split_pending:
            continue
        keep, idx, val = [], [], []
        for t in m._split_pending:
            if not t["ev"].query():
                keep.append(t)
                continue
            r = t["r"]
            if t["kind"] == "p":
                m._split_mp[r] = t["gpu"]
                m._split_free_cpu.append(t["cpu"])
                m._split_promo_n += 1
            else:
                m._split_mp[r] = m.cpu_split_first + t["cpu"]
                m._split_free_gpu.append(t["gpu"])
                m._split_demo_n += 1
            idx.append(r)
            val.append(m._split_mp[r])
            ps.busy[i, r] = False
            if verify:
                _split_promote_verify(m, t)
        m._split_pending = keep
        if idx:
            ps.mp[i, idx] = val
            dev = torch.device(m.device)
            with torch.cuda.device(dev):
                m._split_map[torch.tensor(idx, device = dev)] = torch.tensor(val, device = dev)
    # (2) last step's routes: the snapshots land, the next ones go out
    have = bool(ps.snap_ev)
    for ev in ps.snap_ev.values():
        ev.synchronize()
    if have:
        cur = ps.snap_np.copy()
        routed = cur > ps.prev
        ps.prev = cur
    for i, m in enumerate(ps.modules):
        dev = torch.device(m.device)
        with torch.cuda.device(dev):
            ps.snap[i, :m.num_experts].copy_(m._split_hist, non_blocking = True)
    for d in ps.devices:
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(d))
        ps.snap_ev[d] = ev
    hists = [m._split_hist for m in ps.modules]
    torch._foreach_mul_(hists, _PROMOTE_DECAY)
    if have and rows < ps.modules[0].cpu_host.stream_min_rows:
        import numpy as np
        resident = ps.mp < ps.first
        quiet = resident & ~routed & ~ps.busy                 # demotion candidates
        inf = np.full(ps.prev.shape, np.inf, dtype = np.float32)
        victim = np.maximum(np.where(quiet, ps.prev, inf).min(axis = 1), 1.0)   # [n]
        misses = routed & ~resident & ~ps.busy
        if debug:
            ps.stats["routed"] += int(routed.sum())
            ps.stats["missed"] += int((routed & ~resident).sum())
        # (3) promotions: the misses that clear the ratio, best counts first, one per layer per step
        ok = misses & (ps.prev >= _PROMOTE_RATIO * victim.reshape(-1, 1))
        if ok.any():
            score = np.where(ok, ps.prev, -1.0).reshape(-1)
            order = np.argsort(-score)[:_PROMOTE_BUDGET * 4]
            seen = set()
            for flat in order.tolist():
                if score[flat] < 0 or len(seen) >= _PROMOTE_BUDGET:
                    break
                i, r = divmod(flat, ps.mp.shape[1])
                m = ps.modules[i]
                if i in seen or not m._split_free_gpu:
                    continue
                host, first = m.cpu_host, m.cpu_split_first
                c = m._split_mp[r] - first
                if not host.promote_ready(m.cpu_layer_idx, c):
                    continue
                g = m._split_free_gpu.pop()
                dev = torch.device(m.device)
                with torch.cuda.device(dev):
                    host.promo_stream(dev).wait_stream(torch.cuda.current_stream(dev))
                    ev = host.promote_expert(m.cpu_layer_idx, c, _split_slot_linears(m, g))
                m._split_pending.append(dict(kind = "p", r = r, gpu = g, cpu = c, ev = ev))
                ps.busy[i, r] = True
                seen.add(i)
        # (4) demotions into the worker slots promotions freed: the least-used unrouted residents
        for i, m in enumerate(ps.modules):
            k = len(m._split_free_cpu)
            if not k:
                continue
            cand = np.where(quiet[i], ps.prev[i], inf[i])
            rs = np.argsort(cand)[:k]
            host, first = m.cpu_host, m.cpu_split_first
            dev = torch.device(m.device)
            for r in rs.tolist():
                if cand[r] == np.inf or not m._split_free_cpu:
                    break
                c = m._split_free_cpu.pop()
                if not host.promote_ready(m.cpu_layer_idx, c):
                    m._split_free_cpu.append(c)
                    break
                g = m._split_mp[r]
                with torch.cuda.device(dev):
                    host.promo_stream(dev).wait_stream(torch.cuda.current_stream(dev))
                    ev = host.demote_expert(m.cpu_layer_idx, c, _split_slot_linears(m, g))
                m._split_pending.append(dict(kind = "d", r = r, gpu = g, cpu = c, ev = ev))
                ps.busy[i, r] = True
    if debug:
        st = ps.stats
        st["steps"] += 1
        st["ms"] += time.perf_counter() - t0
        if modules[0]._split_step_count % 128 == 0:
            p = sum(m._split_promo_n for m in ps.modules)
            d = sum(m._split_demo_n for m in ps.modules)
            pend = sum(len(m._split_pending) for m in ps.modules)
            print(f" -- promote-on-miss: step {modules[0]._split_step_count}, {p} promotions, {d} demotions, {pend} in flight; "
                  f"last {st['steps']} steps: {st['routed'] / st['steps']:.1f} routed/step, "
                  f"{st['missed'] / st['steps']:.1f} on the CPU ({100 * (1 - st['missed'] / max(st['routed'], 1)):.1f}% hit), "
                  f"hook {1000 * st['ms'] / st['steps']:.2f} ms/step", flush = True)
            st.update(routed = 0, missed = 0, steps = 0, ms = 0.0)


def _split_slot_linears(m, g):
    """(gate or None, up, down) Linear modules of GPU slot g"""
    return ((m.gates[g].inner if m.gated else None), m.ups[g].inner, m.downs[g].inner)


def _split_promote_verify(m, t):
    """EXL3_MOE_CPU_PROMOTE_VERIFY: a landed transfer's destination against the checkpoint: a
    promotion's GPU slot tensors, a demotion's arena bytes (packed) and aux copies"""
    from ..model.moe_cpu_host import _copy_repacked
    stc = m.config.stc
    full_g, full_u, full_d = m._split_saved_lists_ref()
    r = t["r"]
    host = m.cpu_host
    torch.cuda.synchronize()
    projs = ([("g", full_g, m.gates)] if m.gated else []) + [("u", full_u, m.ups), ("d", full_d, m.downs)]
    if t["kind"] == "p":
        for _, full, lin in projs:
            dst = lin[t["gpu"]].inner
            key = full[r].key
            for tname, live in (("trellis", dst.trellis), ("suh", dst.suh), ("svh", dst.svh)):
                fresh = stc.get_tensor(key + "." + tname, live.device, float2half = tname != "trellis")
                assert torch.equal(fresh, live), f"promote verify failed: {key}.{tname}"
        return
    spec = host.specs[m.cpu_layer_idx]
    layout, _ = host._stream_layout(spec)
    eb = spec["expert_bytes"]
    ci, off = host.layer_blocks[m.cpu_layer_idx][t["cpu"]]
    arena = host.arena_views[ci][off // 2 : (off + eb) // 2]
    aux = host.aux[m.cpu_layer_idx]
    locs = host.layer_auxloc[m.cpu_layer_idx][t["cpu"]]
    for li, name, lin, o, nb in host._promo_parts(m.cpu_layer_idx, _split_slot_linears(m, t["gpu"])):
        key = dict(projs)[name][0][r].key if False else [f for n_, f, _ in projs if n_ == name][0][r].key
        fresh = stc.get_tensor(key + ".trellis", "cpu").view(torch.int16)
        packed = torch.empty_like(fresh)
        group, planar = layout[("g", "u", "d").index(name)]
        _copy_repacked(packed, fresh, group, planar)
        got = arena[o // 2 : (o + nb) // 2].view(fresh.shape)
        assert torch.equal(got, packed), f"demote verify failed: {key}.trellis (arena)"
        for j, an in enumerate(("suh", "svh", "bias")):
            lst = aux.get(f"{an}_{name}")
            if lst is None or lst[t["cpu"]] is None:
                continue
            fr = stc.get_tensor(key + "." + an, "cpu", optional = an == "bias", float2half = True)
            if fr is None:
                continue
            assert torch.equal(lst[t["cpu"]].cpu(), fr), f"demote verify failed: {key}.{an} (aux)"
            lc, lo, ln = locs[li][j]
            assert torch.equal(host.arena_views[lc][lo // 2 : (lo + ln) // 2], fr.view(torch.int16).view(-1)), \
                f"demote verify failed: {key}.{an} (arena)"


def trellis_rate(tile_width: int):
    """Bitrate of an EXL3 trellis from its tile width (last dim): an integer 1..8 for 16 * K, a
    half-integer 1.5 / 2.5 / 3.5 for 16 * K + 8, None for anything the CPU kernels don't take"""
    K, rem = divmod(tile_width, 16)
    if rem == 0 and 1 <= K <= 8:
        return K
    if rem == 8 and 1 <= K <= 3:
        return K + 0.5
    return None


class BlockSparseMLP_CPU:

    def _cpu_init_state(self):
        self._split_spares = 0
        self._split_pending = None
        self._split_submit_rows = 0
        self.cpu_offload = False
        self.cpu_split_first = None   # split offload: first CPU-resident (tail) expert index
        self._split_map = None        # dynamic placement: router id -> physical slot

    def cpu_maybe_offload_load(self, device, **kwargs) -> bool:
        """Whole-layer offload claim: when the budget allows and the layer is eligible,
        register the routed experts with the CPU worker instead of loading them; the caller
        skips its GPU load entirely on True."""
        ip = self.config.infer_params
        comp = getattr(ip, "moe_cpu_component", "text")
        budget = getattr(ip, "moe_cpu_offload", 0) if comp == "text" \
            else getattr(ip, "draft_moe_cpu_offload", 0)
        if budget:
            assert not getattr(ip, "moe_cpu_split", 0), \
                "moe_cpu_split and moe_cpu_offload are mutually exclusive: the split offloads " \
                "a slice of every eligible layer's experts, whole-layer offload takes entire " \
                "layers — pick one"
        if (
            budget > 0 and
            ip.moe_cpu_offload_assigned.get(comp, 0) < budget and
            device is not None and torch.device(device).type == "cuda" and
            (self.num_local_experts is None or self.num_local_experts == self.num_experts) and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2")
        ):
            if self.load_cpu_offload(device, **kwargs):
                ip.moe_cpu_offload_assigned[comp] = ip.moe_cpu_offload_assigned.get(comp, 0) + 1
                return True

        return False

    def cpu_maybe_split_load(self, device, **kwargs):
        """Per-layer expert split registration (EXL3_MOE_CPU_SPLIT = tail experts per layer
        on the CPU worker): shrinks the module to its GPU slice; the caller's normal load
        then loads that slice."""
        ip = self.config.infer_params
        # Registration shrinks the module to its GPU slice, then the normal load below loads
        # that slice. infer_params.moe_cpu_split is authoritative (the EXL3_MOE_CPU_SPLIT env
        # is its construction-time default)
        split_k = int(getattr(ip, "moe_cpu_split", 0))
        if split_k:
            assert not getattr(ip, "moe_cpu_offload", 0) and not getattr(ip, "draft_moe_cpu_offload", 0), \
                "moe_cpu_split and moe_cpu_offload are mutually exclusive: the split offloads " \
                "a slice of every eligible layer's experts, whole-layer offload takes entire " \
                "layers — pick one"
        split_layers = int(os.environ.get("EXL3_MOE_CPU_SPLIT_LAYERS", 0))
        if split_layers and getattr(ip, "moe_cpu_split_assigned", 0) >= split_layers:
            split_k = 0
        # The split keeps at least one expert on the GPU (the module's GPU slice cannot be
        # empty); a larger request is capped rather than silently turning the split off
        if split_k >= self.num_experts > 1:
            if not getattr(ip, "moe_cpu_split_capped", False):
                ip.moe_cpu_split_capped = True
                print(f" !! --moe_cpu_split {split_k} exceeds the {self.num_experts} routed experts per layer, "
                      f"capped to {self.num_experts - 1} (use --moe_cpu_offload to move whole layers)")
            split_k = self.num_experts - 1
        if (
            0 < split_k < self.num_experts and
            self.cpu_split_first is None and
            device is not None and torch.device(device).type == "cuda" and
            self.num_local_experts == self.num_experts and self.routing_first is None and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2")
        ):
            if self.load_cpu_split(device, split_k, **kwargs):
                ip.moe_cpu_split_assigned = getattr(ip, "moe_cpu_split_assigned", 0) + 1

    def cpu_post_load(self):
        """After the module's tensors load on a CUDA device: apply the split placement
        permutation to the router-side tensors, and set up dynamic placement state."""
        # Split placement permutation: move the router (and per-expert routing tensors)
        # into the same hot-to-cold expert order the weight lists were rearranged into.
        # gate_tensor is (hidden, E), logits = y @ W: permute columns
        if self.cpu_split_first is not None and self._split_perm is not None:
            perm_t = torch.tensor(self._split_perm, device = self.routing_gate.inner.weight.device)
            ig = self.routing_gate.inner
            ig.weight = ig.weight[:, perm_t].contiguous()
            if getattr(ig, "bias", None) is not None:
                ig.bias = ig.bias[perm_t].contiguous()
            if self.e_score_correction_bias is not None:
                self.e_score_correction_bias = self.e_score_correction_bias[
                    perm_t.to(self.e_score_correction_bias.device)].contiguous()
            if self.per_expert_scale is not None:
                self.per_expert_scale = self.per_expert_scale[
                    perm_t.to(self.per_expert_scale.device)].contiguous()
        # Dynamic placement state: router->physical-slot map (identity start), decayed
        # selection counts, and the shared module registry the sweep walks. The first
        # registered module owns the step counter
        if self.cpu_split_first is not None and getattr(self, "_split_dynamic", False):
            dev = torch.device(self.device)
            first, spares = self.cpu_split_first, self._split_spares
            # Physical ids: [0, first) the GPU slots, first + i the worker's expert i. The spares'
            # experts start in their worker slots, their GPU slots free
            mp = [r if r < first - spares else first + (r - (first - spares)) for r in range(self.num_experts)]
            self._split_map = torch.tensor(mp, dtype = torch.long, device = dev)
            self._split_hist = torch.zeros(self.num_experts, dtype = torch.float, device = dev)
            self._split_selcpu_t = None
            self._swap_tick_count = 0      # steps since the last placement sweep (the sweep zeroes it)
            self._split_step_count = 0     # steps since load: the profile save gates on this one
            if spares:
                self._split_mp = mp
                self._split_free_gpu = list(range(first - spares, first))
                self._split_free_cpu = []
                self._split_pending = []        # in-flight transfers: dict(kind, r, gpu, cpu, ev)
                self._split_busy = set()        # router ids with a transfer in flight
                self._split_promo_n = 0
                self._split_demo_n = 0
            ip2 = self.config.infer_params
            reg = getattr(ip2, "moe_cpu_swap_modules", None)
            if reg is None:
                reg = []
                ip2.moe_cpu_swap_modules = reg
                # A server killed with Ctrl-C never reaches unload: save what the session
                # learned at interpreter exit as well (a no-op after a normal unload, which
                # empties the registry; best effort, the CUDA tensors may already be gone)
                import atexit, weakref
                ref = weakref.ref(ip2)
                def _save_at_exit():
                    ip = ref()
                    if ip is None:
                        return
                    try:
                        save_expert_stats(ip)
                    except Exception:
                        pass
                atexit.register(_save_at_exit)
            reg.append(self)

    def cpu_unload(self):
        if self.cpu_split_first is not None:
            host = getattr(self, "cpu_host", None)
            if host is not None:
                host.unregister()
            (self.gates, self.ups, self.downs, self.modules,
             self.num_local_experts, self.routing_first, self.routing_last) = self._split_saved
            self._split_saved = None
            self.cpu_split_first = None
            if self._split_map is not None:
                reg = getattr(self.config.infer_params, "moe_cpu_swap_modules", None)
                if reg is not None and self in reg:
                    if reg[0] is self:
                        save_expert_stats(self.config.infer_params)
                    reg.remove(self)
                self._split_map = None
                self._split_hist = None
                self._split_pending = None
                self.config.infer_params.moe_cpu_promote_state = None
        if self.cpu_offload:
            host = getattr(self, "cpu_host", None)
            if host is not None:
                host.unregister()
            # Release this layer's slot in the component's offload budget so a reload of the
            # same config can claim it again
            asn = self.config.infer_params.moe_cpu_offload_assigned
            comp = getattr(self, "cpu_component", "text")
            if isinstance(asn, dict) and asn.get(comp, 0) > 0:
                asn[comp] -= 1
        self.cpu_offload = False
        self.cpu_host = None

    def cpu_split_submit(self, y, bsz, selected_experts, routing_weights):
        """Hand the tail experts' share of the routed sum to the worker. Returns
        (cpu_partial, cpu_pending): decode-size batches use the two-phase issue/collect so
        the flag waits land AFTER the caller's GPU expert work (cpu_split_combine collects);
        big prefill batches take the single-phase streamed path, which overlaps internally.
        Expert ids ship in the worker's local range with -1 sentinels for GPU-resident
        picks."""
        if self._split_map is not None:
            self._split_submit_rows = bsz
            self._split_swap_tick()
            if self._split_spares and bsz >= self.cpu_host.stream_min_rows and self._split_pending:
                # The streamed prefill's stager reads worker slots a demotion may still be writing:
                # let the transfers land first (a prefill step is long anyway)
                self.cpu_host.promo_stream(self.device).synchronize()
        if bsz < self.cpu_host.stream_min_rows:
            # Fused fast path: one kernel does the map translate + staging writes straight
            # into the pinned slot, skipping the whole activation payload (and, downstream,
            # the worker compute and the readback) when no CPU expert was selected. Replaces
            # three cudaMemcpyAsync launches, the int32 cast and the pad per layer per step
            if _split_fused:
                call = lambda: self.cpu_host.submit_issue_fused(
                    self.cpu_layer_idx, y, selected_experts, routing_weights,
                    self._split_map,
                    self._split_hist if self._split_map is not None else None,
                    self.cpu_split_first)
                pending = _sprof_wrap("issue", self.cpu_layer_idx, call) \
                    if _split_prof else call()
                if pending is not None:
                    return None, ("fused", pending)
            sel_cpu = self._split_translate(selected_experts)
            return None, self.cpu_host.submit_issue(
                self.cpu_layer_idx, y, sel_cpu, routing_weights)
        sel_cpu = self._split_translate(selected_experts)
        return self.cpu_host.submit_prefill(
            self.cpu_layer_idx, y, sel_cpu, routing_weights), None

    def _split_translate(self, selected_experts):
        """Copy-path selection translate: dynamic placement runs the map kernel (hit counts,
        in-place physical-slot translate -- the bsz-1 routing statics are read by captured
        graphs at baked addresses -- and the worker-side selection with -1 sentinels for
        GPU-resident picks); the static split is a tail offset."""
        if self._split_map is None:
            return (selected_experts - self.cpu_split_first).clamp_min(-1)
        n = selected_experts.numel()
        buf = self._split_selcpu_t
        if buf is None or buf.numel() < n:
            buf = self._split_selcpu_t = torch.empty(
                max(n, 64), dtype = torch.long, device = selected_experts.device)
        sel_cpu = buf[:n].view(selected_experts.shape)
        ext.moe_split_map(
            selected_experts.view(-1), self._split_map, self._split_hist,
            buf[:n], self.cpu_split_first)
        return sel_cpu

    def cpu_offload_issue(self, shape, y, selected_experts, routing_weights, params):
        """Whole-layer offload: the routed sum (of the given shape) comes entirely from the
        worker. Returns (final_hidden_states, cpu_pending) like cpu_split_submit: decode-size
        batches are issued only, and cpu_split_combine collects them, so GPU work the caller
        enqueues in between (the shared expert) runs while the worker computes instead of
        behind the flag wait. Prefill-size batches take the streamed path, which overlaps
        internally. The autosplit measuring forward only observes VRAM allocation, which the
        CPU compute cannot affect, so it skips the (slow, full-chunk) host pass and just
        allocates the output."""
        if params.get("autosplit_measure"):
            return torch.zeros_like(y, dtype = torch.float).reshape(shape), None
        if y.shape[0] < self.cpu_host.stream_min_rows:
            return None, self.cpu_host.submit_issue(
                self.cpu_layer_idx, y, selected_experts, routing_weights)
        return self.cpu_host.submit_prefill(
            self.cpu_layer_idx, y, selected_experts, routing_weights
        ).reshape(shape), None

    def cpu_offload_forward(self, shape, y, selected_experts, routing_weights, params):
        """Single-phase form of cpu_offload_issue (issue and collect back to back)"""
        final_hidden_states, cpu_pending = self.cpu_offload_issue(shape, y, selected_experts, routing_weights, params)
        return self.cpu_split_combine(final_hidden_states, None, cpu_pending, shape)

    def cpu_split_combine(self, final_hidden_states, cpu_partial, cpu_pending, shape):
        """Fold the CPU tail partial into the routed sum (stream-ordered: the collect
        enqueues a flag wait ahead of the readback, so the add consumes the worker's output
        exactly when it is ready). Fused handles add in place straight from the pinned slot
        (or skip the readback entirely for a job with no CPU-resident picks)."""
        if cpu_pending is not None:
            if isinstance(cpu_pending, tuple) and cpu_pending[0] == "fused":
                f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
                if _split_prof:
                    _sprof_wrap("wait", self.cpu_layer_idx,
                                lambda: self.cpu_host.submit_collect_fused(cpu_pending[1], f2))
                else:
                    self.cpu_host.submit_collect_fused(cpu_pending[1], f2)
                return final_hidden_states
            cpu_partial = self.cpu_host.submit_collect(cpu_pending)
        if cpu_partial is not None:
            # Whole-layer offload: the worker's output is the routed sum
            final_hidden_states = cpu_partial.view(shape) if final_hidden_states is None \
                else final_hidden_states + cpu_partial.view(shape)
        return final_hidden_states

    def cpu_split_permuted(self) -> bool:
        """True when the split's expert lists were permuted hot-to-cold (a saved profile or an
        EXL3_MOE_CPU_SPLIT_STATS file). cpu_post_load then permutes the router tensors right
        after they load, so those must be filled eagerly even inside a deferred-load bracket;
        the rest of the module (every GPU-resident expert) keeps the batched deferred fill"""
        return self.cpu_split_first is not None and getattr(self, "_split_perm", None) is not None

    def load_cpu_offload(self, device: torch.Device, **kwargs) -> bool:
        """
        Experimental CPU expert offload: register the layer with the persistent CPU MoE worker
        (which loads the expert weights itself, concurrently with GPU loading) and load
        everything else (router, norms, shared experts) on the GPU as usual. Eligibility here
        uses header metadata only; the parent never fetches expert data. Returns False without
        side effects when the layer is ineligible (non-mul1 codebook, unsupported K, or mixed per-expert
        biases), in which case the caller falls back to the normal path.
        """
        stc = self.config.stc
        cpu = torch.device("cpu")
        experts = self.gates + self.ups + self.downs

        # Eligibility probe on the first expert of each projection before fetching bulk data
        probe = ([self.gates[0]] if self.gated else []) + [self.ups[0], self.downs[0]]
        for l in probe:
            if stc.get_tensor(l.key + ".mul1", cpu, optional = True) is None:
                print(f" !! {self.key}: experts are not mul1, CPU offload skipped")
                return False

        def hdr_shape(l):
            return stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
        for l in probe:
            if trellis_rate(hdr_shape(l)[-1]) is None:
                print(f" !! {self.key}: unsupported bitrate, CPU offload skipped")
                return False
        def bias_keys(ls):
            has = [(l.key + ".bias") in stc.tensor_file_map for l in ls]
            if any(has) and not all(has):
                print(f" !! {self.key}: mixed expert biases, CPU offload skipped")
                return None
            return all(has)
        checks = [bias_keys(ls) for ls in ([self.gates] if self.gated else []) + [self.ups, self.downs]]
        if any(c is None for c in checks):
            return False

        self.device = torch.device(device)
        expert_set = set(experts)
        for module in self.modules:
            if module not in expert_set:
                module.load(device, **kwargs)
        if self.e_score_correction_bias_key:
            for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
                # Loaded in its checkpoint precision: GLM-5.2's bias sits near 34.0 where
                # fp16 resolution (0.03) would swamp the inter-expert sigmoid score gaps. The
                # kernels get a centered fp16 copy (see RoutingCFG.e_score_bias_h)
                # no_defer: the fp32 working copy below is derived at load time, which a
                # deferred (unfilled) tensor would corrupt
                esb = self.config.stc.get_tensor(
                    f"{self.key}.{k}", self.device, optional = True, allow_bf16 = True,
                    no_defer = True)
                if esb is not None:
                    self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                    break
        if self.per_expert_scale_key:
            self.per_expert_scale = self.config.stc.get_tensor(
                f"{self.key}.{self.per_expert_scale_key}", self.device, optional = True, allow_bf16 = True)
        if self.e_score_bias_vl_key:
            esb_vl = self.config.stc.get_tensor(
                f"{self.key}.{self.e_score_bias_vl_key}", self.device, optional = True, allow_bf16 = True,
                no_defer = True)
            self.e_score_bias_vl = esb_vl.float() if esb_vl is not None else None
        if self.tid2eid_key:
            self.tid2eid = self.config.stc.get_tensor(
                f"{self.key}.{self.tid2eid_key}", self.device, no_defer = True)
        self.load_routing(**kwargs)

        from ..model.moe_cpu_host import MoeCpuHost
        # One worker per component: an MTP head shares the config but loads after the main
        # model's worker has started, so it gets its own child (which loads only its own
        # layers from the same checkpoint)
        comp = getattr(self.config.infer_params, "moe_cpu_component", "text")
        host = MoeCpuHost.for_config(self.config)
        self.cpu_host = host
        self.cpu_component = comp
        def dims_of(l):
            s = stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
            return (s[0] * 16, s[1] * 16, trellis_rate(s[2]))
        gd = dims_of(self.gates[0]) if self.gated else None
        ud = dims_of(self.ups[0])
        dd = dims_of(self.downs[0])
        hi, ho = ud[0], dd[1]

        # Small per-expert tensors resident on the GPU for the streamed-prefill dequant path
        # (lists, not stacks: the fetches may be deferred and fill in place)
        def fetch_aux(ls, suffix, optional = False):
            out = [stc.get_tensor(l.key + suffix, self.device, optional = optional,
                                  float2half = True) for l in ls]
            return out if not optional or out[0] is not None else None
        aux = dict(
            suh_u = fetch_aux(self.ups, ".suh"), svh_u = fetch_aux(self.ups, ".svh"),
            suh_d = fetch_aux(self.downs, ".suh"), svh_d = fetch_aux(self.downs, ".svh"),
            bias_u = fetch_aux(self.ups, ".bias", True),
            bias_d = fetch_aux(self.downs, ".bias", True),
        )
        if self.gated:
            aux["suh_g"] = fetch_aux(self.gates, ".suh")
            aux["svh_g"] = fetch_aux(self.gates, ".svh")
            aux["bias_g"] = fetch_aux(self.gates, ".bias", True)

        self.cpu_layer_idx = host.register_layer(
            self.key,
            [l.key for l in self.gates] if self.gated else [],
            [l.key for l in self.ups],
            [l.key for l in self.downs],
            {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}[self.activation_fn],
            float(self.act_limit or 0.0),
            hi, ho, self.num_experts_per_tok,
            proj_dims = dict(g = gd, u = ud, d = dd),
            aux = aux,
            interm_fp32 = self.interm_dtype == torch.float,
        )
        self.cpu_offload = True
        print(f" -- CPU-offloaded experts (worker): {self.key}")
        return True


    def load_cpu_split(self, device: torch.Device, split_k: int, **kwargs) -> bool:
        """
        Experimental per-layer expert split: register the TAIL split_k experts with the CPU MoE
        worker and shrink this module to a GPU-resident [0, E - split_k) slice (the existing
        expert-interval machinery masks the tail out of every GPU path, exactly like a TP
        expert shard). forward() then submits the tail partial to the worker right after
        routing and adds it to the GPU partial, so the CPU GEMMs overlap the layer's own GPU
        expert compute instead of serializing a whole layer. Registration only. The caller
        proceeds with the normal load, which now loads just the GPU slice.
        """
        stc = self.config.stc
        cpu = torch.device("cpu")
        first = self.num_experts - split_k

        # Same eligibility probes as the whole-layer path, on the tail experts
        probe = ([self.gates[first]] if self.gated else []) + [self.ups[first], self.downs[first]]
        for l in probe:
            if stc.get_tensor(l.key + ".mul1", cpu, optional = True) is None:
                print(f" !! {self.key}: experts are not mul1, CPU split skipped")
                return False
        def hdr_shape(l):
            return stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
        for l in probe:
            if trellis_rate(hdr_shape(l)[-1]) is None:
                print(f" !! {self.key}: unsupported bitrate, CPU split skipped")
                return False
        def bias_keys(ls):
            has = [(l.key + ".bias") in stc.tensor_file_map for l in ls[first:]]
            if any(has) and not all(has):
                print(f" !! {self.key}: mixed expert biases, CPU split skipped")
                return None
            return all(has)
        checks = [bias_keys(ls) for ls in ([self.gates] if self.gated else []) + [self.ups, self.downs]]
        if any(c is None for c in checks):
            return False

        # Optional frequency-guided placement (EXL3_MOE_CPU_SPLIT_STATS = json of per-layer
        # selection counts, e.g. from a probe run): permute the expert order hot-to-cold so
        # the offloaded tail holds the least-selected experts. The router weight rows (and
        # per-expert routing tensors) are permuted to match after they load, so selection
        # indices, GPU pointer tables and the worker's key list all live in the same permuted
        # space. Measured on lfm2.5: the coldest 12/32 experts draw ~8% of selections vs
        # 37.5% uniform, cutting the CPU share per layer ~4.5x
        self._split_perm = None
        self._split_seeded = False
        self._split_saved_lists = (self.gates, self.ups, self.downs)
        # Dynamic placement (default; EXL3_MOE_CPU_SWAP=0 disables): both the GPU's E - k
        # slots and the worker's k slots hold a CHANGING set of experts; a per-layer
        # router->slot map applied right after routing decides placement. A swap re-reads
        # both experts from the checkpoint: the promoted one into the GPU slot tensors in
        # place (all baked pointers stay valid), the demoted one into the worker's arena
        # via the install message (the child re-reads it from its own checkpoint handle)
        self._split_dynamic = os.environ.get("EXL3_MOE_CPU_SWAP", "1") != "0" \
            and not self.tid2eid_key
        # Promote-on-miss (EXL3_MOE_CPU_PROMOTE = spare slots per layer, default 4, 0 off): with
        # the pinned arena the parent moves experts between the GPU slots and the worker's arena
        # by DMA, so a routed expert the CPU had to compute can take a spare GPU slot for the
        # next tokens and the least-used resident goes the other way. The worker is registered
        # from `spares` experts below the GPU slice, whose GPU slots start out as the free
        # spares (their experts live in the worker's extra slots). At least one GPU slot stays a
        # regular resident: a promotion must beat the least-used resident's count, so a layer
        # whose slots all start empty would have nothing to beat and never promote
        spares = 0
        if self._split_dynamic and getattr(self.config.infer_params, "moe_cpu_pinned_arena", False):
            spares = max(0, int(os.environ.get("EXL3_MOE_CPU_PROMOTE", 4)))
        spares = min(spares, max(0, first - 1))
        self._split_spares = spares
        wfirst = first - spares
        # Hot/cold placement: an explicit stats file (EXL3_MOE_CPU_SPLIT_STATS) or, failing
        # that, the profile a previous run of this model saved (save_expert_stats). The expert
        # lists are permuted hottest first, so the GPU slice holds the frequent experts from
        # the first token and the worker the rare ones; the dynamic map then starts from there
        counts = None
        if not self.tid2eid_key:
            stats_path = os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
            if stats_path:
                import json
                counts = json.load(open(stats_path)).get(self.key)
                if counts is None:
                    print(f" !! {self.key}: no routing stats for layer, tail placement unpermuted")
            else:
                counts = _expert_stats_load(self.config, self.num_experts).get(self.key)
        if counts is not None and len(counts) == self.num_experts:
            perm = sorted(range(self.num_experts), key = lambda e: -counts[e])
            self._split_perm = perm
            if self.gated:
                self.gates = [self.gates[e] for e in perm]
            self.ups = [self.ups[e] for e in perm]
            self.downs = [self.downs[e] for e in perm]
            self._split_seeded = True

        self.device = torch.device(device)

        from ..model.moe_cpu_host import MoeCpuHost
        comp = getattr(self.config.infer_params, "moe_cpu_component", "text")
        host = MoeCpuHost.for_config(self.config)
        self.cpu_host = host
        self.cpu_component = comp

        def dims_of(l):
            s = stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
            return (s[0] * 16, s[1] * 16, trellis_rate(s[2]))
        gd = dims_of(self.gates[first]) if self.gated else None
        ud = dims_of(self.ups[first])
        dd = dims_of(self.downs[first])
        hi, ho = ud[0], dd[1]

        def fetch_aux(ls, suffix, optional = False):
            out = [stc.get_tensor(l.key + suffix, self.device, optional = optional,
                                  float2half = True) for l in ls[wfirst:]]
            return out if not optional or out[0] is not None else None
        aux = dict(
            suh_u = fetch_aux(self.ups, ".suh"), svh_u = fetch_aux(self.ups, ".svh"),
            suh_d = fetch_aux(self.downs, ".suh"), svh_d = fetch_aux(self.downs, ".svh"),
            bias_u = fetch_aux(self.ups, ".bias", True),
            bias_d = fetch_aux(self.downs, ".bias", True),
        )
        if self.gated:
            aux["suh_g"] = fetch_aux(self.gates, ".suh")
            aux["svh_g"] = fetch_aux(self.gates, ".svh")
            aux["bias_g"] = fetch_aux(self.gates, ".bias", True)

        self.cpu_layer_idx = host.register_layer(
            self.key,
            [l.key for l in self.gates[wfirst:]] if self.gated else [],
            [l.key for l in self.ups[wfirst:]],
            [l.key for l in self.downs[wfirst:]],
            {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}[self.activation_fn],
            float(self.act_limit or 0.0),
            hi, ho, self.num_experts_per_tok,
            proj_dims = dict(g = gd, u = ud, d = dd),
            aux = aux,
            interm_fp32 = self.interm_dtype == torch.float,
        )

        self._split_wfirst = wfirst
        # Shrink to the GPU slice. The tail Linears leave the module tree entirely (never
        # loaded); unload() restores them so a reload can redo the split cleanly
        tail = set((self.gates[first:] if self.gated else []) + self.ups[first:] + self.downs[first:])
        self._split_saved = (*self._split_saved_lists, self.modules,
                            self.num_local_experts, self.routing_first, self.routing_last)
        del self._split_saved_lists
        self.gates = self.gates[:first] if self.gated else self.gates
        self.ups = self.ups[:first]
        self.downs = self.downs[:first]
        self.modules = [m for m in self.modules if m not in tail]
        self.num_local_experts = first
        self.routing_first = 0
        self.routing_last = first
        self.cpu_split_first = first
        mode = ("promote-on-miss" if spares else "dynamic") if self._split_dynamic else "static"
        print(f" -- CPU split experts (worker, {mode}): {self.key} "
              f"[{first}..{self.num_experts}) of {self.num_experts}"
              + (f", {spares} spare slots" if spares else "")
              + (", placement from saved profile" if getattr(self, "_split_seeded", False) else ""))
        return True


    def _split_swap_tick(self):
        """Dynamic placement sweep trigger: the first registered module counts decode
        steps; every EXL3_MOE_CPU_SWAP_INTERVAL steps, quiesce (one full synchronize,
        every issued worker job's collect is already enqueued, so after the sync the ring is
        drained and the child idle) and let each split layer promote its hottest CPU experts
        into the slots of its coldest GPU experts."""
        ip = self.config.infer_params
        reg = ip.moe_cpu_swap_modules
        if reg[0] is not self:
            return
        self._swap_tick_count += 1
        self._split_step_count += 1
        if self._split_step_count % _STATS_SAVE_INTERVAL == 0:
            save_expert_stats(ip)
        if self._split_spares:
            # Promote-on-miss replaces the sweep: one pass over every split module per step
            _split_promote_step(ip, reg, self._split_submit_rows)
            return
        interval = int(os.environ.get("EXL3_MOE_CPU_SWAP_INTERVAL", 128))
        if self._swap_tick_count < interval:
            return
        # A placement change is a small numeric step for every touched expert (the same
        # weights compute slightly differently on the two devices), so sweeping mid-stream
        # visibly derails a generation in progress. Mark the sweep pending and let the
        # generator run it from its queue-drained hook; fall back to sweeping inline only
        # when badly overdue (raw model.forward drivers with no generator, e.g. perf.py)
        ip.moe_cpu_swap_pending = True
        if self._swap_tick_count < 4 * interval:
            return
        run_pending_swap_sweeps(ip)

    def _split_sweep_layer_reset(self):
        self._swap_tick_count = 0

    def _split_sweep_layer(self, budget):
        """One placement sweep for this layer: repeatedly swap the hottest CPU-resident
        expert with the coldest GPU-resident one while the hit-count ratio clears the
        hysteresis threshold. The promoted expert's tensors are re-read from the checkpoint
        into the demoted expert's slot IN PLACE, so every baked pointer (multilinear tables,
        captured graphs, fused stacks) stays valid; the demoted expert becomes CPU-resident
        by map update alone, since the worker holds every expert. Skips shape-mismatched
        pairs (per-expert quant width differences). Decays the counts afterwards so the
        stats track recent routing."""
        first = self.cpu_split_first
        hyst = float(os.environ.get("EXL3_MOE_CPU_SWAP_HYST", 2.0))
        mp = self._split_map.cpu()
        hist = self._split_hist.cpu()
        # Absolute mass floor on top of the ratio test: with short accumulation windows the
        # per-expert counts are only a few hits, and a tail expert with 3 vs a head expert
        # with 1 clears any pure ratio -- the sweep then churns on noise forever (each churn
        # a checkpoint read). Requiring several multiples of the uniform expectation makes
        # the settled state a fixed point: post-settling tail experts sit far below the
        # floor, so sweeps become no-ops
        floor = float(os.environ.get("EXL3_MOE_CPU_SWAP_FLOOR", 8.0)) \
            * float(hist.sum()) / self.num_experts
        head = [(float(hist[r]), r) for r in range(self.num_experts) if int(mp[r]) < first]
        tail = [(float(hist[r]), r) for r in range(self.num_experts) if int(mp[r]) >= first]
        head.sort()
        tail.sort(reverse = True)
        nswaps = 0
        for (c_cold, r_cold), (c_hot, r_hot) in zip(head, tail):
            if nswaps >= budget or c_hot < max(hyst * max(c_cold, 1.0), floor):
                break
            if self._split_swap_experts(r_cold, r_hot, mp):
                nswaps += 1
        if nswaps:
            if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
                assert mp.sort().values.equal(torch.arange(self.num_experts)), \
                    f"{self.key}: placement map is not a permutation after sweep"
            self._split_map.copy_(mp.to(self._split_map.device))
        self._split_hist.mul_(0.5)
        return nswaps

    def _split_swap_experts(self, r_cold, r_hot, mp):
        """Promote router expert r_hot into the GPU slot of r_cold (checkpoint read +
        in-place tensor copy), demote r_cold by map update. mp is the host-side map, updated
        on success."""
        stc = self.config.stc
        slot = int(mp[r_cold])
        full_g, full_u, full_d = self._split_saved_lists_ref()
        proj = ([(full_g, self.gates)] if self.gated else []) \
            + [(full_u, self.ups), (full_d, self.downs)]
        pairs = []
        for full, cur in proj:
            src_key = full[r_hot].key
            dst = cur[slot].inner
            shp = stc.list_tensors(src_key)[src_key + ".trellis"]["shape"]
            if list(shp) != list(dst.trellis.shape):
                return False
            pairs.append((src_key, dst))
        # The demoted expert also must match the worker slot's tenant (= the promoted
        # expert's shapes, since they exchange homes) for the in-place arena copy
        for full, _ in proj:
            k_cold = full[r_cold].key
            k_hot = full[r_hot].key
            sc = stc.list_tensors(k_cold)[k_cold + ".trellis"]["shape"]
            sh = stc.list_tensors(k_hot)[k_hot + ".trellis"]["shape"]
            if list(sc) != list(sh):
                return False

        for src_key, dst in pairs:
            dst.trellis.copy_(stc.get_tensor(src_key + ".trellis", dst.trellis.device))
            dst.suh.copy_(stc.get_tensor(src_key + ".suh", dst.suh.device, float2half = True))
            dst.svh.copy_(stc.get_tensor(src_key + ".svh", dst.svh.device, float2half = True))
            if getattr(dst, "bias", None) is not None:
                b = stc.get_tensor(src_key + ".bias", dst.bias.device, optional = True,
                                   float2half = True)
                if b is not None:
                    dst.bias.copy_(b)

        # Demote r_cold into the worker slot r_hot vacated: the child re-reads the weights
        # from its own checkpoint handle into the arena in place (we are quiesced), and the
        # streamed-prefill aux copies for that slot update to match
        cpu_local = int(mp[r_hot]) - self.cpu_split_first
        keys = ([full_g[r_cold].key] if self.gated else []) \
            + [full_u[r_cold].key, full_d[r_cold].key]
        self.cpu_host.install_expert(self.cpu_layer_idx, cpu_local, keys)
        aux = self.cpu_host.aux.get(self.cpu_layer_idx)
        if aux:
            names = ([("suh_g", full_g), ("svh_g", full_g), ("bias_g", full_g)] if self.gated else []) \
                + [("suh_u", full_u), ("svh_u", full_u), ("bias_u", full_u),
                   ("suh_d", full_d), ("svh_d", full_d), ("bias_d", full_d)]
            for name, full in names:
                lst = aux.get(name)
                if lst is None or lst[cpu_local] is None:
                    continue
                suffix = "." + name.split("_")[0]
                t = stc.get_tensor(full[r_cold].key + suffix, lst[cpu_local].device,
                                   optional = name.startswith("bias"), float2half = True)
                if t is not None:
                    lst[cpu_local].copy_(t)

        if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
            # Read back the promoted slot and compare against the checkpoint (catches any
            # copy/stream/layout unfaithfulness in the real sweep context)
            torch.cuda.synchronize()
            for src_key, dst in pairs:
                for tname, live in (("trellis", dst.trellis), ("suh", dst.suh), ("svh", dst.svh)):
                    fresh = stc.get_tensor(src_key + "." + tname, live.device,
                                           float2half = tname != "trellis")
                    assert torch.equal(fresh, live), \
                        f"swap verify failed: {src_key}.{tname}"
        mp[r_cold], mp[r_hot] = int(mp[r_hot]), slot
        return True

    def _split_saved_lists_ref(self):
        """The full expert Linear lists (GPU slice and worker tail together) in ROUTER order. The
        saved lists are in checkpoint order; with a hot/cold placement permutation in effect, router
        id r is checkpoint expert _split_perm[r] (the router tensors were permuted the same way at
        load), so the sweep's checkpoint reads, the demotion's install keys and the transfer verify
        must all go through the permutation."""
        g, u, d = self._split_saved[0], self._split_saved[1], self._split_saved[2]
        perm = self._split_perm
        if perm is not None:
            g = [g[e] for e in perm] if self.gated else g
            u = [u[e] for e in perm]
            d = [d[e] for e in perm]
        return g, u, d
