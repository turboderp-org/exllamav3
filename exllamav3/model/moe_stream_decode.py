"""
Streamed single-row decode for whole-layer CPU MoE offload (EXL3_MOE_STREAM_DECODE; needs the
pinned arena, EXL3_MOE_PINNED_ARENA).

With whole-layer offload (-mcl), a decode step's routed experts are all computed by the CPU
worker while the GPU waits on its done flag, and on a host with few cores that wait is most of
the decode budget (Qwen3.8-Flash-Next, 512 experts per layer, top-10, 48 offloaded layers: ~76
ms per token on a six-core AVX2 desktop, 11 tok/s). With the pinned arena the parent already
has every expert's contiguous [gate | up | down] trellis block mapped into the device address
space, so a single-row step can instead pull the selected experts' blocks over PCIe into a VRAM
staging buffer with one gather kernel (moe_stream_gather: zero-copy reads, top-k x expert_bytes
per layer) and run the resident fused MoE kernel (exl3_moe) over them through pointer tables:
the streamed-prefill tier's scheme without its per-batch host loop. Optionally the k lowest-
ranked picks of every layer go to the worker instead (EXL3_MOE_STREAM_DECODE_CPU=k), which
computes them while the GPU streams the others, so the link and the cores work at the same
time. Same host, PCIe 4.0 x16 (24 GB/s pinned->device): 11.2 -> 14.5 tok/s with all ten picks
streamed, 19.0 tok/s with four of them on the worker; lfm2.5-8b-a1b (32 experts, top-4):
30.7 -> 55.5 tok/s with one on the worker. Nothing on the path synchronizes the host with the
stream.

Rows >= 2 (prefill, batched and speculative decode steps) keep the existing paths, as do layers
the fused kernel cannot take (per-expert biases, padded dims, swiglu_oai), per layer. On the
AVX-512 CPU tiers the arena is band-swizzled (EXL3_MOE_CPU_SWIZZLE); the gathered blocks are
then restored to the native tile order in VRAM (moe_unswizzle_trellis), as the prefill tier
does. Any error on the path disables it for the rest of the run (with a traceback), and the
worker serves everything.

EXL3_MOE_STREAM_DECODE=check runs the worker AND the streamed path for every decode layer,
reports the difference and returns the worker's result. EXL3_MOE_STREAM_DECODE_PROF=1 reports
stream timings (gather, fused kernel, worker wait) every 16 tokens.
"""
from __future__ import annotations
import traceback
import torch
from ..ext import exllamav3_ext as ext

# Rows of the per-expert aux pointer table, in the fused kernel's table order (a gateless layer
# passes the up projection's vectors as the gate placeholders; the kernel skips the gate GEMM)
_AUX_KEYS_GATED = ("suh_g", "svh_g", "suh_u", "svh_u", "suh_d", "svh_d")
_AUX_KEYS_GATELESS = ("suh_u", "svh_u", "suh_u", "svh_u", "suh_d", "svh_d")


def staging_bytes(specs):
    """Size of the decode staging buffer for the given layer specs: one slot per selected expert
    of the largest (top-k x expert block) among them, whatever the worker's share"""
    return max((s["topk"] * s["expert_bytes"] for s in specs if s.get("expert_bytes")), default = 0)


def ensure_staging(bufs, specs, device):
    """Allocate (or grow) the staging buffers in a device's persistent buffer dict
    (MoeCpuHost._device_buffers): the raw gather target and, for swizzled arenas, the
    native-order copy the fused kernel reads. Called from the host's measured-load buffer setup,
    so the VRAM is accounted for before the split is planned, and again at first use"""
    need = staging_bytes(specs) // 2
    st = bufs.get("sd_stage")
    if need and (st is None or st.numel() < need):
        bufs["sd_stage"] = torch.empty(need, dtype = torch.int16, device = device)
        bufs["sd_native"] = torch.empty(need, dtype = torch.int16, device = device) if bufs["swz"] else None
    return bufs.get("sd_stage"), bufs.get("sd_native")


class MoeStreamDecode:
    """
    Per-host state of the streamed decode path, built lazily by MoeCpuHost at its first
    single-row step: per device, CUDA aliases of the arena chunks and the per-layer block and
    aux pointer tables; per (expert layout, worker share), the fused kernel's static
    descriptors. forward() returns None wherever the worker should serve the call.
    """

    def __init__(self, host):
        self.host = host
        self.mode = host.stream_decode_mode          # "1" or "check"
        self.cpu_k = host.stream_decode_cpu          # picks per layer handed to the worker
        self.prof = host.stream_decode_prof
        self.disabled = False
        self.calls = 0                               # layer calls served (activation check for A/B runs)
        self.dev = {}
        self.stats = dict(n = 0, max_abs = 0.0, max_rel = 0.0, worst = -1, nonfinite = 0)
        self.prof_ev = []
        if self.mode == "check":
            import atexit
            atexit.register(self.summary)

    def _device(self, device):
        key = device.index if device.index is not None else torch.cuda.current_device()
        D = self.dev.get(key)
        if D is None or len(D["aliases"]) != len(self.host.arena_views):
            # CUDA aliases of the pinned arena chunks (kept here: an alias does not own its
            # mapping) and their base addresses, the table the gather kernel reads through
            aliases = [ext.pinned_cuda_view(v, key) for v in self.host.arena_views]
            if not aliases:
                raise RuntimeError("no pinned arena chunks are mapped (EXL3_MOE_PINNED_ARENA)")
            D = dict(
                key = key, aliases = aliases, bufs = self.host._device_buffers(device),
                chunk_base = torch.tensor([a.data_ptr() for a in aliases], dtype = torch.long, device = device),
                layers = {}, desc = {},
            )
            D["swz"] = D["bufs"]["swz"]
            # Sized for every registered layer (the set is fixed once the worker has started);
            # the measured load allocated it already, this only grows a buffer it missed
            D["stage"], D["native"] = ensure_staging(D["bufs"], self.host.specs, device)
            self.dev[key] = D
        return D

    def _layer(self, D, li, h):
        aux = self.host.aux.get(li)
        L = D["layers"].get(li, False)
        if L is False or (L is not None and L["aux_id"] != id(aux)):
            L = D["layers"][li] = self._build_layer(D, li, h, aux)
        return L

    def _build_layer(self, D, li, h, aux):
        """Per-layer tables, or None for a layer the path does not take (no arena blocks or aux,
        or not fused-kernel eligible): the worker serves it as before"""
        host = self.host
        spec = host.specs[li]
        blocks = host.layer_blocks[li] if li < len(host.layer_blocks) else None
        E = spec["num_experts"]
        exp_b = spec.get("expert_bytes")
        if (not exp_b or exp_b % 16 or aux is None or not blocks or len(blocks) != E
                or not host._stream_fused_t(spec, aux, h)):
            return None
        pd = spec["proj_dims"]
        gated = pd.get("g") is not None
        keys = _AUX_KEYS_GATED if gated else _AUX_KEYS_GATELESS
        if any(aux.get(n) is None or len(aux[n]) != E for n in keys):
            return None
        dev = D["chunk_base"].device
        gb, ub, _ = spec["proj_bytes"]
        Ku, Kd = pd["u"][2], pd["d"][2]
        return dict(
            spec = spec, exp_b = exp_b, aux_id = id(aux),
            blk_chunk = torch.tensor([int(b[0]) for b in blocks], dtype = torch.int32, device = dev),
            blk_off = torch.tensor([int(b[1]) for b in blocks], dtype = torch.long, device = dev),
            aux_ptrs = torch.tensor([[t.data_ptr() for t in aux[n]] for n in keys],
                                    dtype = torch.long, device = dev),
            Ks = (pd["g"][2] if gated else Ku, Ku, Kd),
            # (byte offset, tiles_k, tiles_n, K) per projection, for the un-swizzle of swizzled arenas
            projs = [(off, d[0] // 16, d[1] // 16, d[2])
                     for off, d in ((0, pd.get("g")), (gb, pd["u"]), (gb + ub, pd["d"])) if d],
            fb = host._stream_fused_bufs(D["bufs"], spec, dev),
        )

    def _desc(self, D, L):
        spec = L["spec"]
        topk = spec["topk"]
        k = min(max(int(self.cpu_k), 0), topk - 1)
        g = topk - k
        base_t = D["native"] if D["swz"] else D["stage"]
        key = (spec["proj_bytes"], k, base_t.data_ptr())
        T = D["desc"].get(key)
        if T is None:
            exp_b = L["exp_b"]
            gb, ub, _ = spec["proj_bytes"]
            base, dev = base_t.data_ptr(), base_t.device
            T = D["desc"][key] = dict(
                g = g, k = k,
                # Trellis pointers into the staging slots, gate | up | down per slot (gateless:
                # gb = 0, the gate row aliases the up row and is never dereferenced)
                s3 = torch.tensor([[base + i * exp_b + o for i in range(g)] for o in (0, gb, gb + ub)],
                                  dtype = torch.long, device = dev),
                d6 = torch.empty((6, g), dtype = torch.long, device = dev),   # gathered suh/svh pointers
                ec = torch.tensor([1] * g + [0], dtype = torch.long, device = dev),
                tok = torch.zeros(g, dtype = torch.long, device = dev),
                # Worker share: top-k positions [g, topk) (the lowest-ranked picks of routers that
                # sort their selection); the streamed positions are masked out of its job
                mask_gpu = torch.tensor([True] * g + [False] * k, device = dev) if k else None,
            )
            if not D.get("announced"):
                D["announced"] = True
                print(f" -- stream-decode: cuda:{D['key']}, {len(D['aliases'])} arena chunks mapped, "
                      f"{g} experts per layer streamed" + (f", {k} computed by the worker" if k else "")
                      + f", staging {base_t.numel() * 2 >> 20} MiB"
                      + (" (swizzled arena, native order restored in VRAM)" if D["swz"] else ""), flush = True)
        return T

    def forward(self, layer_idx, y, selected_experts, routing_weights):
        """The routed sum of one single-row layer call as a float (1, h) tensor on y's device,
        or None where the worker path serves the call (rows != 1, ineligible layer, disabled)"""
        if self.disabled or y.shape[0] != 1:
            return None
        try:
            D = self._device(y.device)
            L = self._layer(D, layer_idx, y.shape[1])
        except Exception:
            self._disable(f"setup at layer {layer_idx}")
            return None
        if L is None:
            return None
        if self.mode == "check":
            return self._check(D, L, layer_idx, y, selected_experts, routing_weights)
        try:
            return self._run(D, L, layer_idx, y, selected_experts, routing_weights)
        except Exception:
            self._disable(f"layer {layer_idx}")
            return None

    def _run(self, D, L, li, y, sel, w):
        host, spec = self.host, L["spec"]
        T = self._desc(D, L)
        g, k = T["g"], T["k"]
        sel_flat = sel.reshape(-1)
        if sel_flat.dtype != torch.long:
            sel_flat = sel_flat.long()
        sel_flat = sel_flat.contiguous()
        wts = w.reshape(-1).half().contiguous()
        pending = None
        with torch.cuda.device(y.device):
            out = torch.zeros((1, y.shape[1]), dtype = torch.float, device = y.device)
            ev = [torch.cuda.Event(enable_timing = True) for _ in range(4)] if self.prof else None
            if ev:
                ev[0].record()
            try:
                if k:
                    # Worker share first, so the cores compute while the GPU streams: the streamed
                    # picks are masked to -1, which the worker skips
                    pending = host.submit_issue(li, y, sel_flat.masked_fill(T["mask_gpu"], -1).view(1, -1), w)
                sel_gpu = sel_flat[:g]
                # 1. The streamed picks' blocks, arena (pinned RAM) -> staging (VRAM), and their
                #    suh/svh pointers into the per-slot table, one launch
                ext.moe_stream_gather(D["stage"], D["chunk_base"], L["blk_chunk"], L["blk_off"],
                                      sel_gpu, L["exp_b"], L["aux_ptrs"], T["d6"])
                if D["swz"]:
                    for off, tk, tn, K in L["projs"]:
                        ext.moe_unswizzle_trellis(D["stage"], D["native"], g, L["exp_b"], off, tk, tn, K, K != 8)
                if ev:
                    ev[1].record()
                # 2. The fused MoE kernel over the staged experts, one row each
                fb, s3, d6 = L["fb"], T["s3"], T["d6"]
                Kg, Ku, Kd = L["Ks"]
                ext.exl3_moe(
                    y, out, T["ec"], T["tok"], wts[:g],
                    fb[0], fb[1], fb[2], fb[3],
                    spec["activation"], Kg, Ku, Kd,
                    s3[0], d6[0], d6[1], s3[1], d6[2], d6[3], s3[2], d6[4], d6[5],
                    False, True, False, True, False, True,
                    float(spec["act_limit"] or 0.0), g, None, None, 1, 1, 16,
                )
                if ev:
                    ev[2].record()
            except Exception:
                if pending is not None:
                    # A published job must be collected, or its slot stalls the ring later
                    try:
                        host.submit_collect(pending)
                    except Exception:
                        pass
                raise
            # 3. Fold the worker's partial in (stream-ordered: flag wait, then the readback)
            if pending is not None:
                out.add_(host.submit_collect(pending))
            if ev:
                ev[3].record()
                self._prof_add(ev)
        self.calls += 1
        return out

    def _check(self, D, L, li, y, sel, w):
        ref = self.host.submit(li, y, sel, w)
        try:
            mine = self._run(D, L, li, y, sel, w)
        except Exception:
            self._disable(f"check at layer {li}")
            return ref
        c = self.stats
        max_abs = float((mine - ref).abs().max())
        ref_max = float(ref.abs().max())
        rel = max_abs / max(ref_max, 1e-20)
        finite = bool(torch.isfinite(mine).all())
        c["n"] += 1
        c["nonfinite"] += int(not finite)
        c["max_rel"] = max(c["max_rel"], rel)
        if max_abs > c["max_abs"]:
            c["max_abs"], c["worst"] = max_abs, li
        nl = len(self.host.specs)
        if c["n"] <= nl or not finite:
            print(f" -- stream-decode check L{li}: max|streamed - worker| {max_abs:.3e}, "
                  f"max|worker| {ref_max:.3e}, rel {rel:.2e}{'' if finite else '  NON-FINITE'}", flush = True)
        elif c["n"] % (nl * 16) == 0:
            self.summary()
        return ref

    def summary(self):
        c = self.stats
        print(f" -- stream-decode check: {c['n']} layer calls, max abs {c['max_abs']:.3e} (L{c['worst']}), "
              f"max rel {c['max_rel']:.2e}, non-finite {c['nonfinite']}", flush = True)

    def _prof_add(self, ev):
        self.prof_ev.append(ev)
        nl = len(self.host.specs)
        if len(self.prof_ev) < nl * 16:
            return
        done = [(a.elapsed_time(b), b.elapsed_time(c), c.elapsed_time(d))
                for a, b, c, d in self.prof_ev if d.query()]
        self.prof_ev = []
        if not done:
            return
        n = len(done)
        cols = [sorted(x[i] for x in done) for i in range(3)]
        med = [c[n // 2] for c in cols]
        p90 = [c[int(n * 0.9)] for c in cols]
        print(f" -- stream-decode prof ({n} layer calls, stream ms): gather med {med[0]:.3f} p90 {p90[0]:.3f}, "
              f"fused kernel med {med[1]:.3f} p90 {p90[1]:.3f}, worker wait+fold med {med[2]:.3f} p90 {p90[2]:.3f}"
              f"  -> x{nl} layers: {nl * med[0]:.1f} + {nl * med[1]:.1f} + {nl * med[2]:.1f} ms per token", flush = True)

    def _disable(self, where):
        self.disabled = True
        print(f" !! stream-decode: {where} failed; the worker serves the rest of this run\n"
              + traceback.format_exc(), flush = True)
