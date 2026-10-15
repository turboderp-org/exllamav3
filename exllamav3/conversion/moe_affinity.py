"""
Affinity-weighted Hessians for block-sparse (MoE) experts, convert.py --moe_affinity_eps.

Calibration runs every expert on every token, so by default all experts of a layer share one gate/up Hessian
(H = sum_t x_t x_t^T over all tokens) and each down Hessian counts every token equally, although in inference an
expert only sees the tokens routed to it. With this option each expert e is quantized against

    H_e  ~  sum_t (eps + p_te / mean_t p_te) x_t x_t^T  =  (eps * mean_t p_te * sum_t x_t x_t^T + sum_t p_te x_t x_t^T) / mean_t p_te

where p is the full router softmax (scale does not matter: the quantizer normalizes H and regularizes it relative to
its diagonal). eps keeps every token in play, so rarely routed experts still see the whole input distribution.

  - down: the capture accumulates the weighted sum next to the plain one (Linear.capture_H, capture_row_w), and the
    two are combined after the capture pass (combine_down)
  - gate/up: one in x in matrix per expert does not fit (512 x 2560^2 fp32 is 13 GB per layer), so the capture keeps
    the plain shared Hessian plus a fixed random sample of rows (expert inputs and affinities, BlockSparseMLP
    .forward), and quantize_gate_up builds the per-expert Hessians from the sample a few experts at a time, just
    before quantizing them: H_e = eps * mean_t p_te * H_shared / N + sum_s p_se x_s x_s^T / n
"""
import torch

EXPERT_BATCH = 16
ROW_CHUNK = 32768


def make_store(eps: float, max_rows: int, total_rows: int, seed: int = 0):
    """Store passed to the capture forward as params["moe_affinity"]"""
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "eps": eps,
        "frac": min(1.0, max_rows / max(total_rows, 1)),
        "generator": g,
        "layers": {},
    }


def combine_down(capture_H: dict, eps: float):
    """Fold the affinity-weighted down Hessians into the plain ones. Returns the number of tensors combined."""
    n = 0
    for v in capture_H.values():
        if "H_aff" not in v:
            continue
        s = float(v.pop("aff_sum"))
        H_aff = v.pop("H_aff")
        if s <= 0 or v["count"] == 0:
            continue
        v["H"].mul_(eps * s / v["count"]).add_(H_aff)
        v["count"] = (1.0 + eps) * s
        n += 1
    return n


@torch.inference_mode()
def quantize_gate_up(module, linears: list, capture_H: dict, store: dict, quantize_fn, device):
    """
    Quantize the gate and up projections of every block-sparse MLP in module whose capture went through the
    store, each expert against its own affinity-weighted Hessian. quantize_fn(linears, capture_H) quantizes a list
    of linears with the given Hessians. Returns the linears not handled here.
    """
    from ..modules.block_sparse_mlp import BlockSparseMLP
    eps = store["eps"]
    done = set()
    for m in module:
        if not isinstance(m, BlockSparseMLP) or m.key not in store["layers"] or not m.gated:
            continue
        st = store["layers"].pop(m.key)
        if not st["x"]:
            continue
        pending = {id(l) for l in linears}
        experts = [(e, g, u) for e, (g, u) in enumerate(zip(m.gates, m.ups)) if id(g) in pending and id(u) in pending]
        if not experts:
            continue
        shared_qmap = experts[0][1].qmap
        shared = capture_H[shared_qmap]
        H_shared = shared["H"].to(device, torch.float) / shared["count"]
        p_mean = (st["psum"] / st["rows"]).to(device, torch.float)
        X = torch.cat(st["x"]).to(device)
        P = torch.cat(st["p"]).to(device)
        st = None
        n = X.shape[0]
        print(f" -- Affinity Hessians: {m.key}, {len(experts)} experts, {n:,} sampled rows, eps {eps}", flush = True)

        for b in range(0, len(experts), EXPERT_BATCH):
            batch = experts[b : b + EXPERT_BATCH]
            idx = torch.tensor([e for e, _, _ in batch], device = device)
            B = torch.zeros((len(batch), X.shape[1], X.shape[1]), dtype = torch.float, device = device)
            for r in range(0, n, ROW_CHUNK):
                xc = X[r : r + ROW_CHUNK].float()
                pc = P[r : r + ROW_CHUNK].index_select(1, idx).float()
                for i in range(len(batch)):
                    # B[i] += xc^T diag(pc[:, i]) xc
                    B[i].addmm_((xc * pc[:, i : i + 1]).T, xc)
            local = {}
            sub = []
            saved = []
            for i, (e, g, u) in enumerate(batch):
                H = B[i].div_(n).add_(H_shared, alpha = eps * p_mean[e].item())
                qmap = f"{shared_qmap}.aff{e}"
                local[qmap] = {
                    "H": H, "first_key": g.key, "count": 1, "finalized": False, "num_total": 0,
                    "inf_nan": torch.zeros(2, dtype = torch.long, device = device), "device": device,
                }
                saved += [(g, g.qmap), (u, u.qmap)]
                g.qmap = u.qmap = qmap
                sub += [g, u]
            try:
                quantize_fn(sub, local)
            finally:
                for lin, q in saved:
                    lin.qmap = q
            done.update(id(l) for l in sub)
            del B, local
        del X, P, H_shared
    return [l for l in linears if id(l) not in done]
