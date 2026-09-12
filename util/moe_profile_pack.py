"""Pack a measured census (.npz) into a shipped profile (.exl3moe).

The census and the shipped profile want different things:

  .npz       [prompts, layers, experts] per-prompt counts. Keeps the prompt axis, which is
             what makes held-out re-scoring possible (moe_profile_build.py -score). This is
             the research artifact -- keep it, it is how a profile is audited later.

  .exl3moe   safetensors holding the summed counts AND the precomputed ranking, plus the
             fingerprint and layer keys as metadata. This is what a serving deployment
             loads: mmap, zero-copy, no decompression, no argsort, no numpy needed beyond
             the array view.

Measured on a 42 x 288 profile (load + produce the ranking the loader needs):

    .npz full census [P,L,E]        0.45 MB    1.34 ms
    .npz summed [L,E]               0.02 MB    0.48 ms
    .safetensors counts             0.10 MB    0.26 ms
    .exl3moe counts + ranking       0.15 MB    0.02 ms     <- 67x faster than the census

The rankings are bit-identical; this is purely moving work from load time to build time.
In absolute terms 1.3 ms is nothing against a multi-minute model load, so the reason to
prefer .exl3moe is not the milliseconds -- it is that it is the same container exl3 uses
for weights, it carries its own provenance, and the load path becomes genuinely O(1) as
the feature always claimed.

  python util/moe_profile_pack.py wiki_long.npz -o wiki_long.exl3moe
  python util/moe_profile_pack.py wiki_long.npz -o out.exl3moe --rank-only   # smallest
"""
import argparse, json, os, sys
import numpy as np


def pack(src, dst, bank = None, rank_only = False, meta_extra = None, with_census = True):
    z = np.load(src, allow_pickle = False)

    if bank is None:
        # Decode is the allocation signal -- that is the regime CPU offload runs in. Fall
        # back only if the decode bank is absent or empty.
        for cand in ("counts_decode", "counts", "counts_prefill"):
            if cand in z and np.asarray(z[cand]).sum() > 0:
                bank = cand
                break
    if bank is None:
        raise SystemExit(f"{src}: no non-empty counts bank")

    c = np.asarray(z[bank]).astype(np.float64)
    if c.ndim == 3:
        n_prompts = c.shape[0]
        c = c.sum(axis = 0)
    elif c.ndim == 2:
        n_prompts = 1
    else:
        raise SystemExit(f"{src}: expected 2D or 3D '{bank}', got {c.shape}")

    n_layers, n_experts = c.shape
    ranking = np.argsort(-c, axis = 1, kind = "stable").astype(np.int32)
    census = np.asarray(z[bank]) if (with_census and np.asarray(z[bank]).ndim == 3) else None

    md = {"source_bank": bank, "layers": str(n_layers), "experts": str(n_experts),
          "prompts": str(n_prompts), "has_ranking": "1",
          "packed_from": os.path.basename(src)}

    # Carry the sidecar's fingerprint and layer keys so the shipped file is self-describing:
    # the loader validates (model, checkpoint) identity from these, and keyed layer matching
    # avoids a positional off-by-one on architectures that interleave dense and sparse MLPs.
    side = os.path.splitext(src)[0] + ".meta.json"
    if os.path.isfile(side):
        sm = json.load(open(side))
        if "layer_keys" in sm:  md["layer_keys"] = json.dumps(sm["layer_keys"])
        if "fingerprint" in sm: md["fingerprint"] = json.dumps(sm["fingerprint"])
        if "capture" in sm:     md["capture"] = json.dumps(sm["capture"])
        if "corpus" in sm:      md["corpus"] = str(sm["corpus"])
    if meta_extra:
        md.update(meta_extra)

    from safetensors.numpy import save_file
    tensors = {"ranking": ranking}
    if not rank_only:
        # Counts are what a weighted multi-profile merge renormalizes over; without them a
        # merge can only use the implied order. 21 KB, so keep them unless asked not to.
        tensors["counts"] = c.astype(np.int64)
    if census is not None:
        # The per-prompt census, so one file is both the shipped profile and the audit
        # trail. Serving never reads these bytes: safetensors gives every tensor its own
        # byte range and the loader preads only "ranking", so carrying the census costs
        # disk and nothing else. int32 because counts are small (a few hundred per cell for
        # decode) but int16 would be one long run away from overflowing.
        tensors["census"] = census.astype(np.int32)
        md["census_shape"] = json.dumps(list(census.shape))
    save_file(tensors, dst, metadata = md)
    return {"bank": bank, "layers": n_layers, "experts": n_experts,
            "prompts": n_prompts, "bytes": os.path.getsize(dst),
            "census": census is not None}


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev = False, description = __doc__,
                                formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help = "census .npz produced by moe_profile_build.py")
    p.add_argument("-o", "--out", required = True, help = "output .exl3moe")
    p.add_argument("-b", "--bank", default = None,
                   choices = ["counts_decode", "counts_prefill", "counts"],
                   help = "which bank to pack (default: decode, else whichever is non-empty)")
    p.add_argument("--rank-only", action = "store_true",
                   help = "omit counts; smallest file, but cannot be weighted in a merge")
    p.add_argument("--no-census", action = "store_true",
                   help = "omit the per-prompt census; smaller, but -score can no longer "
                          "re-audit the profile without the original .npz")
    a = p.parse_args()

    info = pack(a.src, a.out, a.bank, a.rank_only, with_census = not a.no_census)
    print(f" -- packed {a.src} -> {a.out}")
    print(f"    bank {info['bank']}, {info['layers']} layers x {info['experts']} experts, "
          f"{info['prompts']} prompt(s), {info['bytes']:,} bytes")
    print("    load path is now mmap + zero argsort; keep the .npz for -score re-auditing")
