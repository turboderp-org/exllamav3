"""Precomputed MoE expert-placement profiles.

CPU MoE offload (-mcs/-mcl) splits each layer's experts between GPU and system RAM. Routing is
heavily skewed, so WHICH experts sit on the GPU decides how much traffic crosses the memory
bus. Upstream discovers that ordering at runtime, which is slow to converge: sweeps move at
most EXL3_MOE_CPU_SWAP_MAX (64) experts across ALL layers every EXL3_MOE_CPU_SWAP_INTERVAL
(128) decode steps, so a large model needs thousands of tokens to settle.

A profile is a table of per-expert selection counts measured OFFLINE and shipped as a file, so
placement starts correct at token zero. Nothing is probed or computed at load beyond one
argsort per layer.

Accepted sources (auto-detected by content, not extension):

  *.npz   usage census: counts_decode / counts_prefill, int64 [n_prompts, n_layers,
          n_experts], summed over prompts. Decode is the allocation signal.
  *.safetensors / *.exl3moe   counts int64 [n_layers, n_experts] (+ optional precomputed perm)
  *.json  {"<layer_key>": [c0..cN]} -- the format upstream's EXL3_MOE_CPU_SPLIT_STATS uses

Sources combine with weights, e.g. "wiki:1,code:3". Counts are normalized per layer
before weighting, so a corpus with more tokens does not simply win.
"""

import os, json, hashlib

MODEL_KEYS = ("architecture", "layers", "experts", "moe_intermediate_size", "hidden_size")
QUANT_KEYS = ("checkpoint_sha", "quant_method", "bits", "head_bits", "codebook")


def checkpoint_hash(directory, max_shards=None):
    """Content hash of a quantized checkpoint, cheap enough to run at every load.

    Hashes the safetensors HEADERS of every shard -- tensor names, dtypes, shapes and byte
    offsets -- plus each shard's size. That is a few MB of reads instead of the ~150 GB the
    weights occupy, and it still changes whenever the checkpoint changes: a requantization
    alters trellis shapes and offsets even at the same nominal bitrate, which a bits/codebook
    comparison alone would miss.

    Returns a 16-hex-char digest, or None if the directory holds no safetensors.
    """
    import glob, struct
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        return None
    if max_shards:
        files = files[:max_shards]
    h = hashlib.sha256()
    for f in files:
        try:
            sz = os.path.getsize(f)
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n > 64 * 1024 * 1024:      # refuse an implausible header
                    return None
                hdr = fh.read(n)
            h.update(os.path.basename(f).encode())
            h.update(struct.pack("<Q", sz))
            h.update(hdr)
        except Exception:
            return None
    return h.hexdigest()[:16]


def model_fingerprint(config, num_experts=None):
    """Identity of the exact artifact a profile was measured on.

    Routing depends on the model AND on how it was quantized -- different bitrates give
    different hidden states and therefore different expert choices -- so a profile is valid
    for one (model, quantization) pair. Split into MODEL_KEYS (never ignorable) and
    QUANT_KEYS (ignorable with an explicit override).
    """
    cd = getattr(config, "config_dict", None) or {}
    t = cd.get("text_config") if isinstance(cd.get("text_config"), dict) else cd
    q = cd.get("quantization_config") or {}
    arch = (cd.get("architectures") or [None])
    fp = {
        "architecture": arch[0] if arch else None,
        "layers": t.get("num_hidden_layers"),
        "experts": num_experts if num_experts is not None else
                   (t.get("n_routed_experts") or t.get("num_experts")),
        "moe_intermediate_size": t.get("moe_intermediate_size"),
        "hidden_size": t.get("hidden_size"),
        "quant_method": q.get("quant_method"),
        "bits": q.get("bits"),
        "head_bits": q.get("head_bits"),
        "codebook": q.get("codebook"),
    }
    d = getattr(config, "directory", None)
    if d and os.path.isdir(d):
        fp["checkpoint_sha"] = checkpoint_hash(d)
    return {k: v for k, v in fp.items() if v is not None}


def check_fingerprint(profile_fp, model_fp, path, allow_quant_mismatch=False):
    """Enforce the per-model / per-quantization policy. Returns a list of warnings."""
    warn = []
    if not profile_fp:
        warn.append(f" !! {os.path.basename(path)}: no fingerprint; cannot verify it was built "
                    f"for this model/quantization. Rebuild with util/moe_profile_build.py to "
                    f"embed one.")
        return warn
    bad_model = [(k, profile_fp.get(k), model_fp.get(k)) for k in MODEL_KEYS
                 if k in profile_fp and k in model_fp and profile_fp[k] != model_fp[k]]
    if bad_model:
        det = "; ".join(f"{k}: profile={a!r} model={b!r}" for k, a, b in bad_model)
        raise ValueError(
            f"{path}: this profile was built for a different MODEL ({det}). "
            f"Expert placement from another architecture is meaningless; rebuild the profile.")
    bad_quant = [(k, profile_fp.get(k), model_fp.get(k)) for k in QUANT_KEYS
                 if k in profile_fp and k in model_fp and profile_fp[k] != model_fp[k]]
    if bad_quant:
        det = "; ".join(f"{k}: profile={a!r} model={b!r}" for k, a, b in bad_quant)
        if not allow_quant_mismatch:
            head = "a different CHECKPOINT" if any(k == "checkpoint_sha" for k, _, _ in bad_quant) \
                   else "a different QUANTIZATION"
            raise ValueError(
                f"{path}: this profile was built on {head} ({det}).\n"
                f"Routing depends on the quantization, so placement measured at one bitrate "
                f"may be wrong at another.\nRebuild the profile for this quant, or pass "
                f"--moe_cpu_profile_any_quant to use it anyway.")
        warn.append(f" !! {os.path.basename(path)}: QUANTIZATION MISMATCH ({det}) -- used "
                    f"anyway because --moe_cpu_profile_any_quant is set. Placement "
                    f"may be worse than dynamic; measure before trusting it.")
    return warn


# Packed first: when a name resolves to several files, prefer the one that is cheapest to
# load. .exl3moe carries the ranking precomputed (mmap-free pread, no argsort); .npz is the
# census, which is complete but must be decompressed, summed over prompts and sorted. Ship
# both and serving picks the fast one while -score can still audit the census by path.
_EXTS = (".exl3moe", ".safetensors", ".npz", ".json")


def _np():
    import numpy as np
    return np


def search_dirs(model_dir=None):
    """Profiles are model-specific. Shipped-with-model wins, then env, then user cache."""
    d = []
    if model_dir:
        d.append(os.path.join(model_dir, "moe_profiles"))
    env = os.environ.get("EXL3_MOE_PROFILE_DIR")
    if env:
        d += [p for p in env.split(os.pathsep) if p]
    d.append(os.path.join(os.path.expanduser("~"), ".cache", "exllamav3", "moe_profiles"))
    return d


def resolve_path(spec, model_dir=None):
    """A path, or a bare name looked up in the search dirs (any accepted extension)."""
    if os.path.isfile(spec):
        return spec
    for d in search_dirs(model_dir):
        if not os.path.isdir(d):
            continue
        for ext in _EXTS:
            p = os.path.join(d, spec + ext)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"MoE profile '{spec}' not found. Looked in: " +
        ", ".join(search_dirs(model_dir)) +
        f" (extensions: {', '.join(_EXTS)}). Build one with util/moe_profile_build.py, "
        f"or point --moe_cpu_profile at a file.")


_ST_DTYPES = {"I64": "<i8", "I32": "<i4", "I16": "<i2", "U8": "|u1",
              "F64": "<f8", "F32": "<f4", "F16": "<f2"}


def _read_safetensors_pread(path, names = None):
    """Minimal safetensors reader: header + only the tensors named, via pread.

    -> (dict name -> ndarray, metadata dict). Avoids mmap (see the call site) and avoids a
    hard dependency on the safetensors package for the read path.

    `names` matters: a packed profile also carries the per-prompt census, which only -score
    reads. Every tensor has its own byte range, so preading just "ranking" skips the census
    entirely -- 0.02 ms regardless of how large the census is. Reading the whole file
    instead costs 0.09 ms and grows with it.
    """
    np = _np()
    import struct
    fd = os.open(path, os.O_RDONLY)
    try:
        n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
        hdr = json.loads(os.pread(fd, n, 8).decode("utf-8"))
        md = hdr.get("__metadata__", {}) or {}
        base = 8 + n
        out = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if names is not None and name not in names:
                continue
            dt = _ST_DTYPES.get(info["dtype"])
            if dt is None:
                raise ValueError(f"{path}: unsupported dtype {info['dtype']} for {name}")
            s0, s1 = info["data_offsets"]
            buf = os.pread(fd, s1 - s0, base + s0)
            if len(buf) != s1 - s0:
                raise ValueError(f"{path}: short read for {name}")
            out[name] = np.frombuffer(buf, dtype=np.dtype(dt)).reshape(info["shape"])
        return out, md
    finally:
        os.close(fd)


def load_counts(path):
    """-> (counts [n_layers, n_experts] float64, layer_keys or None, meta dict)"""
    np = _np()
    low = path.lower()

    if low.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        # decode is the allocation signal, since CPU offload runs in decode.
        def _flat(name):
            if name not in z:
                return None
            v = np.asarray(z[name])
            if v.ndim == 3:      # [n_prompts, n_layers, n_experts] -> sum over prompts
                v = v.sum(axis=0)
            if v.ndim != 2:
                raise ValueError(f"{path}: expected 2D or 3D '{name}', got shape {v.shape}")
            return v.astype(np.float64)

        dec, pre = _flat("counts_decode"), _flat("counts_prefill")
        plain = _flat("counts")
        a, k = (dec, "counts_decode") if dec is not None else \
               (plain, "counts") if plain is not None else (pre, "counts_prefill")
        if a is None:
            raise ValueError(f"{path}: no counts_decode/counts/counts_prefill array")

        # Ranking E experts from a few hits each is noise that masquerades as extreme skew:
        # unsampled experts sort to the tail and the "hot" head trivially covers everything.
        cells = max(a.shape[0] * a.shape[1], 1)
        per = a.sum() / cells
        if per < 20.0:
            alt = pre if k != "counts_prefill" else None
            if alt is not None and alt.sum() / cells >= 20.0:
                print(f" !! {os.path.basename(path)}: {k} has only {per:.1f} hits/expert; "
                      f"using counts_prefill ({alt.sum()/cells:.1f}/expert) instead")
                a, k = alt, "counts_prefill (decode too sparse)"
            else:
                print(f" !! {os.path.basename(path)}: only {per:.1f} hits/expert -- this profile "
                      f"is undersampled and its ranking is largely noise")
        # A positional .npz carries no layer identity, so rows would be matched to modules by
        # registration order -- silently wrong if the census came from another architecture.
        # Prefer exact keys from the sidecar our builder writes next to the .npz.
        keys, meta = None, {"source": "npz", "bank": k}
        side = os.path.splitext(path)[0] + ".meta.json"
        if os.path.isfile(side):
            try:
                m = json.load(open(side))
                keys = m.get("layer_keys") or None
                if keys and len(keys) != a.shape[0]:
                    print(f" !! {os.path.basename(side)}: {len(keys)} layer_keys but "
                          f"{a.shape[0]} rows; ignoring the sidecar")
                    keys = None
                meta["sidecar"] = side
                if isinstance(m.get("fingerprint"), dict):
                    meta["fingerprint"] = m["fingerprint"]
                for f in ("corpus", "prompts", "hits_per_expert_decode"):
                    if f in m:
                        meta[f] = m[f]
            except Exception as e:
                print(f" !! {os.path.basename(side)}: unreadable ({e}); using positional mapping")
        return a, keys, meta

    if low.endswith((".safetensors", ".exl3moe")):
        # Read with pread rather than mmap. A packed profile is ~36 pages, so mmap pays a
        # VMA setup plus a fault per page where pread is one syscall and a copy; measured
        # 0.0080 ms vs 0.0181 ms for safetensors.load_file, which additionally opens the
        # file a second time for metadata. exl3's own bulk weight loader is pread-based for
        # the same reason (exllamav3_ext/stloader.cpp), and it ships a non-mmap SafeOpen.
        # mmap would win on a large file, on random access, or on a mapping shared between
        # processes; none of those apply to a profile read once at load.
        # Never the census: serving needs the ranking, and counts only to merge.
        t, md = _read_safetensors_pread(path, names = ("ranking", "counts"))
        keys = json.loads(md["layer_keys"]) if "layer_keys" in md else None

        # A shipped profile may carry the ranking already computed, in which case loading is
        # a mmap and nothing else -- no decompression, no argsort. Measured on a 42x288
        # profile: 0.02 ms vs 1.34 ms for the .npz census it was built from, and the ranking
        # is bit-identical. Counts are then optional: they are only needed to MERGE several
        # profiles, which renormalizes per layer and re-sorts.
        if "ranking" in t:
            rank = np.asarray(t["ranking"]).astype(np.int64)
            if "counts" in t:
                return (np.asarray(t["counts"]).astype(np.float64), keys,
                        dict(md, source="safetensors", ranking=rank))
            # Rank-only file: synthesise counts that reproduce this exact order under the
            # descending stable argsort in build_ranking(), so a single-source spec needs no
            # special case and a weighted merge still behaves sanely.
            n_layers, n_exp = rank.shape
            counts = np.empty((n_layers, n_exp), dtype = np.float64)
            pos = np.arange(n_exp, dtype = np.float64)[::-1] + 1.0
            np.put_along_axis(counts, rank, np.broadcast_to(pos, rank.shape), axis = 1)
            return counts, keys, dict(md, source="safetensors", ranking=rank)

        if "counts" not in t:
            raise ValueError(f"{path}: no 'counts' or 'ranking' tensor")
        return np.asarray(t["counts"]).astype(np.float64), keys, dict(md, source="safetensors")

    if low.endswith(".json"):
        d = json.load(open(path))
        keys = list(d.keys())
        a = np.asarray([d[k] for k in keys], dtype=np.float64)
        return a, keys, {"source": "json"}

    raise ValueError(f"{path}: unrecognized profile format (want one of {_EXTS})")


def parse_spec(spec):
    """'code:3,wiki:1' -> [('code',3.0), ('wiki',1.0)]"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        w = 1.0
        if ":" in part and not os.path.exists(part):
            head, _, tail = part.rpartition(":")
            try:
                w = float(tail)
                part = head
            except ValueError:
                pass
        if w < 0:
            raise ValueError(f"negative weight in --moe_cpu_profile: {part}:{w}")
        out.append((part, w))
    if not out:
        raise ValueError("empty --moe_cpu_profile spec")
    return out


def build_ranking(spec, model_dir=None, num_experts=None, model_fp=None,
                  allow_quant_mismatch=False):
    """Merge the requested profiles into a hot->cold expert ranking per layer.

    Returns (ranking [n_layers, n_experts] int64, layer_keys or None, info dict). Row i is the
    expert ids of layer i ordered most- to least-used.
    """
    np = _np()
    parts = parse_spec(spec)
    acc = None
    keys = None
    used = []
    precomputed = None
    for name, w in parts:
        p = resolve_path(name, model_dir)
        c, k, meta = load_counts(p)
        if model_fp is not None:
            # NB: not `w` -- that is the source weight for this iteration.
            for _warn in check_fingerprint(meta.get("fingerprint"), model_fp, p,
                                           allow_quant_mismatch):
                print(_warn)
        if num_experts is not None and c.shape[1] != num_experts:
            raise ValueError(
                f"{p}: profile has {c.shape[1]} experts/layer but the model has {num_experts}. "
                f"This profile was built for a different model.")
        # Normalize per layer so corpus size does not decide the merge.
        tot = c.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        f = c / tot
        if acc is None:
            acc = f * w
            keys = k
        else:
            if f.shape != acc.shape:
                raise ValueError(
                    f"{p}: shape {f.shape} does not match {acc.shape} from earlier profiles")
            acc = acc + f * w
            keys = keys or k
        used.append({"name": name, "path": p, "weight": w, "shape": list(c.shape),
                     **{k: v for k, v in meta.items() if k != "ranking"}})
        if meta.get("ranking") is not None:
            precomputed = meta["ranking"]
    # A single source that already carries its ranking needs no sort at all -- that is the
    # point of the packed .exl3moe format. Merging several sources still has to renormalize
    # per layer and re-sort, so the fast path applies only when there is exactly one.
    if len(parts) == 1 and precomputed is not None:
        ranking = np.asarray(precomputed).astype(np.int64)
    else:
        # Descending sort; stable so equal counts keep natural expert order.
        ranking = np.argsort(-acc, axis=1, kind="stable").astype(np.int64)
    return ranking, keys, {"sources": used, "layers": int(acc.shape[0]),
                           "experts": int(acc.shape[1])}


def ranking_lookup(ranking, layer_keys):
    """Map a module key -> its ranking row. Falls back to MoE-layer ordinal when the profile
    carries no keys (the .npz format is positional)."""
    by_key = {}
    if layer_keys:
        for i, k in enumerate(layer_keys):
            if i < ranking.shape[0]:
                by_key[k] = ranking[i]
    return by_key
