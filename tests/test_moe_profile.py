from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

# Import the module directly so the test runs without CUDA or a built extension.
_SRC = Path(__file__).resolve().parents[1] / "exllamav3" / "model" / "moe_profile.py"
_spec = importlib.util.spec_from_file_location("moe_profile", _SRC)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


# Real MoE geometries. Placement is architecture-agnostic; only the profile is model-specific.
GEOMETRIES = [
    ("Mixtral-8x7B", 32, 8),
    ("Qwen3-30B-A3B", 48, 128),
    ("Qwen3-235B-A22B", 94, 128),
    ("Qwen3-Next-80B", 48, 512),
    ("DeepSeek-V3", 61, 256),
    ("GLM-5.3-Flash", 42, 288),
    ("Llama-4-Scout", 48, 16),
]

FP = {
    "architecture": "TestForCausalLM", "layers": 4, "experts": 8,
    "hidden_size": 4096, "quant_method": "exl3", "bits": 4.05,
    "codebook": "mul1", "checkpoint_sha": "aaaabbbbccccdddd",
}


def _profile_dir(tmp_path: Path) -> Path:
    d = tmp_path / "moe_profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_npz(dirpath: Path, name: str, layers: int, experts: int, *,
               descending: bool = True, decode_hits: int = 10_000,
               fingerprint: dict | None = None, layer_keys: list[str] | None = None):
    """Census layout: counts_decode/counts_prefill [prompts, layers, experts]."""
    order = np.arange(experts)[::-1] if descending else np.arange(experts)
    counts = np.zeros((1, layers, experts), dtype=np.int64)
    for l in range(layers):
        counts[0, l] = (order + 1) * decode_hits
    np.savez(dirpath / f"{name}.npz", counts_decode=counts,
             counts_prefill=np.zeros_like(counts))
    if fingerprint is not None or layer_keys is not None:
        meta = {}
        if fingerprint is not None:
            meta["fingerprint"] = fingerprint
        if layer_keys is not None:
            meta["layer_keys"] = layer_keys
        (dirpath / f"{name}.meta.json").write_text(json.dumps(meta))


# --- format handling ---------------------------------------------------------------------

def test_npz_sums_prompts_and_prefers_decode_bank(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8)
    ranking, _, info = mp.build_ranking("p", str(tmp_path), 8)
    assert ranking.shape == (4, 8)
    assert info["sources"][0]["bank"] == "counts_decode"
    assert list(ranking[0]) == list(range(8))          # hot -> cold


def test_json_format_and_layer_keys(tmp_path):
    d = _profile_dir(tmp_path)
    keys = [f"model.layers.{i}.mlp" for i in range(4)]
    (d / "j.json").write_text(json.dumps({k: list(range(8)) for k in keys}))
    ranking, got_keys, _ = mp.build_ranking("j", str(tmp_path), 8)
    assert got_keys == keys
    assert list(ranking[0]) == list(reversed(range(8)))


def test_sidecar_layer_keys_are_used(tmp_path):
    d = _profile_dir(tmp_path)
    keys = [f"model.layers.{i + 3}.mlp" for i in range(4)]   # MoE may start after dense layers
    _write_npz(d, "p", 4, 8, layer_keys=keys)
    _, got_keys, _ = mp.build_ranking("p", str(tmp_path), 8)
    assert got_keys == keys


def test_undersampled_decode_falls_back_to_prefill(tmp_path, capsys):
    d = _profile_dir(tmp_path)
    counts_dec = np.ones((1, 4, 8), dtype=np.int64)            # 1 hit/expert: pure noise
    counts_pre = np.full((1, 4, 8), 1000, dtype=np.int64)
    np.savez(d / "p.npz", counts_decode=counts_dec, counts_prefill=counts_pre)
    _, _, info = mp.build_ranking("p", str(tmp_path), 8)
    assert "prefill" in info["sources"][0]["bank"]
    assert "hits/expert" in capsys.readouterr().out


# --- merging -----------------------------------------------------------------------------

def test_weighting_pulls_toward_the_heavier_source(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "a", 4, 8, descending=True)
    _write_npz(d, "b", 4, 8, descending=False)
    heavy_a, _, _ = mp.build_ranking("a:9,b:1", str(tmp_path), 8)
    heavy_b, _, _ = mp.build_ranking("a:1,b:9", str(tmp_path), 8)
    assert list(heavy_a[0]) == list(range(8))
    assert list(heavy_b[0]) == list(reversed(range(8)))


def test_merge_normalizes_per_layer_so_corpus_size_does_not_decide(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "big", 4, 8, descending=True, decode_hits=1_000_000)
    _write_npz(d, "small", 4, 8, descending=False, decode_hits=10)
    # Equal weights: the 100000x larger corpus must not simply win.
    merged, _, _ = mp.build_ranking("big:1,small:1", str(tmp_path), 8)
    assert merged.shape == (4, 8)
    heavy_small, _, _ = mp.build_ranking("big:1,small:5", str(tmp_path), 8)
    assert list(heavy_small[0]) == list(reversed(range(8)))


@pytest.mark.parametrize("spec,expected", [
    ("a", [("a", 1.0)]),
    ("a,b", [("a", 1.0), ("b", 1.0)]),
    ("a:2.5, b ", [("a", 2.5), ("b", 1.0)]),
])
def test_spec_parsing(spec, expected):
    assert mp.parse_spec(spec) == expected


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        mp.parse_spec("a:-1")


# --- model / quantization identity policy -------------------------------------------------

def test_expert_count_mismatch_is_rejected(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8)
    with pytest.raises(ValueError, match="different model"):
        mp.build_ranking("p", str(tmp_path), 16)


def test_model_mismatch_is_fatal_even_with_override(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8, fingerprint=FP)
    other = dict(FP, architecture="SomethingElseForCausalLM")
    for allow in (False, True):
        with pytest.raises(ValueError, match="different MODEL"):
            mp.build_ranking("p", str(tmp_path), 8, model_fp=other,
                             allow_quant_mismatch=allow)


def test_checkpoint_mismatch_blocks_unless_overridden(tmp_path, capsys):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8, fingerprint=FP)
    other = dict(FP, checkpoint_sha="9999888877776666")
    with pytest.raises(ValueError, match="different CHECKPOINT"):
        mp.build_ranking("p", str(tmp_path), 8, model_fp=other)
    mp.build_ranking("p", str(tmp_path), 8, model_fp=other, allow_quant_mismatch=True)
    assert "MISMATCH" in capsys.readouterr().out


def test_missing_fingerprint_warns_but_loads(tmp_path, capsys):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8)                                  # e.g. a shipped census
    mp.build_ranking("p", str(tmp_path), 8, model_fp=dict(FP))
    assert "no fingerprint" in capsys.readouterr().out


# --- lookup and errors --------------------------------------------------------------------

def test_missing_profile_names_every_search_dir(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        mp.build_ranking("nope", str(tmp_path), 8)
    assert "moe_profiles" in str(e.value)


def test_model_dir_profile_beats_user_cache(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 4, 8)
    dirs = mp.search_dirs(str(tmp_path))
    assert dirs[0] == str(d)


def test_env_profile_dir_is_searched(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    _write_npz(extra, "e", 4, 8)
    monkeypatch.setenv("EXL3_MOE_PROFILE_DIR", str(extra))
    ranking, _, _ = mp.build_ranking("e", None, 8)
    assert ranking.shape == (4, 8)


# --- architecture independence -------------------------------------------------------------

@pytest.mark.parametrize("name,layers,experts", GEOMETRIES)
def test_any_moe_geometry_loads(tmp_path, name, layers, experts):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", layers, experts)
    ranking, _, _ = mp.build_ranking("p", str(tmp_path), experts)
    assert ranking.shape == (layers, experts)
    assert list(ranking[0]) == list(range(experts))


def test_profiles_do_not_leak_between_architectures(tmp_path):
    d = _profile_dir(tmp_path)
    _write_npz(d, "p", 42, 288)
    for _, _, experts in GEOMETRIES:
        if experts == 288:
            continue
        with pytest.raises(ValueError):
            mp.build_ranking("p", str(tmp_path), experts)


# --- checkpoint hashing ---------------------------------------------------------------------

def test_checkpoint_hash_none_without_safetensors(tmp_path):
    assert mp.checkpoint_hash(str(tmp_path)) is None


def test_checkpoint_hash_reads_only_headers(tmp_path):
    """A shard's payload must not affect the hash cost; only the header identifies it."""
    import struct
    header = json.dumps({"__metadata__": {"k": "v"}}).encode()
    for payload, name in ((b"\x00" * 1024, "model-00001.safetensors"),):
        with open(tmp_path / name, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(payload)
    h1 = mp.checkpoint_hash(str(tmp_path))
    assert h1 is not None and len(h1) == 16
    # Same header, different payload size -> different hash (size is part of the identity).
    with open(tmp_path / "model-00001.safetensors", "ab") as f:
        f.write(b"\x00" * 16)
    assert mp.checkpoint_hash(str(tmp_path)) != h1


# ---------------------------------------------------------------------------------------
# Held-out capture scoring. These encode the failure that shipped: a ranking fitted to one
# sample scores ~93% against that sample and near-random on anything else.
# ---------------------------------------------------------------------------------------

def _build_mod():
    """Import the builder wherever it sits: beside this file in the patch dir, or at
    util/moe_profile_build.py once applied to an exllamav3 tree."""
    import importlib.util
    here = Path(__file__).resolve().parent
    for cand in (here / "moe_profile_build.py",
                 here.parent / "util" / "moe_profile_build.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("mpb", cand)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    pytest.skip("moe_profile_build.py not found")


def test_capture_at_is_hit_weighted():
    np = pytest.importorskip("numpy")
    b = _build_mod()
    fit = np.array([[10.0, 5.0, 1.0, 0.0]])       # ranking 0,1,2,3
    test = np.array([[7.0, 3.0, 5.0, 5.0]])       # top-2 by fit -> 7+3 of 20
    assert abs(b.capture_at(fit, test, 2) - 0.5) < 1e-9
    assert abs(b.capture_at(fit, test, 4) - 1.0) < 1e-9


def test_capture_uniform_routing_scores_at_chance():
    """No signal to find: held-out capture must land on R/E, not above it."""
    np = pytest.importorskip("numpy")
    b = _build_mod()
    rng = np.random.default_rng(0)
    c = rng.poisson(20, size=(32, 8, 288)).astype(float)
    rep = b.capture_report(c, 108)
    assert abs(rep["head"]["held_out"] - 108 / 288) < 0.03


def test_capture_stable_skew_generalizes():
    np = pytest.importorskip("numpy")
    b = _build_mod()
    rng = np.random.default_rng(1)
    base = np.zeros((8, 288)); base[:, :60] = 500.0; base += 5.0
    c = rng.poisson(np.broadcast_to(base, (32, 8, 288))).astype(float)
    rep = b.capture_report(c, 108)
    assert rep["head"]["held_out"] > 0.90
    assert rep["head"]["held_out"] <= rep["head"]["oracle"] + 1e-9


def test_capture_exposes_per_prompt_overfit():
    """The shipped bug: each prompt hot on its OWN experts. In-sample looks like signal;
    held-out must collapse to chance."""
    np = pytest.importorskip("numpy")
    b = _build_mod()
    rng = np.random.default_rng(2)
    c = np.full((32, 8, 288), 1.0)
    for i in range(32):
        c[i][:, rng.choice(288, 60, replace=False)] += 500.0
    c = rng.poisson(c).astype(float)
    rep = b.capture_report(c, 108)
    assert rep["head"]["in_sample"] > rep["head"]["held_out"] + 0.10
    assert rep["head"]["held_out"] < 0.45


def test_capture_report_refuses_to_score_too_few_prompts():
    """A single-prompt census cannot be scored; returning None is what forces the warning
    instead of an in-sample number presented as capture."""
    np = pytest.importorskip("numpy")
    b = _build_mod()
    assert b.capture_report(np.ones((1, 8, 288)), 108) is None
    assert b.capture_report(np.ones((3, 8, 288)), 108) is None
    assert b.capture_report(np.ones((4, 8, 288)), 108) is not None



# ---------------------------------------------------------------------------------------
# Packed .exl3moe profiles. A shipped profile carries its ranking precomputed, so loading
# is a mmap and nothing else. These assert the packed order is identical to argsorting the
# census it came from -- the whole correctness claim of the format.
# ---------------------------------------------------------------------------------------

def _pack_mod():
    import importlib.util
    here = Path(__file__).resolve().parent
    for cand in (here / "moe_profile_pack.py", here.parent / "util" / "moe_profile_pack.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("mpp", cand)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    pytest.skip("moe_profile_pack.py not found")


def _census(tmp_path, P=6, L=4, E=32, seed=0):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    dec = rng.poisson(20, size=(P, L, E)).astype(np.int64)
    pre = rng.poisson(50, size=(P, L, E)).astype(np.int64)
    src = tmp_path / "c.npz"
    np.savez_compressed(src, counts_decode=dec, counts_prefill=pre)
    return str(src), dec


def test_packed_ranking_matches_census(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, dec = _census(tmp_path)
    dst = str(tmp_path / "c.exl3moe")
    _pack_mod().pack(src, dst)
    want = np.argsort(-dec.sum(axis=0).astype(np.float64), axis=1, kind="stable")
    got, _, meta = mp.load_counts(dst)
    assert meta.get("ranking") is not None
    assert (meta["ranking"] == want).all()


def test_packed_single_source_skips_the_sort(tmp_path):
    """A single packed source must return its stored ranking verbatim."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, dec = _census(tmp_path)
    dst = str(tmp_path / "c.exl3moe")
    _pack_mod().pack(src, dst)
    want = np.argsort(-dec.sum(axis=0).astype(np.float64), axis=1, kind="stable")
    rank, _, _ = mp.build_ranking(dst)
    assert (rank == want).all()


def test_rank_only_file_reproduces_its_own_order(tmp_path):
    """--rank-only drops counts. The synthesised counts must still sort back to the
    stored order, so a single-source spec needs no special case."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, dec = _census(tmp_path)
    dst = str(tmp_path / "r.exl3moe")
    _pack_mod().pack(src, dst, rank_only=True)
    counts, _, meta = mp.load_counts(dst)
    assert (np.argsort(-counts, axis=1, kind="stable") == meta["ranking"]).all()


def test_pack_carries_fingerprint_and_layer_keys(tmp_path):
    """The sidecar's identity travels with the packed file, or the loader cannot validate
    (model, checkpoint) and keyed layer matching silently degrades to positional."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, _ = _census(tmp_path)
    json.dump({"layer_keys": [f"model.layers.{i}.mlp" for i in range(4)],
               "fingerprint": {"architecture": "X", "layers": 4, "experts": 32}},
              open(str(tmp_path / "c.meta.json"), "w"))
    dst = str(tmp_path / "c.exl3moe")
    _pack_mod().pack(src, dst)
    _, keys, meta = mp.load_counts(dst)
    assert keys == [f"model.layers.{i}.mlp" for i in range(4)]
    # decoded on load, not left as the JSON text safetensors metadata stores
    assert meta["fingerprint"]["experts"] == 32


def test_pack_prefers_decode_bank(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, dec = _census(tmp_path)
    dst = str(tmp_path / "c.exl3moe")
    info = _pack_mod().pack(src, dst)
    assert info["bank"] == "counts_decode"


def test_packed_profile_wins_resolution(tmp_path):
    """Shipping .npz and .exl3moe side by side is only useful if the packed one is chosen.
    Guards against the resolution order silently reverting."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    d = tmp_path / "moe_profiles"
    d.mkdir()
    rng = np.random.default_rng(3)
    dec = rng.poisson(20, size=(6, 4, 32)).astype(np.int64)
    np.savez_compressed(str(d / "p.npz"), counts_decode=dec)
    _pack_mod().pack(str(d / "p.npz"), str(d / "p.exl3moe"))
    assert mp.resolve_path("p", str(tmp_path)).endswith(".exl3moe")
    # and the packed path must still produce the census's ranking
    want = np.argsort(-dec.sum(axis=0).astype(np.float64), axis=1, kind="stable")
    rank, _, _ = mp.build_ranking("p", model_dir=str(tmp_path))
    assert (rank == want).all()


def test_census_travels_in_the_packed_file(tmp_path):
    """One file must serve both uses: the ranking for serving and the per-prompt census for
    -score. Once counts are summed the prompt axis is gone and the profile can no longer be
    audited, so losing the census silently would be the worst kind of regression."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, dec = _census(tmp_path)
    dst = str(tmp_path / "p.exl3moe")
    info = _pack_mod().pack(src, dst)
    assert info["census"] is True
    import struct
    with open(dst, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n).decode("utf-8"))
    assert hdr["census"]["shape"] == list(dec.shape)


def test_serving_read_skips_the_census(tmp_path):
    """Serving must not pay for the census. The loader names the tensors it wants, so the
    census byte range is never read however large it grows."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, _ = _census(tmp_path)
    dst = str(tmp_path / "p.exl3moe")
    _pack_mod().pack(src, dst)
    t, _ = mp._read_safetensors_pread(dst, names=("ranking", "counts"))
    assert set(t) == {"ranking", "counts"}
    assert "census" not in t


def test_packed_fingerprint_is_a_dict_not_json_text(tmp_path):
    """safetensors metadata is string-valued, so structured fields are stored as JSON text.
    They must be decoded on load: check_fingerprint() indexes them, and a raw string raises
    "string indices must be integers" at model load -- silently only for packed profiles."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    src, _ = _census(tmp_path)
    fp = {"architecture": "X", "layers": 4, "experts": 32, "checkpoint_sha": "deadbeef"}
    json.dump({"layer_keys": [f"l{i}" for i in range(4)], "fingerprint": fp},
              open(str(tmp_path / "c.meta.json"), "w"))
    dst = str(tmp_path / "c.exl3moe")
    _pack_mod().pack(src, dst)
    _, _, meta = mp.load_counts(dst)
    assert isinstance(meta.get("fingerprint"), dict), "fingerprint must decode to a dict"
    assert meta["fingerprint"]["checkpoint_sha"] == "deadbeef"
    # and the identity check must run against it without raising
    warns = list(mp.check_fingerprint(meta["fingerprint"], fp, dst, False))
    assert warns == []
    bad = dict(fp, checkpoint_sha="different")
    assert list(mp.check_fingerprint(meta["fingerprint"], bad, dst, True)), "should warn"
