"""
Guards for the fused-decode / runtime-LoRA interaction (Session 22 finding).

The fused multi-linear decode kernels (exl3_mgemm over MultiLinear pairs, the
BC_GatedMLP bsz-1 graph) read trellis storage directly and never call
Linear.forward -- which is the only place a runtime LoRA (model.lora.LoRA) is
applied. Before the fix, token-by-token generation on an EXL3-quantized base
silently dropped the k/v (and gated-attn q), gate/up (and BC-path down) LoRA
components, so an adapter that visibly steered the training-side forward
looked like a no-op at inference. The mgemm branches in attention
(attn/sliding_attn) and GatedMLP now stay fused and add the low-rank delta
onto the mgemm output (pre-RoPE / pre-activation, matching Linear.forward
semantics -- MultiLinear pairs are bias/softcap/scale-free by construction).
The whole-block graph paths (bc_attn/bc_swa, the BC bsz-1 MLP graph, the GDN
split graph, the fused MoE expert kernels) cannot take a post-hoc delta and
still fall back to unfused dispatch while any involved Linear carries LoRA
tensors.

Because the failure mode is SILENT (generation stays coherent, the adapter
just doesn't run), these tests are tripwires on the guard conditions in the
source itself, dependency-free so they run in any container. The semantics of
the helper are tested with real imports where torch is available. The real
end-to-end check -- an adapter visibly changing bsz=1 decode output on an EXL3
quant, fused kernels present -- needs a GPU and a quantized model; it is on
the handoff box list, not here.
"""

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*rel: str) -> str:
    with open(os.path.join(REPO, *rel), encoding = "utf8") as f:
        return f.read()


def test_helper_defined():
    src = _src("exllamav3", "modules", "linear.py")
    assert "def has_runtime_lora(" in src


@pytest.mark.parametrize("fname", ["attn.py", "sliding_attn.py"])
def test_attention_mgemm_branches_apply_lora(fname):
    # The mgemm branches stay fused under a runtime LoRA; the low-rank delta
    # must be added onto the mgemm output INSIDE each branch (i.e. before the
    # head reshape and RoPE that follow project_qkv's branch bodies).
    src = _src("exllamav3", "modules", fname)
    qg_block = src[src.index("self.multi_qg.ptrs_trellis"):
                   src.index("self.multi_kv is None")]
    assert "self.q_proj.apply_lora(xf" in qg_block, \
        f"{fname}: multi_qg branch lost its q LoRA delta"
    assert "self.g_proj.apply_lora(xf" in qg_block, \
        f"{fname}: multi_qg branch lost its g LoRA delta"
    i = src.index("self.multi_kv.ptrs_trellis")
    kv_block = src[i:src.index("self.num_q_heads, self.head_dim", i)]
    assert "self.k_proj.apply_lora(xf" in kv_block, \
        f"{fname}: multi_kv branch lost its k LoRA delta"
    assert "self.v_proj.apply_lora(xf" in kv_block, \
        f"{fname}: multi_kv branch lost its v LoRA delta"


def test_gated_mlp_fused_branches_guarded():
    src = _src("exllamav3", "modules", "mlp.py")
    # Checked across ALL slices (uniform BC decision), once per forward.
    assert "gu_lora = has_runtime_lora(*self.gates, *self.ups)" in src
    assert "down_lora = has_runtime_lora(*self.downs)" in src
    # gate/up mgemm branch stays fused and adds the LoRA delta onto the mgemm
    # output BEFORE the activation (gate/up inject pre-activation); down goes
    # through Linear.forward, which applies its own LoRA
    i = src.index("self.multi_gu[s].ptrs_trellis")
    gu_block = src[i:src.index("activation_fn_call", i)]
    assert "self.gates[s].apply_lora(xf" in gu_block, \
        "GatedMLP: multi_gu branch lost its gate LoRA delta"
    assert "self.ups[s].apply_lora(xf" in gu_block, \
        "GatedMLP: multi_gu branch lost its up LoRA delta"
    # the BC bszN graph (v1.2.0: generalized from bsz-1 to bsz*q_len <=
    # MAX_BSZN) fuses the whole MLP (gate/up/act/down) and cannot take a
    # post-hoc delta, so it must yield to an unfused branch when ANY of the
    # three carries a LoRA
    assert re.search(
        r"self\.bc is not None and bsz \* q_len <= MAX_BSZN[^:]*?"
        r"not \(gu_lora or down_lora\)",
        src, re.S), "GatedMLP: BC bszN branch lost its runtime-LoRA guard"


@pytest.mark.parametrize("fname", ["attn.py", "sliding_attn.py"])
def test_bc_attn_graph_dispatch_guarded(fname):
    # bc_attn/bc_swa graph-captured decode blocks run projections through
    # o_proj as one C++ call; the dispatch (not the cached graph build) must
    # check for a runtime LoRA on every involved projection
    src = _src("exllamav3", "modules", fname)
    assert re.search(
        r"bsz <= _bc_max_bsz and seqlen <= _bc_max_qlen and\s*"
        r"not has_runtime_lora\(self\.q_proj, self\.k_proj, self\.v_proj,",
        src), f"{fname}: graph-captured decode dispatch lost its runtime-LoRA guard"


@pytest.mark.parametrize("fname", ["attn.py", "sliding_attn.py"])
def test_sliced_qkv_bundle_dispatch_guarded(fname):
    # Upstream v1.4.9's sliced Q/K/V(/G) bundle (project_qkv_sliced) is one
    # mgemm launch with no LoRA epilogue and takes precedence over the pairwise
    # bundles at decode; the dispatch must skip it while any involved
    # projection carries a runtime LoRA so the pairwise branches (which add
    # the delta) run instead
    src = _src("exllamav3", "modules", fname)
    assert re.search(
        r"self\.multi_qkv is not None and bsz \* q_len <= 32 and\s*"
        r"not has_runtime_lora\(self\.q_proj, self\.k_proj, self\.v_proj, self\.g_proj\)",
        src), f"{fname}: sliced qkv bundle dispatch lost its runtime-LoRA guard"


def test_gdn_sliced_qkvz_bundle_dispatch_guarded():
    # Same for the GDN qkv/z sliced bundle on the non-graph torch path
    src = _src("exllamav3", "modules", "gated_delta_net.py")
    assert re.search(
        r"getattr\(self, \"multi_qkvz\", None\) is not None and bsz \* seqlen <= 32 and\s*"
        r"not has_runtime_lora\(self\.qkv_proj, self\.z_proj\)",
        src), "GDN: sliced qkv/z bundle dispatch lost its runtime-LoRA guard"


def test_mla_graph_dispatch_guarded():
    # v1.3.0: MLAttention (DeepseekV3-class) has a graph-captured decode block
    # binding q/q_a/kv_a/o projections' base trellis (inner.bc); the dispatch
    # must fall back to the Linear.forward-based path under a runtime LoRA.
    # (q_a_proj here is DeepSeek's architectural low-rank factorization, not an
    # adapter; kv_b_proj is absorbed into w_uk_flat on every path.)
    src = _src("exllamav3", "modules", "mla_attn.py")
    assert re.search(
        r"params\.get\(\"causal\", True\)[^:]*?"
        r"not has_runtime_lora\(self\.q_proj, self\.q_a_proj,",
        src, re.S), "MLAttention: graph-captured decode lost its runtime-LoRA guard"


def test_plain_mlp_bsz1_graph_guarded():
    # BC_MLP (v1.0.0, non-gated up/act/down for NemotronH-class models) runs
    # the whole MLP in one graph-captured C++ call, reading the up/down
    # trellis directly; it must yield to the per-linear path under a LoRA
    src = _src("exllamav3", "modules", "mlp.py")
    assert re.search(
        r"self\.bc is not None and bsz == 1 and q_len == 1[^:]*?"
        r"not has_runtime_lora\(\*self\.ups, \*self\.downs\)",
        src, re.S), "MLP: BC bsz-1 branch lost its runtime-LoRA guard"


def test_mamba2_bsz1_graph_guarded():
    # BC_Mamba2 runs the whole layer (in_proj through o_proj) in one
    # graph-captured call, reading both projection trellises directly
    # (v1.2.0: generalized from bsz-1 to bsz/seqlen up to the BC limits)
    src = _src("exllamav3", "modules", "mamba2.py")
    assert re.search(
        r"self\.bc is not None and save_state and[^:]*?"
        r"not has_runtime_lora\(self\.in_proj, self\.o_proj\)",
        src, re.S), "Mamba2: BC fused decode lost its runtime-LoRA guard"


def test_gdn_split_bsz1_graph_guarded():
    # BC_GatedDeltaNetSplit runs the whole layer in one call, reading qkv/z/o
    # trellis and the merged (base-weights-only) ba_weight_t buffer directly
    # (v1.2.0: generalized from bsz-1 to bsz/seqlen up to the BC limits)
    src = _src("exllamav3", "modules", "gated_delta_net.py")
    assert re.search(
        r"self\.bc_split and save_state and[^:]*?"
        r"not has_runtime_lora\(self\.qkv_proj, self\.z_proj, self\.b_proj,",
        src, re.S), "GatedDeltaNet: split fused decode lost its runtime-LoRA guard"


def test_moe_expert_dispatch_guarded():
    # Every fused expert path must yield to the per-expert torch path (the
    # mlp() closure calling gates/ups/downs .forward) while an adapter is
    # loaded on any routed expert projection
    src = _src("exllamav3", "modules", "block_sparse_mlp.py")
    assert "experts_lora = has_runtime_lora(*self.gates, *self.ups, *self.downs)" in src
    # branch selection forces the torch-capable branch
    assert re.search(
        r"no_reconstruct or experts_lora or",
        src), "BlockSparseMLP: torch-path branch selection lost experts_lora"
    # exl3_moe fused kernel skipped under LoRA
    assert re.search(
        r"self\.fused_mode_buffers is not None and not experts_lora",
        src), "BlockSparseMLP: exl3_moe fused path lost its runtime-LoRA guard"
    # BC single-expert graph/DQ kernels skipped under LoRA
    assert re.search(
        r"self\.bc is not None and self\.support_quant_paths and not experts_lora",
        src), "BlockSparseMLP: BC single-expert path lost its runtime-LoRA guard"
    # bszN graph (v1.2.0: replaces the bsz-1 graph; may embed shared experts
    # + shared gate) skipped when the fused shared-expert linears carry a LoRA
    assert "sh_fused_lora" in src and re.search(
        r"elif bszn_eligible and not sh_fused_lora",
        src), "BlockSparseMLP: bszN graph lost its shared-experts LoRA guard"
    # raw-weight shared gate projection falls back to Linear.forward
    assert re.search(
        r"bsz > 32 or has_runtime_lora\(self\.shared_gate\)",
        src), "BlockSparseMLP: add_sigmoid_gate_proj lost its shared-gate LoRA guard"
    # CPU expert offload (v1.2.0) computes from a CPU copy of the base expert
    # weights with no GPU fallback available; must reject a routed-expert LoRA
    # loudly instead of silently bypassing it
    assert re.search(
        r"if self\.cpu_offload:\s*\n\s*if experts_lora:\s*\n\s*raise RuntimeError",
        src), "BlockSparseMLP: CPU expert offload lost its runtime-LoRA rejection"


def test_moe_expert_lora_slow_path_notice_present():
    # Expert adapters now apply via the unfused per-expert path; the loader
    # should still tell the user MoE decode will be slower while loaded.
    src = _src("exllamav3", "model", "lora.py")
    assert re.search(r"\\\.experts\\\.", src) and "per-expert" in src, \
        "lora.py: routed-expert slow-path notice removed"


def _stash_lora(*linears):
    stash = [(l, l.lora_a_tensors, l.lora_b_tensors) for l in linears if l is not None]
    for l, _, _ in stash:
        l.lora_a_tensors = {}
        l.lora_b_tensors = {}
    return stash


def _restore_lora(stash):
    for l, a, b in stash:
        l.lora_a_tensors = a
        l.lora_b_tensors = b


def test_mgemm_lora_delta_parity_gpu():
    """
    End-to-end numeric check that the delta-on-top mgemm path produces the same
    LoRA contribution as the per-linear path. Needs a quantized model and a
    trained adapter: set EXL3_TEST_MODEL and EXL3_TEST_ADAPTER.
    """
    model_dir = os.environ.get("EXL3_TEST_MODEL")
    adapter_dir = os.environ.get("EXL3_TEST_ADAPTER")
    if not model_dir or not adapter_dir:
        pytest.skip("set EXL3_TEST_MODEL and EXL3_TEST_ADAPTER to run")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from exllamav3 import Config, Model, Cache
    from exllamav3.model.lora import LoRA
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.mlp import GatedMLP

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 2048)  # noqa: F841 (must precede load)
    model.load(device = "cuda:0")
    lora = LoRA.from_directory(model, adapter_dir)

    def deltas(fused_fn, perlin_fn, *linears):
        with torch.inference_mode():
            with_lora_f = fused_fn()
            stash = _stash_lora(*linears)
            base_f = fused_fn()
            _restore_lora(stash)
            with_lora_p = perlin_fn()
            stash = _stash_lora(*linears)
            base_p = perlin_fn()
            _restore_lora(stash)
        return [
            ((wf - bf).float(), (wp - bp).float(), bf.float(), bp.float())
            for wf, bf, wp, bp in zip(with_lora_f, base_f, with_lora_p, base_p)
            if wf is not None
        ]

    def check(name, quads, min_delta = 1e-4):
        for df, dp, bf, bp in quads:
            # base parity between kernels is approximate; the LoRA delta itself
            # must match tightly (identical x @ a @ b on both paths)
            assert torch.allclose(bf, bp, atol = 0.05, rtol = 0.05), \
                f"{name}: fused/per-linear BASE outputs diverge"
            assert dp.abs().max() > min_delta, \
                f"{name}: adapter contributes no delta here (vacuous test)"
            assert torch.allclose(df, dp, atol = 0.02, rtol = 0.05), \
                f"{name}: LoRA delta differs between fused and per-linear paths " \
                f"(max diff {(df - dp).abs().max().item():.4g})"

    # --- Attention k/v (and q/g when built) ---
    attn = next(
        (m for m in model if isinstance(m, Attention) and m.multi_kv is not None
         and (m.k_proj.lora_a_tensors or m.v_proj.lora_a_tensors)),
        None)
    assert attn is not None, "no Attention with multi_kv + k/v LoRA in this model/adapter"
    x = (torch.randn(1, 8, attn.hidden_size, device = attn.device) * 0.05).half()

    def qkv_fused():
        q, k, v, g = attn.project_qkv(x, {})
        return k.reshape(-1), v.reshape(-1)

    def qkv_perlin():
        mkv, mqg = attn.multi_kv, attn.multi_qg
        attn.multi_kv = None
        attn.multi_qg = None
        try:
            q, k, v, g = attn.project_qkv(x, {})
        finally:
            attn.multi_kv, attn.multi_qg = mkv, mqg
        return k.reshape(-1), v.reshape(-1)

    check("attn k/v", deltas(qkv_fused, qkv_perlin,
                             attn.q_proj, attn.g_proj, attn.k_proj, attn.v_proj))

    # --- GatedMLP gate/up (delta lands pre-activation; compare full output) ---
    gmlp = next(
        (m for m in model if isinstance(m, GatedMLP) and m.num_slices > 0
         and m.multi_gu[0] is not None
         and (m.gates[0].lora_a_tensors or m.ups[0].lora_a_tensors)),
        None)
    assert gmlp is not None, "no GatedMLP with multi_gu + gate/up LoRA in this model/adapter"
    xm = (torch.randn(1, 8, gmlp.hidden_size, device = gmlp.gates[0].device) * 0.05).half()

    def mlp_fused():
        return (gmlp.forward(xm, {}).reshape(-1),)

    def mlp_perlin():
        mgu = list(gmlp.multi_gu)
        for i in range(gmlp.num_slices):
            gmlp.multi_gu[i] = None
        try:
            return (gmlp.forward(xm, {}).reshape(-1),)
        finally:
            for i, m in enumerate(mgu):
                gmlp.multi_gu[i] = m

    check("mlp", deltas(mlp_fused, mlp_perlin,
                        *gmlp.gates, *gmlp.ups, *gmlp.downs))

    lora.unload()


def test_has_runtime_lora_semantics():
    torch = pytest.importorskip("torch")  # noqa: F841  (linear.py imports it)
    try:
        from exllamav3.modules.linear import has_runtime_lora
    except ImportError as e:
        pytest.skip(f"exllamav3 not importable here: {e}")

    class Stub:
        def __init__(self, loaded: bool):
            self.lora_a_tensors = {object(): object()} if loaded else {}

    assert not has_runtime_lora()
    assert not has_runtime_lora(None)
    assert not has_runtime_lora(Stub(False), None, Stub(False))
    assert has_runtime_lora(Stub(True))
    assert has_runtime_lora(Stub(False), Stub(True))
    assert has_runtime_lora(None, Stub(True))


@pytest.mark.parametrize("bsz,q_len", [(1, 1), (2, 3)])
@pytest.mark.parametrize("fuse_qg,fuse_kv", [(True, True), (True, False), (False, True)])
def test_sliding_attention_padded_mgemm_lora(bsz, q_len, fuse_qg, fuse_kv):
    """Padded kernel inputs must still apply adapters to the original channels."""
    import ast
    from types import SimpleNamespace
    import torch

    # Execute the real method with a CPU replacement for the base-weight kernel.
    # This isolates adapter/input geometry from CUDA and model loading.
    tree = ast.parse(_src("exllamav3", "modules", "sliding_attn.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SlidingAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "project_qkv")
    finish = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "finish_qkv")
    dim, padded, width = 6, 8, 4
    torch.manual_seed(17)
    x = torch.randn(bsz, q_len, dim, dtype=torch.float16)

    class Projection:
        in_features = padded

        def __init__(self):
            self.a = torch.randn(dim, 2, dtype=torch.float16)
            self.b = torch.randn(2, width, dtype=torch.float16)

        def apply_lora(self, inputs, output):
            assert inputs.shape == (bsz * q_len, dim)
            output.add_((inputs @ self.a) @ self.b)

        def forward(self, inputs, params):
            return ((inputs[..., :dim] @ self.a) @ self.b).reshape(bsz, q_len, width)

    def mgemm(inputs, ptrs, output, *args):
        assert inputs.shape == (1, bsz * q_len, padded)
        assert torch.count_nonzero(inputs[..., dim:]) == 0
        output.zero_()

    ns = {"torch": torch, "ext": SimpleNamespace(exl3_mgemm=mgemm),
          "has_runtime_lora": lambda *args: True}
    exec(compile(ast.Module(body=[method, finish], type_ignores=[]), "sliding_attn.py", "exec"), ns)
    multi = SimpleNamespace(ptrs_trellis=None, ptrs_suh=None, ptrs_svh=None, K=4, mcg=False, mul1=False)
    layer = SimpleNamespace(
        q_proj=Projection(), g_proj=Projection(), k_proj=Projection(), v_proj=Projection(),
        multi_qg=multi if fuse_qg else None, multi_kv=multi if fuse_kv else None,
        # The sliced bundle is present but the stub has no project_qkv_sliced:
        # with a runtime LoRA loaded the dispatch must not reach it
        multi_qkv=multi,
        num_q_heads=1, num_kv_heads=1, head_dim=width, v_norm=None,
        prealloc_qgh_1=torch.empty(2, 1, padded, dtype=torch.float16),
        prealloc_kvh_1=torch.empty(2, 1, padded, dtype=torch.float16),
        prealloc_qg_1=torch.empty(2, 1, width, dtype=torch.float16),
        prealloc_kv_1=torch.empty(2, 1, width, dtype=torch.float16),
    )
    layer.finish_qkv = lambda *args: ns["finish_qkv"](layer, *args)
    actual = ns["project_qkv"](layer, x, {})
    for result, proj in zip(actual, [layer.q_proj, layer.k_proj, layer.v_proj, layer.g_proj]):
        torch.testing.assert_close(result.reshape(bsz, q_len, width), proj.forward(x, {}))
