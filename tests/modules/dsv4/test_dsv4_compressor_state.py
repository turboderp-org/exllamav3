"""
DeepSeek-V4 compressor state: the chunked (stateful) compressor must reproduce the single-shot result -
sub-window buffering, Ca-overlap carry and entry positioning included - and the stateless forward must equal the
stateful-from-empty one on the shared prefix of complete windows. Checked on the CSA compressor, the CSA indexer
compressor and the HCA compressor of the tiny random checkpoint.
"""

import pytest
import torch

from exllamav3.modules.dsv4 import DSV4CompressorState

from testlib.dsv4 import attention_layers, build_tiny_dsv4, chunk_splits
from testlib.tiny_models import DSV4_TINY

# The wkv/wgate projections are fp16 GEMMs whose tiling depends on the row count, so chunked and whole runs see
# ulp-level projection differences (the mechanism documented for the MLA chunked-vs-whole tests). The buffering
# logic itself is exact: small-seq cases where cuBLAS uses a single tile compare bitwise
TOL = 2e-3


@pytest.fixture(scope = "module")
def attn_layers(tmp_path_factory, device):
    config, model = build_tiny_dsv4(tmp_path_factory.mktemp("dsv4_state"), seed = 11)
    model.load(str(device))
    csa, hca = attention_layers(model, "csa"), attention_layers(model, "hca")
    assert csa and hca, "tiny checkpoint has no CSA or no HCA layer"
    yield {"csa": csa[0], "hca": hca[0]}
    model.unload()


@pytest.mark.parametrize("site", ["csa.compressor", "csa.indexer", "hca.compressor"])
@pytest.mark.parametrize("seq, pattern", [
    pytest.param(315, [100, 107, 108], id = "uneven"),          # non-aligned ends
    pytest.param(313, [1] * 9 + [304], id = "decode_then_bulk"),  # single-token steps then bulk
    pytest.param(64, [3, 5, 7, 49], id = "sub_window"),          # the buffer must carry
    pytest.param(7, [2, 2, 3], id = "shorter_than_window"),     # shorter than the HCA window entirely
])
def test_chunked_compressor(attn_layers, device, site, seq, pattern):
    layer_type, comp_name = site.split(".")
    attn = attn_layers[layer_type]
    comp = getattr(attn, comp_name)
    inv_freq = attn.inv_freq_compress
    torch.manual_seed(seq)
    x = torch.randn((1, seq, DSV4_TINY["hidden_size"]), dtype = torch.half, device = device)

    with torch.inference_mode():
        whole = comp.forward(x, {}, inv_freq, DSV4CompressorState())
        chunk_state = DSV4CompressorState()
        parts = []
        for a, b in chunk_splits(seq, pattern):
            e = comp.forward(x[:, a:b], {}, inv_freq, chunk_state)
            if e.shape[1]:
                parts.append(e)
        chunked = torch.cat(parts, dim = 1) if parts else whole[:, :0]
        stateless = comp.forward(x, {}, inv_freq, None)

    assert whole.shape == chunked.shape, f"shape {whole.shape} vs chunked {chunked.shape}"
    err = (whole.float() - chunked.float()).abs().max().item() if whole.numel() else 0.0
    assert err < TOL, f"chunked vs whole maxerr {err:.2e} over {whole.shape[1]} entries"
    n = stateless.shape[1]
    err2 = (whole[:, :n].float() - stateless.float()).abs().max().item() if n else 0.0
    assert err2 < TOL, f"stateless vs stateful prefix maxerr {err2:.2e}"
    assert chunk_state.entry_count == whole.shape[1]
