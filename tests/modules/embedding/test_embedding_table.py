"""
Token embedding tables behind a RowTable: the generalized row codec and dequant kernel (n-gram rows and rotated
256-wide groups), quantization of a table, and the Embedding module's storage forms (resident, streamed from
disk, quantized in RAM or streamed), which must all return what the table holds. References: the codec's bitwise
definition, the torch codec (dequant_rows) and the source table.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import Embedding
from exllamav3.modules.embedding import TableEmbedding
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
from exllamav3.modules.quant.exl3_lib.ngram_codec import (
    ROW_DIM, GROUP_DIM, words_per_row, mul1_codebook, pack_rows, unpack_rows, dequant_rows, hadamard_rotate,
)
from exllamav3.conversion.ngram import quantize_embedding
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from exllamav3.util.hadamard import get_hadamard_dt

from testlib.checkpoint import write_tensors
from testlib.env import get_test_device

DEV = get_test_device()
KEY = "model.embed_tokens"
VOCAB = 3000

# Relative error of a rotated group per bitrate, with margin. The trellis lands within a fraction of a dB of the
# rate-distortion bound of a Gaussian source, 2^-K
MAX_ERROR = {K: 1.2 * 2.0 ** -K for K in range(1, 9)}


@pytest.fixture(scope = "module", autouse = True)
def _inference_mode():
    with torch.inference_mode():
        yield


def unpack_rows_bitwise(packed, K):
    """The codec's definition, bit by bit: state_i bit m lives at stream bit ((i - m // K) mod dim) * K + m % K"""
    dim = (packed.shape[1] - 1) * 16 // K
    words = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    stream = ((words.unsqueeze(-1) >> torch.arange(16)) & 1).reshape(packed.shape[0], dim * K)
    i = torch.arange(dim).unsqueeze(1)
    m = torch.arange(16).unsqueeze(0)
    return (stream[:, ((i - m // K) % dim) * K + m % K] << m).sum(dim = -1)


def rand_packed(n, K, dim, gen):
    packed = torch.randint(0, 65536, (n, words_per_row(K, dim)), dtype = torch.int32, generator = gen).to(torch.int16)
    packed[:, 0] = (torch.rand((n,), generator = gen) * 0.1 + 0.01).half().view(torch.int16)
    return packed


def rand_table(hidden, gen, dtype = torch.float16):
    """Embedding-like table: heavy-tailed elements, a shared offset, a few large and a few all-zero rows"""
    w = torch.randn((VOCAB, hidden), generator = gen) * 0.02
    w = w * (1.0 + 4.0 * (torch.rand((VOCAB, hidden), generator = gen) < 0.01))
    w = w + torch.randn((hidden,), generator = gen) * 0.01
    w[10:20] *= 25.0
    w[100:110] = 0.0
    return w.to(dtype)


def make_model(directory, tensors):
    write_tensors(tensors, directory)
    return directory


def load_embedding(directory, hidden, device, stream, **kwargs):
    stc = SafetensorsCollection(directory)
    config = SimpleNamespace(stc = stc, infer_params = SimpleNamespace(embed_stream_from_disk = stream))
    module = Embedding(config, KEY, VOCAB, hidden, out_dtype = torch.half, **kwargs)
    load_device = torch.device("cpu") if module.caps.get("prefer_cpu") else device
    module.load(load_device)
    return module


def reference(tensors, K, hidden):
    """The quantized table decoded by the torch codec"""
    trellis, signs = tensors[f"{KEY}.trellis"], tensors[f"{KEY}.signs"]
    groups = signs.shape[0]
    rings = trellis.view(VOCAB * groups, -1).to(DEV)
    out = dequant_rows(rings, K, mul1_codebook(DEV), signs = signs.to(DEV).repeat(VOCAB, 1))
    return out.view(VOCAB, -1)[:, :hidden].half().cpu()


def ids_batches(gen):
    yield torch.tensor([[17]])
    yield torch.tensor([[VOCAB - 1]])
    yield torch.randint(0, VOCAB, (4, 1), generator = gen)
    yield torch.randint(0, VOCAB, (1, 300), generator = gen)
    yield torch.randint(0, 40, (3, 700), generator = gen)             # heavy repetition
    yield torch.arange(VOCAB).view(1, -1)


@pytest.mark.nogpu
@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("dim", (ROW_DIM, GROUP_DIM))
def test_unpack_matches_definition(K, dim):
    gen = torch.Generator().manual_seed(K * 1000 + dim)
    packed = rand_packed(64, K, dim, gen)
    states, scales = unpack_rows(packed, K)
    assert states.shape == (64, dim)
    assert torch.equal(states, unpack_rows_bitwise(packed, K))
    assert torch.equal(scales.view(torch.int16), packed[:, 0])


@pytest.mark.parametrize("K", (1, 3, 4, 8))
@pytest.mark.parametrize("dim", (ROW_DIM, GROUP_DIM))
def test_pack_roundtrip(K, dim):
    gen = torch.Generator().manual_seed(K + dim)
    tiles = torch.randn((48, dim), generator = gen).to(DEV)
    q, states = quantize_tiles(tiles.contiguous(), {"K": K, "mul1": True})
    scales = (torch.rand((48,), generator = gen) + 0.5).half().to(DEV)
    packed = pack_rows(states, scales, K)
    assert packed.shape == (48, words_per_row(K, dim))
    states2, scales2 = unpack_rows(packed, K)
    assert torch.equal(states2, states.to(torch.int64) & 0xFFFF)
    assert torch.equal(scales2, scales)
    # The decoded ring is the quantizer's reconstruction
    deq = dequant_rows(packed, K, mul1_codebook(DEV))
    assert torch.allclose(deq, q * scales.float().unsqueeze(1), rtol = 2e-3, atol = 1e-6)


@pytest.mark.nogpu
def test_hadamard_rotate():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn((9, GROUP_DIM), generator = gen, dtype = torch.float64)
    H = get_hadamard_dt(GROUP_DIM, "cpu", torch.float64, GROUP_DIM ** -0.5)
    assert torch.allclose(hadamard_rotate(x), x @ H, atol = 1e-12)
    assert torch.allclose(hadamard_rotate(hadamard_rotate(x)), x, atol = 1e-12)
    assert hadamard_rotate(x.view(3, 3, GROUP_DIM)).shape == (3, 3, GROUP_DIM)


@pytest.mark.parametrize("K", range(1, 9))
def test_kernel_ngram_rows(K):
    gen = torch.Generator().manual_seed(K)
    n, heads = 301, 16
    packed = rand_packed(n, K, ROW_DIM, gen).to(DEV)
    bias = (torch.randn((heads, ROW_DIM), generator = gen) * 0.01).half().to(DEV)
    head = torch.randint(0, heads, (n,), generator = gen, dtype = torch.int32).to(DEV)
    out = torch.empty((n, ROW_DIM), dtype = torch.half, device = DEV)
    ext.ngram_dequant(packed, K, head, bias, out, False)
    ref = dequant_rows(packed, K, mul1_codebook(DEV), bias.float()[head.long()])
    assert torch.equal(out, ref.half())
    # No bias
    ext.ngram_dequant(packed, K, None, None, out, False)
    assert torch.equal(out, dequant_rows(packed, K, mul1_codebook(DEV)).half())


@pytest.mark.parametrize("K", range(1, 9))
def test_kernel_rotated_groups(K):
    gen = torch.Generator().manual_seed(100 + K)
    groups, rows = 5, 77
    packed = rand_packed(rows * groups, K, GROUP_DIM, gen).to(DEV)
    signs = (torch.randint(0, 2, (groups, GROUP_DIM), generator = gen) * 2 - 1).half().to(DEV)
    out = torch.empty((rows * groups, GROUP_DIM), dtype = torch.half, device = DEV)
    ext.ngram_dequant(packed, K, None, signs, out, True)
    ref = dequant_rows(packed, K, mul1_codebook(DEV), signs = signs.repeat(rows, 1))
    assert torch.equal(out, ref.half())
    # Explicit sign rows instead of ring index mod groups
    head = torch.randint(0, groups, (rows * groups,), generator = gen, dtype = torch.int32).to(DEV)
    ext.ngram_dequant(packed, K, head, signs, out, True)
    ref = dequant_rows(packed, K, mul1_codebook(DEV), signs = signs[head.long()])
    assert torch.equal(out, ref.half())


@pytest.mark.parametrize("hidden", (640, 1024))
@pytest.mark.parametrize("K", (2, 4, 6, 8))
def test_quantize_embedding(hidden, K):
    gen = torch.Generator().manual_seed(hidden + K)
    weight = rand_table(hidden, gen)
    tensors, rfn = quantize_embedding(KEY, weight, K, DEV, chunk_rows = 1000, verbose = False)
    groups = (hidden + GROUP_DIM - 1) // GROUP_DIM
    assert tensors[f"{KEY}.trellis"].shape == (VOCAB, groups * (1 + 16 * K))
    assert tensors[f"{KEY}.trellis"].dtype == torch.int16
    assert tensors[f"{KEY}.signs"].shape == (groups, GROUP_DIM)
    ref = reference(tensors, K, hidden).float()
    w = weight.float()
    err = (ref - w).norm() / w.norm()
    assert abs(err.item() - rfn) < 1e-3
    assert err < MAX_ERROR[K], err
    # Every row on its own, the large ones included; all-zero rows decode to zero
    live = w.norm(dim = 1) > 0
    row_err = (ref - w).norm(dim = 1)[live] / w.norm(dim = 1)[live]
    assert row_err.max() < 1.25 * MAX_ERROR[K], row_err.max()
    assert ref[~live].abs().max() == 0
    # The same table every time
    again, _ = quantize_embedding(KEY, weight, K, DEV, chunk_rows = 777, verbose = False)
    assert all(torch.equal(tensors[k], again[k]) for k in tensors)


@pytest.mark.parametrize("hidden", (640, 1024))
@pytest.mark.parametrize("stream", (False, True))
def test_quantized_table_lookup(tmp_path, hidden, stream):
    K = 4
    gen = torch.Generator().manual_seed(hidden)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), K, DEV, verbose = False)
    ref = reference(tensors, K, hidden)
    module = load_embedding(make_model(str(tmp_path / "q"), tensors), hidden, DEV, stream)
    assert isinstance(module.embedding, TableEmbedding) and module.embedding.table.on_disk == stream
    assert module.device == DEV and not module.caps["prefer_cpu"] and module.caps["x_cpu"]
    for ids in ids_batches(gen):
        x = module.prepare_for_device(ids, {})
        assert x.device.type == "cpu"
        params = {}
        out = module.forward(x, params)
        assert out.device == DEV and out.dtype == torch.half and out.shape == (*ids.shape, hidden)
        assert torch.equal(out.cpu(), ref[ids])
        assert params["input_ids"] is x
    # Tensors for a recompile: what the file holds
    if not stream:
        out_tensors = module.get_tensors()
        assert set(out_tensors) == set(tensors)
        assert all(torch.equal(out_tensors[k].cpu(), tensors[k]) for k in tensors)
    module.unload()
    assert module.embedding is None


def test_quantized_table_cpu_device(tmp_path):
    """A module placed on the CPU (the converter's state advance) decodes with the torch codec"""
    K, hidden = 3, 640
    gen = torch.Generator().manual_seed(8)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), K, DEV, verbose = False)
    ref = reference(tensors, K, hidden)
    module = load_embedding(make_model(str(tmp_path / "q"), tensors), hidden, torch.device("cpu"), False)
    for ids in ids_batches(gen):
        out = module.forward(ids, {})
        assert out.device.type == "cpu"
        assert torch.equal(out, ref[ids])


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_streamed_unquantized_table(tmp_path, dtype):
    hidden = 640
    gen = torch.Generator().manual_seed(21)
    weight = rand_table(hidden, gen, dtype)
    directory = make_model(str(tmp_path / "w"), {f"{KEY}.weight": weight})
    resident = load_embedding(directory, hidden, DEV, False)
    streamed = load_embedding(directory, hidden, DEV, True)
    assert isinstance(resident.embedding, torch.nn.Embedding)
    assert isinstance(streamed.embedding, TableEmbedding) and streamed.embedding.table.on_disk
    assert streamed.device.type == "cpu"
    held = []
    for ids in ids_batches(gen):
        a = resident.forward(ids, {})
        b = streamed.forward(ids, {})
        assert a.dtype == b.dtype and torch.equal(a, b)
        assert torch.equal(a.float(), weight.float()[ids].half().float())
        held.append((a, b))
    # Outputs stay what they were once later lookups have reused the staging buffers
    assert all(torch.equal(a, b) for a, b in held)
    # A table read by other code stays resident whatever the option says
    pinned = load_embedding(directory, hidden, DEV, True, allow_table = False)
    assert isinstance(pinned.embedding, torch.nn.Embedding)


def test_multiplier_and_normalize(tmp_path):
    hidden = 640
    gen = torch.Generator().manual_seed(4)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), 4, DEV, verbose = False)
    ref = reference(tensors, 4, hidden)
    directory = make_model(str(tmp_path / "q"), tensors)
    module = load_embedding(directory, hidden, DEV, False, normalize = True, multiplier = 0.5)
    ids = torch.randint(0, VOCAB, (2, 50), generator = gen)
    exp = ref[ids].to(DEV)
    exp *= 0.5
    exp *= hidden ** 0.5
    assert torch.equal(module.forward(ids, {}), exp)
    # The table itself is untouched by the in-place scaling of the output
    assert torch.equal(module.forward(ids, {}), exp)


def test_indexed_embeddings(tmp_path):
    """Multimodal embeddings spliced into the token embeddings of a table on the device"""
    hidden = 640
    gen = torch.Generator().manual_seed(6)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), 4, DEV, verbose = False)
    ref = reference(tensors, 4, hidden)
    module = load_embedding(make_model(str(tmp_path / "q"), tensors), hidden, DEV, True)
    mm = SimpleNamespace(
        first_index = FIRST_MM_EMBEDDING_INDEX + 100, mm_length = 12, deepstack_embeddings = None,
        embeddings = torch.randn((12, hidden), generator = gen).half())
    ids = torch.randint(0, VOCAB, (2, 40), generator = gen)
    ids[0, 5:17] = torch.arange(12) + mm.first_index
    ids[1, 20:26] = torch.arange(6) + mm.first_index
    out = module.forward(ids, {"indexed_embeddings": [mm]}).cpu()
    exp = ref[ids.clamp(max = VOCAB - 1)]
    exp[0, 5:17] = mm.embeddings
    exp[1, 20:26] = mm.embeddings[:6]
    assert torch.equal(out, exp)
    # The same batch without any multimodal ids takes the plain path
    plain = torch.randint(0, VOCAB, (2, 40), generator = gen)
    assert torch.equal(module.forward(plain, {"indexed_embeddings": [mm]}).cpu(), ref[plain])


@pytest.mark.parametrize("stream", (False, True))
def test_row_out_of_range(tmp_path, stream):
    hidden = 640
    gen = torch.Generator().manual_seed(2)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), 4, DEV, verbose = False)
    module = load_embedding(make_model(str(tmp_path / "q"), tensors), hidden, DEV, stream)
    for bad in (torch.tensor([[VOCAB]]), torch.tensor([[1, 2, VOCAB + 5]]), torch.tensor([[-1, 4]])):
        with pytest.raises((AssertionError, IndexError)):
            module.forward(bad, {})
    # and the table still serves lookups afterwards
    assert module.forward(torch.tensor([[1, 2]]), {}).shape == (1, 2, hidden)


def test_prefetch_tokens(tmp_path):
    """The read-ahead hint changes nothing but timing, in every storage form"""
    hidden = 640
    gen = torch.Generator().manual_seed(12)
    weight = rand_table(hidden, gen)
    tensors, _ = quantize_embedding(KEY, weight, 4, DEV, verbose = False)
    ref = reference(tensors, 4, hidden)
    q = make_model(str(tmp_path / "q"), tensors)
    w = make_model(str(tmp_path / "w"), {f"{KEY}.weight": weight})
    for directory, stream, exp in ((q, True, ref), (q, False, ref), (w, True, weight), (w, False, weight)):
        module = load_embedding(directory, hidden, DEV, stream)
        for _ in range(20):
            ids = torch.randint(0, VOCAB, (1, 1), generator = gen)
            # ids past the table are multimodal placeholders, which have no row
            module.prefetch_tokens([ids.item(), FIRST_MM_EMBEDDING_INDEX + 3])
            assert torch.equal(module.forward(ids, {}).cpu(), exp[ids])
        module.unload()


def test_table_prefetch_worker(tmp_path):
    """
    Chunk-sized lookups staged ahead on the worker thread return what the inline path returns. Forwards are
    issued back to back behind a long kernel, so each chunk's uploads are still pending when the worker two
    chunks later takes the same staging set: it has to wait for them.
    """
    hidden = 1024
    gen = torch.Generator().manual_seed(13)
    tensors, _ = quantize_embedding(KEY, rand_table(hidden, gen), 4, DEV, verbose = False)
    ref = reference(tensors, 4, hidden)
    module = load_embedding(make_model(str(tmp_path / "q"), tensors), hidden, DEV, True)
    emb = module.embedding
    chunks = [torch.randint(0, VOCAB, (1, n), generator = gen) for n in (512, 2048, 300, 1024, 2000, 700, 1500, 256, 2048, 900)]
    spin = torch.empty((12288, 12288), dtype = torch.float, device = DEV)
    outs = []
    emb.table.prefetch(chunks[0], chunks[0].numel(), emb.resolve)
    for i, ids in enumerate(chunks):
        if i + 1 < len(chunks):
            emb.table.prefetch(chunks[i + 1], chunks[i + 1].numel(), emb.resolve)
        torch.matmul(spin, spin)
        outs.append(module.forward(ids, {}))
    torch.cuda.synchronize(DEV)
    assert emb.table.prefetch_stats == {"hit": len(chunks), "miss": 0, "retired": 0}
    for ids, out in zip(chunks, outs):
        assert torch.equal(out.cpu(), ref[ids])
