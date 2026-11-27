"""
RowTable under tensor parallelism: the table never travels, RowTable.export() carries the shard locations and the
importing rank rebuilds the table with RowTable.from_export(). With ram / lock (--ngram_ram / --ngram_lock,
forwarded by NGramEmbedding.tp_export) the importing rank reads the table into its own RAM, and locks it.

Contracts (bitwise, CPU only):

- from_export() without ram / lock keeps disk handles (streams rows), as before.
- from_export(ram = True) holds every shard as a CPU tensor with the stored dtype and shape, equal to the stored
  shard; fetch() returns the stored rows for any row ids, across shard boundaries and the short last shard.
- from_export(lock = True) does the same and mlock()s the shards; close() releases them.
- The same holds for a sharded table whose shards sit back to back in one file (open() merges them into one
  handle) and for a table split over two files.
"""

import pytest
import torch

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules.row_table import RowTable

from testlib.checkpoint import write_tensors

pytestmark = pytest.mark.nogpu

KEY = "model.layers.0.ple.ngram_embedding"
ROWS = (96, 96, 40)      # rows per shard, last one short
WIDTH = 24


def shards(dtype, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for n in ROWS:
        if dtype.is_floating_point:
            t = torch.randn((n, WIDTH), generator = gen).to(dtype)
        else:
            t = torch.randint(-32768, 32768, (n, WIDTH), generator = gen, dtype = torch.int32).to(dtype)
        out.append(t)
    return out


def make_table(tmp_path, dtype, split_files):
    """Write the shards (one file, or the last shard in a second file), open the table streamed from disk as a
    loading rank would, and return (exported, reference shards)"""
    ref = shards(dtype)
    tensors = {f"{KEY}.shard_{i}.weight": t for i, t in enumerate(ref)}
    directory = str(tmp_path / "model")
    if split_files:
        last = f"{KEY}.shard_{len(ref) - 1}.weight"
        write_tensors({k: v for k, v in tensors.items() if k != last}, directory, "model-00001-of-00002.safetensors")
        write_tensors({last: tensors[last]}, directory, "model-00002-of-00002.safetensors")
    else:
        write_tensors(tensors, directory)
    stc = SafetensorsCollection(directory)
    table = RowTable.find(stc, KEY, "weight")
    assert table is not None
    table.open(stc, True)
    assert table.on_disk
    exported = table.export(stc)
    table.close()
    stc.close()
    return exported, ref


def row_ids():
    total = sum(ROWS)
    b = ROWS[0]
    ids = [0, 1, b - 1, b, b + 1, 2 * b - 1, 2 * b, total - 1, total - 2, 5, b + 7, 2 * b + 3, 0, total - 1]
    gen = torch.Generator().manual_seed(1)
    ids += torch.randint(0, total, (200,), generator = gen).tolist()
    return torch.tensor(ids, dtype = torch.int64)


def check_rows(table, ref):
    want = torch.cat(ref)
    ids = row_ids()
    got = table.fetch(ids)
    assert got.dtype == want.dtype
    assert torch.equal(got.view(-1).view(torch.uint8), want[ids].contiguous().view(-1).view(torch.uint8))


DTYPES = (torch.float16, torch.bfloat16, torch.int16)


@pytest.mark.parametrize("split_files", (False, True), ids = ("one_file", "two_files"))
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
def test_from_export_streams_by_default(tmp_path, dtype, split_files):
    exported, ref = make_table(tmp_path, dtype, split_files)
    table = RowTable.from_export(exported)
    assert table.on_disk
    check_rows(table, ref)
    table.close()


@pytest.mark.parametrize("split_files", (False, True), ids = ("one_file", "two_files"))
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
def test_from_export_ram(tmp_path, dtype, split_files):
    exported, ref = make_table(tmp_path, dtype, split_files)
    table = RowTable.from_export(exported, ram = True)
    assert not table.on_disk
    assert all(isinstance(s, torch.Tensor) and s.device.type == "cpu" for s in table.stores)
    assert table.num_rows == sum(ROWS) and table.row_dtype == dtype
    # Same rows as stored, whatever the shard layout of the stores (merged handle or one store per shard)
    assert torch.equal(torch.cat([s.reshape(-1, WIDTH) for s in table.stores]).view(torch.uint8),
                       torch.cat(ref).view(torch.uint8))
    check_rows(table, ref)
    table.close()


def lock_possible(nbytes: int) -> bool:
    """The loader's own pre-check (it keeps headroom on top of the table size)"""
    from exllamav3.util.memory import prepare_host_lock
    try:
        prepare_host_lock(nbytes, "test")
        return True
    except RuntimeError:
        return False


@pytest.mark.parametrize("split_files", (False, True), ids = ("one_file", "two_files"))
def test_from_export_lock(tmp_path, split_files):
    dtype = torch.float16
    if not lock_possible(sum(ROWS) * WIDTH * dtype.itemsize):
        pytest.skip("RLIMIT_MEMLOCK too low to lock host memory (needs the loader's headroom)")
    exported, ref = make_table(tmp_path, dtype, split_files)
    table = RowTable.from_export(exported, lock = True)
    assert not table.on_disk
    assert table._locked, "lock = True must mlock() the stores"
    check_rows(table, ref)
    table.close()
    assert not table._locked
