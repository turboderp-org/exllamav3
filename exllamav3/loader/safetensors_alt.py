from __future__ import annotations
import io
import json
import os
import struct
from dataclasses import dataclass
import torch

_ST_DTYPE_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_TORCH_DTYPE_TO_ST = {v: k for k, v in _ST_DTYPE_TO_TORCH.items()}

_U64_LE = struct.Struct("<Q")


def _prod(shape) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return int(p)


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _tensor_bytes_view(t: torch.Tensor) -> memoryview:
    """
    Return a writable/readable memoryview over t's storage (CPU tensor).
    """
    assert t.device.type == "cpu", "Tensor must be CPU"
    assert t.is_contiguous(), "Tensor must be contiguous"

    try:
        us = t.untyped_storage()
        mv = memoryview(us).cast("B")
        byte_off = int(t.storage_offset() * t.element_size())
        nbytes = _tensor_nbytes(t)
        return mv[byte_off: byte_off + nbytes]
    except TypeError:
        if t.dtype is torch.bfloat16:
            arr_u8 = t.view(torch.uint8).numpy()  # zero-copy
            return memoryview(arr_u8)             # already bytes
        else:
            arr = t.numpy()  # zero-copy view for CPU contiguous tensors
            return memoryview(arr).cast("B")


def _read_exact(f: io.BufferedReader, n: int) -> bytes:
    b = f.read(n)
    if b is None or len(b) != n:
        raise EOFError(f"Wanted {n} bytes, got {0 if b is None else len(b)}")
    return b


def _stream_readinto(f: io.BufferedReader, mv: memoryview, nbytes: int, chunk: int = 8 << 20) -> None:
    """
    Read exactly nbytes from f into mv (a bytes memoryview), in chunks.
    """
    if len(mv) < nbytes:
        raise ValueError("Destination buffer too small.")
    off = 0
    remaining = nbytes
    while remaining:
        step = chunk if remaining > chunk else remaining
        view = mv[off : off + step]
        got = f.readinto(view)
        if got is None or got == 0:
            raise EOFError("Unexpected EOF while streaming tensor bytes.")
        off += got
        remaining -= got


def _stream_write(f: io.BufferedWriter, mv: memoryview, chunk: int = 8 << 20) -> None:
    """
    Write the full memoryview mv to f in chunks.
    """
    off = 0
    n = len(mv)
    while off < n:
        step = chunk if (n - off) > chunk else (n - off)
        f.write(mv[off : off + step])
        off += step


@dataclass(frozen=True)
class _TensorInfo:
    dtype_tag: str
    shape: tuple
    begin: int  # relative to data section
    end: int    # relative to data section


class SafeOpen:
    """
    Context manager for reading safetensors without mmap.
    Provides .keys() and .get_tensor(name)->torch.Tensor (CPU).
    """

    def __init__(self, path: str):
        self.path = path
        self._f: io.BufferedReader | None = None
        self._infos: dict = {}
        self._data_start: int = 0
        self._file_size: int = 0
        self.metadata: dict = {}

    def __enter__(self) -> "SafeOpen":
        self._file_size = os.path.getsize(self.path)
        self._f = open(self.path, "rb", buffering = 0)

        # header length (8 bytes LE u64)
        header_len = _U64_LE.unpack(_read_exact(self._f, 8))[0]

        # header JSON
        header_bytes = _read_exact(self._f, int(header_len))
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid safetensors header JSON: {e}") from e

        # parse header entries
        self._data_start = 8 + int(header_len)

        # optional metadata
        md = header.get("__metadata__")
        if md is not None:
            if not isinstance(md, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in md.items()):
                raise ValueError("__metadata__ must be a string->string map.")
            self.metadata = dict(md)

        for k, v in header.items():
            if k == "__metadata__":
                continue
            if not isinstance(v, dict):
                raise ValueError(f"Tensor entry {k!r} must be an object.")
            dtype_tag = v.get("dtype")
            shape = v.get("shape")
            offsets = v.get("data_offsets")
            if dtype_tag not in _ST_DTYPE_TO_TORCH:
                raise ValueError(f"Unsupported dtype tag {dtype_tag!r} for tensor {k!r}.")
            if not (isinstance(shape, list) and all(isinstance(x, int) for x in shape)):
                raise ValueError(f"Invalid shape for tensor {k!r}.")
            if not (isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(x, int) for x in offsets)):
                raise ValueError(f"Invalid data_offsets for tensor {k!r}.")
            begin, end = int(offsets[0]), int(offsets[1])
            if begin < 0 or end < begin:
                raise ValueError(f"Invalid offsets for tensor {k!r}: {offsets}")

            # validate size matches shape*dtype
            torch_dtype = _ST_DTYPE_TO_TORCH[dtype_tag]
            expected = _prod(shape) * torch.tensor([], dtype=torch_dtype).element_size()
            if (end - begin) != expected:
                raise ValueError(
                    f"Size mismatch for {k!r}: offsets imply {end-begin} bytes, "
                    f"but shape/dtype imply {expected} bytes."
                )

            # validate within file
            abs_begin = self._data_start + begin
            abs_end = self._data_start + end
            if abs_end > self._file_size:
                raise ValueError(f"Tensor {k!r} out of bounds: needs up to {abs_end}, file is {self._file_size}")

            self._infos[k] = _TensorInfo(dtype_tag=dtype_tag, shape=tuple(shape), begin=begin, end=end)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def keys(self):
        return self._infos.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._f is None:
            raise RuntimeError("SafeOpen is closed.")
        info = self._infos.get(key)
        if info is None:
            raise KeyError(key)

        dtype = _ST_DTYPE_TO_TORCH[info.dtype_tag]
        t = torch.empty(info.shape, dtype=dtype, device="cpu")  # system RAM
        mv = _tensor_bytes_view(t)

        abs_pos = self._data_start + info.begin
        self._f.seek(abs_pos, io.SEEK_SET)
        nbytes = info.end - info.begin
        _stream_readinto(self._f, mv, nbytes)
        return t


def safe_open(path: str, framework = "pt", device = "cpu") -> SafeOpen:
    """
    Drop-in-ish replacement constructor:
        with safe_open("x.safetensors") as f:
            t = f.get_tensor("weight")
    """
    assert framework == "pt" and device == "cpu", \
        "Can only load CPU PyTorch tensors"
    return SafeOpen(path)


def save_file(
    tensors: dict,
    filename: str,
    metadata: dict | None = None,
) -> None:
    """
    Replacement for safetensors.torch.save_file:
      - streams tensor bytes directly to disk (no intermediate copies)
      - constructs compatible header with relative offsets
    """
    if metadata is not None:
        if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("metadata must be a dict[str,str] (all values must be strings).")

    # Validate + build entries with relative offsets into the data section
    header: dict = {}
    if metadata:
        header["__metadata__"] = dict(metadata)

    offset = 0
    infos: dict = {}
    for name, t in tensors.items():
        if not isinstance(name, str):
            raise ValueError("All tensor names must be strings.")
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Value for {name!r} is not a torch.Tensor.")
        st_dtype = _TORCH_DTYPE_TO_ST.get(t.dtype)
        if st_dtype is None:
            raise ValueError(f"Unsupported torch dtype for {name!r}: {t.dtype}")
        nbytes = _tensor_nbytes(t)
        begin, end = offset, offset + nbytes
        infos[name] = _TensorInfo(dtype_tag = st_dtype, shape = tuple(t.shape), begin = begin, end = end)
        offset = end

    # Fill header entries
    for name, info in infos.items():
        header[name] = {
            "dtype": info.dtype_tag,
            "shape": list(info.shape),
            "data_offsets": [info.begin, info.end],
        }

    # Encode header JSON (minified)
    header_bytes = json.dumps(header, separators = (",", ":"), ensure_ascii = False).encode("utf-8")
    header_len = len(header_bytes)

    # Write file: [u64 header_len][header_json][raw bytes...]
    with open(filename, "wb", buffering = 0) as f:
        f.write(_U64_LE.pack(header_len))
        f.write(header_bytes)

        # stream tensors in the same order we built offsets
        for name, t in tensors.items():
            t = t.contiguous().cpu()
            mv = _tensor_bytes_view(t)
            _stream_write(f, mv)
