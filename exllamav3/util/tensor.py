from __future__ import annotations
import torch
from .device_copy import to_device

class SeqTensor:

    PAGE_SIZE = 256

    tensor: torch.Tensor | None
    seq_dim: int
    seq_len: int
    seq_cap: int

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        seq_dim: int,
        device: torch.device = "cpu",
        init_cap: int = -1
    ):
        if seq_dim < 0: seq_dim = len(shape) + seq_dim
        self.seq_dim = seq_dim
        self.seq_len = 0
        if init_cap == -1:
            init_cap = self.PAGE_SIZE
        else:
            init_cap = (init_cap // self.PAGE_SIZE + 1) * self.PAGE_SIZE
        shape = list(shape)
        shape[seq_dim] = self.seq_cap = init_cap
        shape = tuple(shape)
        # Lazily allocate inner Tensor object to avoid committing too much virtual memory, which can crash the
        # process on Windows
        # self.tensor = torch.empty(shape, dtype = dtype, device = device)
        self.init_shape = shape
        self.dtype = dtype
        self.device = device
        self.tensor = None

    def __len__(self):
        return self.seq_len

    def __bool__(self):
        return self.seq_len > 0

    def _ensure_init(self):
        if self.tensor is None:
            self.tensor = torch.empty(self.init_shape, dtype = self.dtype, device = self.device)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, seq_dim: int):
        s = SeqTensor(tensor.shape, tensor.dtype, seq_dim, tensor.device, init_cap = tensor.shape[seq_dim])
        s.append(tensor)
        return s

    def clone(self, drop: int | None = None):
        if drop and drop <= self.seq_len:
            return SeqTensor.from_tensor(self.torch_slice(None, self.seq_len - drop), self.seq_dim)
        else:
            return SeqTensor.from_tensor(self.torch(), self.seq_dim)

    def clear(self):
        self.seq_len = 0

    def set(self, new_data: SeqTensor | torch.tensor | None = None):
        self.clear()
        self.append(new_data)

    def append(self, new_data: SeqTensor | torch.tensor | None):
        self._ensure_init()
        if new_data is None: return
        if isinstance(new_data, SeqTensor):
            new_data = new_data.torch()
        new_len = new_data.shape[self.seq_dim]
        end_pos = self.seq_len + new_len
        if end_pos >= self.seq_cap:
            new_cap = (end_pos // self.PAGE_SIZE + 1) * self.PAGE_SIZE
            grow_shape = list(new_data.shape)
            grow_shape[self.seq_dim] = new_cap - self.seq_cap
            grow_shape = tuple(grow_shape)
            grow_tensor = torch.empty(grow_shape, dtype = self.tensor.dtype, device = self.tensor.device)
            self.tensor = torch.cat((self.tensor, grow_tensor), dim = self.seq_dim)
            self.seq_cap = new_cap
        s = self.tensor.narrow(self.seq_dim, self.seq_len, end_pos - self.seq_len)
        s.copy_(new_data)
        self.seq_len += new_len

    def truncate(self, new_len: int):
        assert new_len <= self.seq_len
        self.seq_len = new_len

    def torch(self):
        self._ensure_init()
        s = self.tensor.narrow(self.seq_dim, 0, self.seq_len)
        return s

    def slice(self, a: int | None, b: int | None):
        return SeqTensor.from_tensor(self.torch_slice(a, b), self.seq_dim)

    def torch_slice(self, a: int | None, b: int | None):
        self._ensure_init()
        if a is None and b is None:
            return self.torch()
        elif b is None:
            s = self.tensor.narrow(self.seq_dim, a, self.seq_len - a)
        elif a is None:
            s = self.tensor.narrow(self.seq_dim, 0, b)
        else:
            s = self.tensor.narrow(self.seq_dim, a, b - a)
        return s


no_default = object()

def get_for_device(
    input_dict: dict,
    key: str | int,
    device: torch.device,
    default = no_default,
) -> torch.Tensor | None:
    """
    Read a tensor from a dict and ensure it is available on the specified device. Caches access per device and may
    break if the tensor is updated after being accessed in this way. Intended for tensors that are read-only for the
    lifetime of the dict, such as RoPE coefficients during a single forward pass.
    """
    if key not in input_dict and default is not no_default:
        return default

    v = input_dict[key]
    if v is None:
        return None

    cache = input_dict.get("dev_cache")
    if cache is None:
        cache = {}
        input_dict["dev_cache"] = cache

    # Key by tensor identity so the same tensor under two params keys (e.g. positions aliasing
    # cache_seqlens) uploads only once per device. The cache entry keeps the source tensor
    # alive so its id cannot be recycled within the lifetime of the dict
    cache_key = (id(v), device)
    hit = cache.get(cache_key)
    if hit is not None:
        return hit[1]

    # Tensors marked _static_dev_cache = True are never mutated in place and keep persistent
    # per-device copies (stored on the tensor itself) across forward passes
    if getattr(v, "_static_dev_cache", False):
        scache = v.__dict__.get("_static_dev_copies")
        if scache is None:
            scache = {}
            v._static_dev_copies = scache
        dv = scache.get(device)
        if dv is None:
            dv = to_device(v, device)
            scache[device] = dv
    else:
        # Pinned sources upload asynchronously: the copy is stream-ordered ahead of the kernels
        # that consume it, so the host never stalls. Callers that reuse pinned staging buffers
        # must not refill them until a sync point (the generator syncs every iteration when
        # collecting sampled tokens)
        nb = v.device.type == "cpu" and v.is_pinned()
        dv = to_device(v, device, non_blocking = nb)
    cache[cache_key] = (v, dv)
    return dv


buffered_aranges = {}
def buffered_arange(r: int, device: torch.device):
    if r not in buffered_aranges:
        buffered_aranges[r] = torch.arange(r)
    return get_for_device(buffered_aranges, r, device)

def buffered_interleaved_arange(r: int, k: int, device: torch.device):
    if (r, k) not in buffered_aranges:
        buffered_aranges[(r, k)] = torch.arange(r).repeat_interleave(k)
    return get_for_device(buffered_aranges, (r, k), device)

cpu_matmul_chunk = 256


def repack_64(weight: torch.Tensor) -> torch.Tensor:
    """
    Repack a [K, N] CPU weight into panel-major [NB, K, 64] (NB = ceil(N / 64)) for the
    native vision GEMM: each k-step then walks B sequentially. A strided walk over plain
    [K, N] stalls ~5x on L3 latency with this kernel's reuse pattern (measured), so the
    one-time repack (same dtype, same values, last panel zero-padded) pays for itself on
    the first forward. Callers cache the result keyed on the resident weight's identity.
    """
    assert weight.dim() == 2 and weight.device.type == "cpu"
    K, N = weight.shape
    NB = (N + 63) // 64
    out = torch.zeros((NB, K, 64), dtype = weight.dtype, device = weight.device)
    for g in range(NB):
        m = min(64, N - g * 64)
        if m:
            out[g, :, :m].copy_(weight[:, g * 64 : g * 64 + m])
    return out


def cpu_matmul(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype | None = None, bias: torch.Tensor | None = None, weight_blocked: torch.Tensor | None = None):
    """
    CPU projection primitive: x @ weight for 2D weight stored [in_features, out_features]
    (the repository's linear layout) and x with arbitrary leading dimensions.

    Half/bfloat16 inputs producing float output accumulate in FP32 over fixed-size chunks of
    the reduction dimension, converting only the current activation slice and matching weight
    rows, so no full float copy of the weight is ever materialized. Matching dtypes use plain
    matmul.

    The `x2.to(weight.dtype)` activation cast below allocates one M x K temporary per call
    (~10% of vision wall); the native vision GEMM fast path ahead avoids it by streaming A
    with on-the-fly conversion instead. `bias`, when given, is added to the 2D product
    before the output reshape (same result as the callers' trailing `y += bias`).
    `weight_blocked` is the repack_64() panel-major form of `weight`, cached by the caller
    keyed on the resident weight; when present and usable the native kernel consumes it,
    else the plain-`weight` oneDNN/chunked paths below run.
    """

    assert weight.dim() == 2
    assert x.shape[-1] == weight.shape[0], \
        f"matmul shape mismatch: input {tuple(x.shape)} vs weight {tuple(weight.shape)}"
    out_shape = x.shape[:-1] + (weight.shape[1],)
    # Non-contiguous views (callers pass contiguous; be tolerant, never misread)
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    if out_dtype is None:
        out_dtype = x.dtype

    # Native vision GEMM: register-blocked fp32 accumulation over panel-major B with
    # on-the-fly activation conversion and a fused bias+cast epilogue (no temporaries).
    # Any kernel gap falls through to the oneDNN/chunked paths below; a forward never
    # fails here.
    if (
        weight_blocked is not None
        and x.is_cpu
        and (bias is None or bias.is_cpu)
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight_blocked.dtype in (torch.float32, torch.bfloat16)
        and weight_blocked.dim() == 3 and weight_blocked.size(2) == 64
        and weight_blocked.size(1) == weight.shape[0]
        and weight_blocked.size(0) * 64 >= weight.shape[1]
        and out_dtype in (torch.float32, torch.float16, torch.bfloat16)
    ):
        try:
            from ..ext import exllamav3_ext as ext
            y = torch.empty((x2.shape[0], weight.shape[1]), dtype = out_dtype, device = "cpu")
            ext.vision_gemm_blocked_cpu(x2, weight_blocked, bias, y)
            return y.reshape(out_shape)
        except Exception:
            pass
    same_dtype = x.dtype == weight.dtype and out_dtype == x.dtype

    if same_dtype:
        y = torch.matmul(x2, weight)
        if bias is not None:
            y = y + bias
        return y.reshape(out_shape)

    # Fast compute paths (oneDNN GEMM, fp32 accumulation): fp32 weights reach ~0.9 TFLOP/s
    # SGEMM and bf16 weights ~3 TFLOP/s (AVX512-BF16 dot) on desktop Zen5, while fp16
    # matmul has no vectorized CPU kernel at all (~3 GFLOP/s scalar fallback). Weights are
    # kept resident in fp32 (or bf16 where callers opt into it); activations are cast here.
    if weight.dtype in (torch.float32, torch.bfloat16):
        y = torch.matmul(x2.to(weight.dtype), weight)
        if out_dtype != weight.dtype:
            y = y.to(out_dtype)
        if bias is not None:
            y = y + bias
        return y.reshape(out_shape)

    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError(
            f"cpu_matmul expects floating-point tensors, got x: {x.dtype}, weight: {weight.dtype}"
        )
    if out_dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Unsupported mixed-dtype cpu_matmul output: {out_dtype}")

    # Mixed input/output dtypes (e.g. half weights, float activations): accumulate in FP32,
    # chunked over the reduction dimension to bound the float conversion temporaries
    acc_dtype = torch.float32 if out_dtype == torch.float32 else torch.float64
    acc = torch.zeros(x2.shape[0], weight.shape[1], dtype = acc_dtype)
    k_dim = x2.shape[1]
    for k0 in range(0, k_dim, cpu_matmul_chunk):
        x_chunk = x2[:, k0:k0 + cpu_matmul_chunk].to(acc_dtype)
        w_chunk = weight[k0:k0 + cpu_matmul_chunk, :].to(acc_dtype)
        acc.addmm_(x_chunk, w_chunk)
    out = acc.to(out_dtype) if acc_dtype != out_dtype else acc
    if bias is not None:
        out = out + bias
    return out.reshape(out_shape)


def to2(
    x: torch.Tensor,
    dtype1: torch.dtype | None,
    dtype2: torch.dtype | None = None
):
    if dtype1 is not None:
        x = x.to(dtype1)
    elif dtype2 is not None:
        x = x.to(dtype2)
    return x


def save_tensor_image(
    t: torch.Tensor,
    path: str,
):
    import matplotlib.cm as cm
    from PIL import Image

    t = t.detach().to("cpu", copy = True).float()

    k = 3
    _, sigma = t.mean(), t.std()
    lo, hi = -k * sigma, k * sigma
    t.clamp_(lo, hi)
    t -= lo
    t /= (hi - lo + 1e-8)

    rgba = cm.get_cmap("berlin")(t.numpy())
    rgb8 = (rgba[..., :3] * 255).astype("uint8")
    im = Image.fromarray(rgb8)
    im.save(path)


class GTensorCache:
    """
    Kept per-device tensors shared between modules, for SMALL decode-class buffers: a decode step is often CPU-bound,
    and reusing a buffer skips torch.empty. Anything sized by a prefill chunk or the context belongs to torch's
    caching allocator instead (which shares memory between modules just as well, and gives it back): a kept entry
    pins its VRAM for the life of the process. get_scratch() applies that rule for workspaces whose size follows a
    runtime row count.
    """

    # Row count up to which get_scratch() keeps a workspace (decode, MTP verify, small batches)
    SCRATCH_STATIC_ROWS = 32

    def __init__(self):
        self.cache = {}

    def make_key(self, device, shape, dtype, x):
        device = torch.device(device)
        return f"{device}/{str(shape)}/{str(dtype)}/{x}"

    def get(self, device, shape, dtype, x = ""):
        key = self.make_key(device, shape, dtype, x)
        if key not in self.cache:
            refc, v = (0, torch.empty(shape, dtype = dtype, device = device))
        else:
            refc, v = self.cache[key]
        self.cache[key] = (refc + 1, v)
        return v

    def get_bucketed(self, device, numel, dtype, x = ""):
        """Flat backing rounded up to the next power of two, sliced to numel. For workspaces
        whose size is parameterized by a runtime shape (batch size, query length, split
        count): nearby sizes share one kept entry instead of ratcheting a new allocation per
        distinct size, and the total kept per tag is bounded by 2x the largest use."""
        nb = 1 << max(numel - 1, 0).bit_length()
        return self.get(device, (nb,), dtype, x)[:numel]

    def get_scratch(self, device, numel, dtype, x, rows: int, static_rows: int | None = None):
        """Flat workspace of numel elements for one call processing `rows` rows. Decode-class calls (rows up to
        static_rows, default SCRATCH_STATIC_ROWS) take a kept bucketed backing (get_bucketed); larger calls allocate
        through the caching allocator, rounded up to a power of two so the freed block is reused across chunk sizes
        and context lengths instead of one segment being cached per size. Must not be used for buffers a CUDA graph
        captures (those have to stay static at any size)"""
        if rows <= (self.SCRATCH_STATIC_ROWS if static_rows is None else static_rows):
            return self.get_bucketed(device, numel, dtype, x)
        nb = 1 << max(numel - 1, 0).bit_length()
        return torch.empty((nb,), dtype = dtype, device = device)[:numel]

    # def drop(self, device, shape, dtype, x = ""):
    #     key = self.make_key(device, shape, dtype, x)
    #     refc, v = self.cache[key]
    #     if refc == 1:
    #         del self.cache[key]
    #     else:
    #         self.cache[key] = (refc - 1, v)

    def drop_all(self):
        self.cache.clear()

g_tensor_cache = GTensorCache()
