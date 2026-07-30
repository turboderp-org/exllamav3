from __future__ import annotations
import torch

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
            dv = v.to(device)
            scache[device] = dv
    else:
        # Pinned sources upload asynchronously: the copy is stream-ordered ahead of the kernels
        # that consume it, so the host never stalls. Callers that reuse pinned staging buffers
        # must not refill them until a sync point (the generator syncs every iteration when
        # collecting sampled tokens)
        nb = v.device.type == "cpu" and v.is_pinned()
        dv = v.to(device, non_blocking = nb)
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
