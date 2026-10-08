"""
Host-memory registration and device-query helpers of the extension (exllamav3_ext/cuda_host.cpp,
quant/exl3_devctx.cu), against torch's own device properties and round trips through real memory:

- g_get_num_sms(dev) == torch multi_processor_count; g_get_cc(dev) is the architecture CLASS the host-side kernel
  selection keys on (CC_OLD 1 / AMPERE 2 / ADA 3 / HOPPER 4 / BLACKWELL 5), derived from torch's compute capability;
  both stable across calls (cached per device).
- cuda_device_get_attribute(attr, dev) == the attribute (SM count 16, CC major/minor 75/76 against torch); an invalid
  attribute raises, and must not leave the CUDA runtime's last-error set (the siblings cuda_host_register /
  cuda_host_unregister clear it for that reason: torch's next launch check would report it as its own failure).
- cuda_host_get_device_pointer(ptr) of memory registered mapped: a device alias that kernels can read and write
  through; equal to the host pointer (offsets included) when cudaDevAttrCanUseHostPointerForRegisteredMem (91) is
  set, which the TP backend logs and relies on.
- cuda_host_unregister: unpins (torch then reports the region unpinned); unregistering memory that is not
  registered, or twice, is a silent no-op that leaves no pending CUDA error.
- pinned_cuda_view(t, dev): a non-owning CUDA tensor over the pinned storage of t (sizes, strides, dtype and storage
  offset preserved, the given device), zero-copy in both directions; rejects non-CPU and non-pinned tensors.
"""
import mmap

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.cuda_only

ATTR_MULTIPROCESSOR_COUNT = 16
ATTR_CC_MAJOR = 75
ATTR_CC_MINOR = 76
ATTR_CAN_USE_HOST_POINTER = 91
HOST_REGISTER_PORTABLE = 0x01
HOST_REGISTER_MAPPED = 0x02


def dev_index(device) -> int:
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def expected_cc_class(major: int, minor: int) -> int:
    if major >= 10: return 5
    if major >= 9: return 4
    if major == 8 and minor >= 9: return 3
    if major >= 8: return 2
    return 1


def launch_check(device):
    """A torch kernel launch plus sync: raises if a stale CUDA error is pending on this thread"""
    x = torch.arange(64, dtype = torch.float32, device = device)
    torch.softmax(x, 0).sum().item()


class MappedRegion:
    """Anonymous mapping registered with cudaHostRegister (portable | mapped), as PinnedArena does"""

    def __init__(self, nbytes):
        self.size = -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
        self.map = mmap.mmap(-1, self.size, flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.tensor = torch.frombuffer(self.map, dtype = torch.uint8)
        self.ptr = self.tensor.data_ptr()
        ext.cuda_host_register(self.ptr, self.size, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED)

    def close(self):
        torch.cuda.synchronize()
        ext.cuda_host_unregister(self.ptr)


@pytest.fixture
def mapped_region(device):
    torch.cuda.init()
    r = MappedRegion(1 << 20)
    yield r
    r.close()


def test_num_sms(device):
    idx = dev_index(device)
    expect = torch.cuda.get_device_properties(idx).multi_processor_count
    assert ext.g_get_num_sms(idx) == expect
    assert ext.g_get_num_sms(idx) == expect      # cached value


def test_cc_class(device):
    idx = dev_index(device)
    major, minor = torch.cuda.get_device_capability(idx)
    expect = 1 if torch.version.hip else expected_cc_class(major, minor)
    assert ext.g_get_cc(idx) == expect
    assert ext.g_get_cc(idx) == expect


@pytest.mark.parametrize("major,minor,cls", [
    (7, 5, 1), (8, 0, 2), (8, 6, 2), (8, 7, 2), (8, 9, 3), (9, 0, 4), (10, 0, 5), (12, 0, 5),
])
@pytest.mark.nogpu
def test_cc_class_table(major, minor, cls):
    # The reference mapping itself, pinned to the classes the C++ code defines (exl3_devctx.cuh CC_*)
    assert expected_cc_class(major, minor) == cls


def test_device_attribute(device):
    idx = dev_index(device)
    props = torch.cuda.get_device_properties(idx)
    assert ext.cuda_device_get_attribute(ATTR_MULTIPROCESSOR_COUNT, idx) == props.multi_processor_count
    assert ext.cuda_device_get_attribute(ATTR_CC_MAJOR, idx) == props.major
    assert ext.cuda_device_get_attribute(ATTR_CC_MINOR, idx) == props.minor
    assert ext.cuda_device_get_attribute(ATTR_CAN_USE_HOST_POINTER, idx) in (0, 1)


def test_device_attribute_invalid_leaves_no_error(device):
    idx = dev_index(device)
    launch_check(device)
    with pytest.raises(RuntimeError, match = "cudaDeviceGetAttribute"):
        ext.cuda_device_get_attribute(100000, idx)
    # The failed query must not surface later as an unrelated torch launch failure
    launch_check(device)


@pytest.mark.platform("linux")
def test_host_device_pointer(device, mapped_region):
    idx = dev_index(device)
    r = mapped_region
    for off in (0, 4096, 12345):
        dptr = ext.cuda_host_get_device_pointer(r.ptr + off)
        assert dptr != 0
        if ext.cuda_device_get_attribute(ATTR_CAN_USE_HOST_POINTER, idx):
            assert dptr == r.ptr + off
        else:
            assert dptr == ext.cuda_host_get_device_pointer(r.ptr) + off


@pytest.mark.platform("linux")
def test_unregister(device):
    torch.cuda.init()
    r = MappedRegion(1 << 16)
    assert r.tensor.is_pinned()
    ext.cuda_host_unregister(r.ptr)
    assert not r.tensor.is_pinned()
    # Second unregister of the same region, and one of memory never registered: silent no-ops
    ext.cuda_host_unregister(r.ptr)
    plain = torch.zeros(4096, dtype = torch.uint8)
    ext.cuda_host_unregister(plain.data_ptr())
    launch_check(device)
    # The region can be registered again afterwards
    ext.cuda_host_register(r.ptr, r.size, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED)
    assert r.tensor.is_pinned()
    r.close()
    assert not r.tensor.is_pinned()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32, torch.uint8])
@torch.inference_mode()
def test_pinned_cuda_view_torch_pinned(device, dtype):
    idx = dev_index(device)
    base = torch.randint(0, 100, (37, 53), generator = torch.Generator().manual_seed(0)).to(dtype)
    src = base.pin_memory()
    v = ext.pinned_cuda_view(src, idx)
    assert v.device == torch.device("cuda", idx)
    assert v.dtype == dtype and v.shape == src.shape and v.stride() == src.stride()
    if ext.cuda_device_get_attribute(ATTR_CAN_USE_HOST_POINTER, idx):
        assert v.data_ptr() == src.data_ptr()
    # GPU reads host values
    assert torch.equal(v.clone().cpu(), src)
    # GPU writes land in host memory
    v.add_(1)
    torch.cuda.synchronize(idx)
    assert torch.equal(src, base + 1)


@pytest.mark.platform("linux")
@torch.inference_mode()
def test_pinned_cuda_view_layout(device, mapped_region):
    """Registered (PinnedArena-style) memory, a slice at a storage offset, non-contiguous strides"""
    idx = dev_index(device)
    r = mapped_region
    flat = r.tensor[:65536].view(torch.float32)
    flat.copy_(torch.arange(flat.numel(), dtype = torch.float32))
    sl = flat[1000 : 1000 + 64 * 100].view(64, 100)
    assert sl.is_pinned()
    v = ext.pinned_cuda_view(sl, idx)
    assert v.shape == sl.shape and v.stride() == sl.stride()
    assert v.data_ptr() == ext.cuda_host_get_device_pointer(sl.data_ptr())
    assert torch.equal(v.cpu(), sl)

    nc = sl[:, ::3].t()
    assert not nc.is_contiguous()
    vn = ext.pinned_cuda_view(nc, idx)
    assert vn.shape == nc.shape and vn.stride() == nc.stride()
    assert torch.equal(vn.cpu(), nc)

    # A device write through the strided view touches exactly those elements
    before = flat.clone()
    vn.fill_(-1.0)
    torch.cuda.synchronize(idx)
    expect = before.clone()
    expect[1000 : 1000 + 6400].view(64, 100)[:, ::3] = -1.0
    assert torch.equal(flat, expect)


def test_pinned_cuda_view_rejects(device):
    idx = dev_index(device)
    with pytest.raises(RuntimeError, match = "must be pinned"):
        ext.pinned_cuda_view(torch.zeros(16), idx)
    with pytest.raises(RuntimeError, match = "must be a CPU tensor"):
        ext.pinned_cuda_view(torch.zeros(16, device = device), idx)
