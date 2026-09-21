"""Loader headroom uses dedicated free memory on WDDM and CUDA elsewhere."""
import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from exllamav3.util import memory


MiB = 1 << 20
UUID = "00112233-4455-6677-8899-aabbccddeeff"


class FakeNVML:
    def __init__(self, free = 700 * MiB, driver = 0):
        self.free = free
        self.total = 1024 * MiB
        self.driver = driver
        self.nvmlInit_v2 = Mock(return_value = 0)
        self.nvmlShutdown = Mock(return_value = 0)
        self.nvmlDeviceGetHandleByUUID = Mock(side_effect = self.get_handle)
        self.nvmlDeviceGetDriverModel = Mock(side_effect = self.get_driver)
        self.nvmlDeviceGetMemoryInfo = Mock(side_effect = self.get_memory)

    def get_handle(self, uuid, handle):
        assert uuid == ("GPU-" + UUID).encode("ascii")
        handle._obj.value = 0x123456789
        return 0

    def get_driver(self, handle, current, pending):
        assert handle.value == 0x123456789
        current._obj.value = self.driver
        pending._obj.value = 1 - self.driver
        return 0

    def get_memory(self, handle, result):
        assert handle.value == 0x123456789
        result._obj.total = self.total
        result._obj.free = self.free
        result._obj.used = self.total - self.free
        return 0


@pytest.fixture
def queries(monkeypatch, tmp_path):
    nvml = FakeNVML()
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "Program Files"))
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value = nvml))
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform = "win32"))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(return_value =
        SimpleNamespace(uuid = UUID, total_memory = 1000 * MiB)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value = (0, 1000 * MiB)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", Mock(return_value = 100 * MiB))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", Mock())
    monkeypatch.setattr(memory, "touch_device", Mock())
    yield nvml
    module = sys.modules.get("exllamav3.util.nvml")
    if module is not None:
        module._init_nvml.cache_clear()
        module._get_device.cache_clear()


def test_wddm_reserve_includes_current_reservation(queries):
    assert memory.set_memory_fraction_reserve(200 * MiB, 3) == 600 * MiB
    torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.6, device = 3)


@pytest.mark.parametrize("use, expected", [(400, 500), (900, 800)])
def test_wddm_use_is_limited_by_dedicated_free_memory(queries, use, expected):
    assert memory.set_memory_fraction_use(use * MiB, 3) == expected * MiB


def test_wddm_maps_cuda_uuid_and_reads_free_memory_again(queries):
    assert memory.get_memory_info(torch.device("cuda:3")) == (700 * MiB, 1000 * MiB)
    queries.free = 250 * MiB
    assert memory.get_memory_info(torch.device("cuda:3")) == (250 * MiB, 1000 * MiB)
    torch.cuda.get_device_properties.assert_called_with(torch.device("cuda:3"))


@pytest.mark.parametrize("platform, hip", [("linux", None), ("linux", "6.4"), ("win32", "6.4")])
def test_non_windows_cuda_keeps_native_query_without_nvml(queries, monkeypatch, platform, hip):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform = platform))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setitem(sys.modules, "exllamav3.util.nvml", None)
    torch.cuda.mem_get_info.return_value = (321 * MiB, 987 * MiB)
    assert memory.get_memory_info(3) == (321 * MiB, 987 * MiB)
    ctypes.CDLL.assert_not_called()
    torch.cuda.get_device_properties.assert_not_called()


def test_tcc_uses_cuda_even_when_pending_driver_is_wddm(queries):
    queries.driver = 1
    torch.cuda.mem_get_info.return_value = (123 * MiB, 1000 * MiB)
    assert memory.get_memory_info(3) == (123 * MiB, 1000 * MiB)
    queries.nvmlDeviceGetMemoryInfo.assert_not_called()


@pytest.mark.parametrize("free, expected", [(0, 0), (1024 * MiB, 1000 * MiB)])
def test_wddm_free_is_bounded_by_cuda_capacity(queries, free, expected):
    queries.free = free
    assert memory.get_memory_info(3) == (expected, 1000 * MiB)


@pytest.mark.parametrize("function", ["nvmlInit_v2", "nvmlDeviceGetHandleByUUID",
                                      "nvmlDeviceGetDriverModel", "nvmlDeviceGetMemoryInfo"])
def test_nvml_query_errors_are_not_capacity_estimates(queries, function):
    query = getattr(queries, function)
    query.side_effect = None
    query.return_value = 15
    with pytest.raises(RuntimeError, match = function):
        memory.get_memory_info(3)


def test_missing_nvml_is_reported(queries):
    ctypes.CDLL.side_effect = OSError("nvml.dll is missing")
    with pytest.raises(RuntimeError, match = "NVML"):
        memory.get_memory_info(3)


@pytest.mark.parametrize("location", ["Windows/System32", "Program Files/NVIDIA Corporation/NVSMI"])
def test_windows_driver_dll_locations(queries, tmp_path, location):
    dll = tmp_path / location / "nvml.dll"
    dll.parent.mkdir(parents = True)
    dll.touch()

    def load_dll(path):
        if path != str(dll):
            raise OSError("NVML is not on the default DLL search path")
        return queries

    ctypes.CDLL.side_effect = load_dll
    assert memory.get_memory_info(3) == (700 * MiB, 1000 * MiB)


def test_invalid_nvml_free_is_rejected(queries):
    queries.free = queries.total + 1
    with pytest.raises(RuntimeError, match = "NVML.*memory"):
        memory.get_memory_info(3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("free, loads", [(512 * MiB, True), (0, False)])
def test_autosplit_checks_selected_headroom_and_keeps_margin(monkeypatch, free, loads):
    from test_autosplit_transient_ import FakeModule, _load
    from exllamav3.model import model_ls

    memory.free_mem()
    total = torch.cuda.get_device_properties(0).total_memory
    monkeypatch.setattr(model_ls, "set_memory_fraction_use", lambda use, device: 2048 * MiB)
    monkeypatch.setattr(model_ls, "get_memory_info", Mock(return_value = (free, total)), raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value = (0, total)))
    monkeypatch.setenv("EXL3_AUTOSPLIT_MARGIN_MB", "256")
    modules = [FakeModule("a", 128 * MiB)]
    try:
        if loads:
            _load(modules, use = 2048 * MiB)
            assert modules[0].weight is not None
        else:
            with pytest.raises(RuntimeError, match = "Insufficient VRAM.*physical headroom"):
                _load(modules, use = 2048 * MiB)
        assert modules[0].forwards >= 1
    finally:
        for module in modules:
            module.unload()
        memory.free_mem()
