from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

_ARCH_LIST = Path(__file__).resolve().parents[1] / "exllamav3" / "util" / "arch_list.py"
_spec = importlib.util.spec_from_file_location("exllamav3_arch_list_contract", _ARCH_LIST)
assert _spec is not None and _spec.loader is not None
arch_list = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch_list)


@pytest.mark.parametrize(
    "gcn_arch_name,device_name,expected",
    [
        ("gfx1201:sramecc+:xnack-", "AMD Radeon RX 7900 XTX", "gfx1201"),
        ("", "AMD Radeon RX 9060 XT", "gfx1200"),
        ("", "AMD Radeon RX 9070 XT", "gfx1201"),
        ("", "AMD Radeon AI PRO R9700", "gfx1201"),
        ("", "AMD Radeon RX 7900 XTX", "gfx1100"),
    ],
)
def test_rocm_arch_fallback_prefers_gcn_arch_name_and_recognizes_gfx12(
    monkeypatch, gcn_arch_name, device_name, expected,
):
    props = SimpleNamespace(gcnArchName = gcn_arch_name)
    fake_torch = SimpleNamespace(
        version = SimpleNamespace(hip = "6.4", cuda = None),
        cuda = SimpleNamespace(
            device_count = lambda: 1,
            get_device_properties = lambda _index: props,
            get_device_name = lambda _index: device_name,
        ),
    )
    monkeypatch.setattr(arch_list, "torch", fake_torch)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising = False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising = False)

    arch_list.maybe_set_arch_list_env()

    assert arch_list.os.environ["PYTORCH_ROCM_ARCH"] == expected


def test_rocm_arch_list_uses_rocminfo_and_preserves_first_seen_dedup_order(monkeypatch):
    fake_torch = SimpleNamespace(version = SimpleNamespace(hip = "6.4", cuda = None))
    rocminfo = """\
Name: gfx1201
Name: gfx1100
Name: gfx1201
Name: amdgcn-amd-amdhsa--gfx11-generic
Name: gfx1030
Name: gfx1100
"""
    monkeypatch.setattr(arch_list, "torch", fake_torch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = rocminfo),
    )
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising = False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising = False)

    arch_list.maybe_set_arch_list_env()

    assert arch_list.os.environ["PYTORCH_ROCM_ARCH"] == "gfx1201;gfx1100;gfx1030"
