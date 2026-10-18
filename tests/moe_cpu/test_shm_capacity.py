"""
check_shm_capacity: a shared-memory segment larger than /dev/shm must be refused up front with a message naming
the Docker --shm-size fix, and segments that fit must pass. Checked against a stubbed statvfs, plus one query of
the real filesystem.
"""

import os

import pytest

from exllamav3.util import shm

pytestmark = pytest.mark.nogpu


def _fake_statvfs(free_bytes, total_bytes):
    class St:
        f_bavail = free_bytes // 4096
        f_blocks = total_bytes // 4096
        f_frsize = 4096
    return lambda path: St()


def test_refuses_segment_larger_than_shm(monkeypatch):
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(64 * 2**20, 64 * 2**20))
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    with pytest.raises(RuntimeError) as e:
        shm.check_shm_capacity(71397376, "The CPU MoE offload handoff segment")
    msg = str(e.value)
    assert "68.1 MiB" in msg and "64.0 MiB" in msg and "--shm-size" in msg


def test_accepts_segment_that_fits(monkeypatch):
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1024 * 2**20, 1024 * 2**20))
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    shm.check_shm_capacity(71397376, "x")


def test_skips_where_shm_is_unavailable(monkeypatch):
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    shm.check_shm_capacity(1 << 40, "x")


def test_real_filesystem_query():
    fs = shm.shm_free_bytes()
    if fs is None:
        pytest.skip("no /dev/shm")
    free, total = fs
    assert 0 <= free <= total
