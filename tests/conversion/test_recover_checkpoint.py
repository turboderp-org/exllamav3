"""
Resuming after an interrupt between checkpoint saving's two renames: ckpt/ is gone, and recover_checkpoint restores
the complete new checkpoint from ckpt_new/, else the previous one from ckpt_old/. An existing ckpt/ is left alone.
"""

import json

import pytest

from exllamav3.conversion.convert_model import recover_checkpoint


def _ckpt(work_dir, name, idx):
    d = work_dir / name
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"next_module_idx": idx}))


def _restored_idx(work_dir):
    return json.loads((work_dir / "ckpt" / "job.json").read_text())["next_module_idx"]


@pytest.mark.nogpu
def test_restores_new_checkpoint_after_first_rename(tmp_path):
    _ckpt(tmp_path, "ckpt_old", 4)
    _ckpt(tmp_path, "ckpt_new", 5)
    recover_checkpoint(str(tmp_path))
    assert _restored_idx(tmp_path) == 5
    assert (tmp_path / "ckpt_old").exists() and not (tmp_path / "ckpt_new").exists()


@pytest.mark.nogpu
def test_falls_back_to_old_checkpoint(tmp_path):
    _ckpt(tmp_path, "ckpt_old", 4)
    recover_checkpoint(str(tmp_path))
    assert _restored_idx(tmp_path) == 4


@pytest.mark.nogpu
def test_existing_checkpoint_is_kept(tmp_path):
    _ckpt(tmp_path, "ckpt", 5)
    _ckpt(tmp_path, "ckpt_new", 6)  # a save interrupted before its renames: incomplete, ignored
    recover_checkpoint(str(tmp_path))
    assert _restored_idx(tmp_path) == 5
    assert (tmp_path / "ckpt_new").exists()


@pytest.mark.nogpu
def test_empty_ckpt_from_failed_resume_is_replaced(tmp_path):
    (tmp_path / "ckpt").mkdir()  # prepare_env recreated it during an earlier failed resume
    _ckpt(tmp_path, "ckpt_new", 5)
    recover_checkpoint(str(tmp_path))
    assert _restored_idx(tmp_path) == 5
