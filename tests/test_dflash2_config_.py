import os

import pytest

DFLASH2_HF_DIR = os.environ.get("EXL3_TEST_DFLASH2", "")

requires_dflash2_hf = pytest.mark.skipif(
    not DFLASH2_HF_DIR or not os.path.exists(os.path.join(DFLASH2_HF_DIR, "config.json")),
    reason = "DFlash2 draft checkpoint not available (set EXL3_TEST_DFLASH2 to its directory)",
)


def test_dflash2_arch_registered():
    from exllamav3.architecture.architectures import get_architectures
    archs = get_architectures()
    assert "DFlash2DraftModel" in archs
    assert archs["DFlash2DraftModel"]["model_class"].__name__ == "DFlash2Model"


@requires_dflash2_hf
def test_dflash2_config_reads_dflash2_fields():
    from exllamav3.architecture.dflash2 import DFlash2Config
    cfg = DFlash2Config(DFLASH2_HF_DIR)
    assert cfg.arch_string == "DFlash2DraftModel"
    assert cfg.conv_kernel_size == 2
    assert cfg.conv_group_size == 16
    assert cfg.selector_rank == 256
    assert cfg.selector_top_k == 16
    assert cfg.block_size == 8
    # reference extract_context_feature uses hidden_states[layer_id + 1]; v1 tap_shift default is 1
    assert cfg.target_layer_ids == [6, 20, 34, 48, 62]
    assert cfg.mask_token_id == 248070
    # v1-derived caps must remain intact
    assert cfg.arch_string != "DFlashDraftModel"
