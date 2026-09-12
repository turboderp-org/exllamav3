from unittest.mock import patch

import pytest

from exllamav3.architecture.mtp_hot_vocab import MTPHotVocabConfig
from exllamav3.architecture.qwen3_5_mtp import validate_mtp_hot_blocks


def test_configuration():
    assert not MTPHotVocabConfig().enabled
    assert MTPHotVocabConfig(blocks_path = "blocks.txt", embedding_dtype = "fp8").enabled


@pytest.mark.parametrize("kwargs", [
    {"blocks_path": "blocks.txt", "embedding_dtype": "int8"},
    {"validate_full_head": True},
])
def test_configuration_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        MTPHotVocabConfig(**kwargs)


def test_environment_configuration():
    env = {
        "EXL3_MTP_HOT_BLOCKS": "blocks.txt",
        "EXL3_MTP_HOT_EMBED_DTYPE": "fp8",
        "EXL3_MTP_VALIDATE_SUBHEAD": "1",
    }
    with patch.dict("os.environ", env, clear = True):
        config = MTPHotVocabConfig.from_env()
    assert config.blocks_path == "blocks.txt"
    assert config.embedding_dtype == "fp8"
    assert config.validate_full_head


def test_accepts_noncontiguous_hadamard_groups():
    validate_mtp_hot_blocks(
        list(range(8, 16)) + list(range(24, 32)),
        full_vocab = 4096,
    )


@pytest.mark.parametrize("blocks", [
    list(range(7)),
    list(range(1, 9)),
    list(range(8)) + list(range(7, 15)),
])
def test_rejects_partial_or_misaligned_groups(blocks):
    with pytest.raises(ValueError):
        validate_mtp_hot_blocks(blocks, full_vocab = 4096)


def test_rejects_out_of_range_group():
    with pytest.raises(ValueError):
        validate_mtp_hot_blocks(list(range(32, 40)), full_vocab = 512)
