"""
Recurrent checkpoint interval arguments of Generator: unaligned intervals are rejected, defaults and the
architecture default apply, an explicit interval overrides it, and the prefill interval rounds up to the chunk
size. Checked against the documented argument contract with a stub model and cache, without GPU memory.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from exllamav3.generator.generator import Generator

pytestmark = pytest.mark.nogpu

UNALIGNED = (1, 255, 257, 1000, 32769)


def make_generator(recurrent = True, default_interval = None, **kwargs):
    caps = {"recurrent_states": recurrent}
    if default_interval is not None:
        caps["default_recurrent_checkpoint_interval"] = default_interval
    model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = caps)
    cache = SimpleNamespace(num_slots = 4, reset_states = Mock())
    with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
         patch("exllamav3.generator.generator.ThreadPoolExecutor"):
        return Generator(model, cache, tokenizer = None, **kwargs)


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("interval", UNALIGNED)
def test_rejects_unaligned_prefill_interval(recurrent, interval):
    with pytest.raises(AssertionError, match = "checkpoint interval must be a multiple"):
        make_generator(recurrent = recurrent, recurrent_checkpoint_interval_pp = interval)


@pytest.mark.parametrize("interval", UNALIGNED)
def test_rejects_unaligned_generation_interval(interval):
    with pytest.raises(AssertionError, match = "checkpoint interval must be a multiple"):
        make_generator(recurrent_checkpoint_interval = interval)


def test_defaults():
    gen = make_generator()
    assert gen.recurrent_checkpoint_interval == 2048
    assert gen.recurrent_checkpoint_interval_pp == 32768


def test_architecture_default():
    gen = make_generator(default_interval = 8192)
    assert gen.recurrent_checkpoint_interval == 8192
    assert gen.recurrent_checkpoint_interval_pp == 32768


def test_explicit_interval_overrides_architecture_default():
    gen = make_generator(default_interval = 8192, recurrent_checkpoint_interval = 256)
    assert gen.recurrent_checkpoint_interval == 256


@pytest.mark.parametrize("interval, expected", [(256, 2048), (2048, 2048), (2304, 4096), (32768, 32768)])
def test_aligned_prefill_interval_still_rounds_to_chunk_size(interval, expected):
    gen = make_generator(recurrent_checkpoint_interval_pp = interval, max_chunk_size = 2048)
    assert gen.recurrent_checkpoint_interval_pp == expected
