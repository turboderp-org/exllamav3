import math
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.util.rope import RoPE, RopeSettings


def yarn_settings(yarn_mscale_ratio = False, **overrides):
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
    }
    rope_scaling.update(overrides)
    return RopeSettings(
        head_dim = 128,
        rope_theta = 1_000_000_000.0,
        rope_scaling = rope_scaling,
        max_position_embeddings = 262144,
        yarn_mscale_ratio = yarn_mscale_ratio,
    )


def test_yarn_mscale_keys_are_inert_without_arch_opt_in():
    # Mistral yarn configs carry mscale = mscale_all_dim = 1.0 as defaults; taking the
    # DeepSeek ratio there drops the attention factor entirely (issue #259)
    rope = RoPE("cpu", yarn_settings(mscale = 1.0, mscale_all_dim = 1.0))
    assert rope.attn_factor == pytest.approx(0.1 * math.log(16.0) + 1.0)


def test_yarn_mscale_ratio_with_arch_opt_in():
    rope = RoPE("cpu", yarn_settings(yarn_mscale_ratio = True, mscale = 1.0, mscale_all_dim = 1.0))
    assert rope.attn_factor == pytest.approx(1.0)
    rope = RoPE("cpu", yarn_settings(yarn_mscale_ratio = True, mscale = 0.707, mscale_all_dim = 0.707))
    assert rope.attn_factor == pytest.approx(1.0)
    rope = RoPE("cpu", yarn_settings(yarn_mscale_ratio = True, mscale = 1.0, mscale_all_dim = 0.707))
    num = 0.1 * 1.0 * math.log(16.0) + 1.0
    den = 0.1 * 0.707 * math.log(16.0) + 1.0
    assert rope.attn_factor == pytest.approx(num / den)


def test_yarn_llama_4_scaling_supersedes_attention_factor():
    # Ministral-3: position-dependent attention scaling replaces the static YaRN factor
    rope = RoPE("cpu", yarn_settings(mscale = 1.0, mscale_all_dim = 1.0, llama_4_scaling_beta = 0.1))
    assert rope.attn_factor == pytest.approx(1.0)


def test_yarn_explicit_attention_factor():
    rope = RoPE("cpu", yarn_settings(mscale = 1.0, mscale_all_dim = 1.0, attention_factor = 0.5))
    assert rope.attn_factor == pytest.approx(0.5)


def test_yarn_default_attention_factor():
    rope = RoPE("cpu", yarn_settings())
    assert rope.attn_factor == pytest.approx(0.1 * math.log(16.0) + 1.0)


def test_yarn_accepts_integral_original_context_float():
    rope = RoPE("cpu", yarn_settings(original_max_position_embeddings = 16384.0))
    assert rope.llama_4_scaling_original == 16384
    assert isinstance(rope.llama_4_scaling_original, int)


def test_yarn_rejects_fractional_original_context():
    with pytest.raises(ValueError, match = "original_max_position_embeddings must be an integer"):
        RoPE("cpu", yarn_settings(original_max_position_embeddings = 16384.5))


def test_yarn_rejects_nonnumeric_original_context():
    with pytest.raises(ValueError, match = "original_max_position_embeddings must be an integer"):
        RoPE("cpu", yarn_settings(original_max_position_embeddings = "16384"))
