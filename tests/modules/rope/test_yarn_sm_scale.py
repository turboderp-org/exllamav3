"""
yarn_sm_scale_fold, the softmax-scale factor DeepSeek-family attention applies under YaRN (exllamav3.util.rope):
yarn_get_mscale(factor, mscale_all_dim)**2 for YaRN-type ropes with mscale_all_dim, 1.0 otherwise, against the
closed form 0.1 * mscale_all_dim * ln(factor) + 1 (HF transformers' yarn_get_mscale).
"""

import math

import pytest

from exllamav3.util.rope import yarn_get_mscale, yarn_sm_scale_fold

pytestmark = pytest.mark.nogpu


def test_get_mscale():
    assert yarn_get_mscale(1.0, 1.0) == 1.0
    assert yarn_get_mscale(0.5, 1.0) == 1.0
    assert yarn_get_mscale(40.0, 1.0) == pytest.approx(0.1 * math.log(40.0) + 1.0)
    assert yarn_get_mscale(40.0, 0.707) == pytest.approx(0.1 * 0.707 * math.log(40.0) + 1.0)


@pytest.mark.parametrize("type_key", ["type", "rope_type"])
def test_fold_deepseek_v3_yarn(type_key):
    # DeepSeek-V3's release config
    rs = {type_key: "yarn", "factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0, "beta_fast": 32, "beta_slow": 1,
          "original_max_position_embeddings": 4096}
    assert yarn_sm_scale_fold(rs) == pytest.approx((0.1 * math.log(40.0) + 1.0) ** 2)


@pytest.mark.parametrize("rs", [
    None,
    {},
    {"rope_type": "default", "rope_theta": 10000.0},                   # rope_parameters of an unscaled model
    {"rope_type": "yarn", "factor": 40},                               # no mscale_all_dim
    {"rope_type": "yarn", "factor": 40, "mscale_all_dim": 0},
    {"rope_type": "yarn", "factor": 1.0, "mscale_all_dim": 1.0},       # no extension
], ids = ["none", "empty", "default", "no_mscale_all_dim", "zero_mscale_all_dim", "factor_1"])
def test_no_fold(rs):
    assert yarn_sm_scale_fold(rs) == 1.0
