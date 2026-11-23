from argparse import ArgumentParser

import pytest
import torch

from exllamav3 import model_init

pytestmark = pytest.mark.nogpu


def parse_args(*args):
    parser = ArgumentParser()
    model_init.add_args(parser)
    return parser.parse_args(["-m", "dummy", *args])


def test_recurrent_state_format_cli():
    assert parse_args().recurrent_state_format == "fp16"

    args = parse_args("-rsf", "int9")
    assert model_init.RECURRENT_STATE_FORMATS[args.recurrent_state_format] == (torch.int8, 9)

    args = parse_args("-rs32")
    assert model_init.RECURRENT_STATE_FORMATS[args.recurrent_state_format] == (torch.float, 0)
