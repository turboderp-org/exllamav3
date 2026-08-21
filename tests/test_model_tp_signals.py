import os
import signal
import sys
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exllamav3.model.model_tp_fn as model_tp_fn


class FakeConnection:

    def recv(self):
        return "quit"


class CloseTracker:

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_worker_ignores_terminal_sigint():
    backend = CloseTracker()
    consumer = CloseTracker()

    with (
        patch.object(model_tp_fn, "set_t0"),
        patch.object(model_tp_fn, "log_tp"),
        patch.object(model_tp_fn, "install_parent_death_signal", return_value = False),
        patch.object(model_tp_fn, "init_pg", return_value = {"backend": backend}),
        patch.object(model_tp_fn, "SMConsumer", return_value = consumer),
        patch.object(model_tp_fn.torch.cuda, "synchronize"),
        patch.object(signal, "signal") as set_signal,
    ):
        model_tp_fn.mp_model_worker(
            FakeConnection(),
            device = 0,
            active_devices = [0, 1],
            output_device = 1,
            backend_args = {"type": "nccl"},
            producer = {},
            dbg_t0_ = 0.0,
        )

    set_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
    assert consumer.closed
    assert backend.closed
