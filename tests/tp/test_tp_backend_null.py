"""
TPBackendNull, the backend single-process paths (warmup, non-TP loads) run collectives through, must accept
every collective and lifecycle call and leave the tensor untouched.
"""

import pytest
import torch

from exllamav3.model.model_tp_backend import TPBackendNull

pytestmark = pytest.mark.nogpu


def test_null_backend_is_inert():
    b = TPBackendNull()
    t = torch.ones(8)
    b.all_reduce(t); b.all_reduce(t, False); b.broadcast(t, 0); b.gather(t, None, [0], 0, [8])
    b.fwd_barrier(); b.end_cpu_reduce_jobs(); b.run_cpu_reduce_jobs(); b.close()
    assert torch.equal(t, torch.ones(8))
