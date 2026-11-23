"""
Ctrl-C while conversion's worker threads are busy kills the process by SIGINT. Both thread pools are
covered: quantize_linears_parallel (stub groups whose conversion blocks) and run_row_workers (a blocking worker),
so no device is needed.
"""

import os
import signal
import subprocess
import sys

import pytest

PRELUDE = """
import time
from exllamav3.conversion import convert_model as cm

def busy(*args, **kwargs):
    print("busy", flush = True)
    time.sleep(60)
"""

QUANTIZE_PARALLEL = PRELUDE + """
class StubLinear:
    key = "l"
    qmap = "q"
    def weights_numel(self):
        return 1

cm.get_out_hessian_diags = lambda *args: {}
cm.group_quant_linears = lambda *args, **kwargs: [[StubLinear(), StubLinear()]]
cm.make_quant_args = lambda *args, **kwargs: {}
cm.convert_exl3_group = busy
cm.quantize_linears_parallel(
    {"image_dump": False, "verbose": False}, [StubLinear(), StubLinear()], None, {"l": None}, 0, [0], None,
    {"q": None}, None
)
"""

ROW_WORKERS = PRELUDE + """
cm.run_row_workers(" -- Rows", 1, [busy], [0])
"""


@pytest.mark.nogpu
@pytest.mark.skipif(sys.platform == "win32", reason = "sends SIGINT to a child process")
@pytest.mark.parametrize("child", [QUANTIZE_PARALLEL, ROW_WORKERS], ids = ["quantize_linears_parallel", "run_row_workers"])
def test_ctrl_c_aborts_busy_workers(child):
    proc = subprocess.Popen(
        [sys.executable, "-c", child], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        # A runner started with SIGINT ignored would pass that on, and the child would never see Ctrl-C
        preexec_fn = lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)
    )
    try:
        for line in proc.stdout:
            if line.strip() == "busy":
                break
        else:
            pytest.fail(f"child exited before the workers started, code {proc.wait()}")
        os.kill(proc.pid, signal.SIGINT)
        rest = proc.communicate(timeout = 30)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == -signal.SIGINT, rest
    assert "Aborted." in rest
