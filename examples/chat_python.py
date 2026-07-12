from __future__ import annotations

import os
from pathlib import Path
import importlib.util
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, NamedTuple


PYTHON_TIMEOUT_SECONDS = 120
_CODE_BLOCK_RE = re.compile(r"^```[^\n`]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_PRIVATE_DBUS_CONFIG = """<!DOCTYPE busconfig PUBLIC
 "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>unix:tmpdir=/tmp</listen>
  <auth>EXTERNAL</auth>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>
"""


def extract_longest_codeblock(text: str) -> str | None:
    """Return the contents of the longest complete Markdown code block."""
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None
    return max(matches, key = len).strip()


def bubblewrap_help() -> str:
    return (
        "The /python command requires Bubblewrap (bwrap) on Linux. "
        "Install it with your package manager, e.g. "
        "'sudo pacman -S bubblewrap' on Arch/Manjaro or "
        "'sudo apt install bubblewrap' on Debian/Ubuntu."
    )


def find_bubblewrap() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    return shutil.which("bwrap")


class WaylandDisplay(NamedTuple):
    runtime_dir: Path
    name: str
    socket: Path


class MatplotlibBackend(NamedTuple):
    name: str
    extra_site_packages: Path | None = None


def find_wayland_display(environ: Mapping[str, str] | None = None) -> WaylandDisplay | None:
    """Return the active Wayland socket, if the environment names a real one."""
    environ = os.environ if environ is None else environ
    runtime_string = environ.get("XDG_RUNTIME_DIR")
    display_name = environ.get("WAYLAND_DISPLAY")
    if not runtime_string or not display_name:
        return None

    runtime_dir = Path(runtime_string)
    if not runtime_dir.is_absolute():
        return None
    socket_path = Path(display_name)
    if not socket_path.is_absolute():
        socket_path = runtime_dir / socket_path
    try:
        if not stat.S_ISSOCK(socket_path.stat().st_mode):
            return None
    except OSError:
        return None
    return WaylandDisplay(runtime_dir, display_name, socket_path)


def find_wayland_matplotlib_backend() -> MatplotlibBackend:
    """Choose a Wayland-native backend, falling back to noninteractive Agg."""
    for module in ("PyQt6", "PySide6", "PyQt5", "PySide2"):
        try:
            if importlib.util.find_spec(module):
                return MatplotlibBackend("QtAgg")
        except ModuleNotFoundError:
            pass
    try:
        if importlib.util.find_spec("gi"):
            return MatplotlibBackend("GTK4Agg")
    except ModuleNotFoundError:
        pass

    # Some distro venvs exclude system site-packages even though their base
    # interpreter and GTK libraries live under the already-mounted /usr.
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for site_packages in (
        Path("/usr/lib") / version / "site-packages",
        Path("/usr/lib") / version / "dist-packages",
        Path("/usr/lib/python3/dist-packages"),
    ):
        if (site_packages / "gi").is_dir():
            return MatplotlibBackend("GTK4Agg", site_packages)
    return MatplotlibBackend("Agg")


def _sandbox_path_args(path: Path) -> list[str]:
    """Create empty parents for a bind mount without exposing their contents."""
    args: list[str] = []
    parents = list(path.parents)
    for parent in reversed(parents[:-1]):
        if parent != Path("/"):
            args += ["--dir", str(parent)]
    return args


def build_bubblewrap_command(
    bwrap: str,
    run_dir: Path,
    python_executable: Path | None = None,
    venv_root: Path | None = None,
    wayland: WaylandDisplay | None = None,
    matplotlib_backend: MatplotlibBackend | None = None,
) -> list[str]:
    """Build a Bubblewrap command exposing only the active venv and run directory."""
    python_executable = Path(python_executable or sys.executable).absolute()
    venv_root = Path(venv_root or sys.prefix).absolute()
    matplotlib_backend = matplotlib_backend or (
        find_wayland_matplotlib_backend() if wayland else MatplotlibBackend("Agg")
    )

    command = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
    ]

    # Common merged-/usr layouts use these root-level symlinks. On older
    # layouts, expose the corresponding directories read-only instead.
    for path_string in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(path_string)
        if path.is_symlink():
            command += ["--symlink", os.readlink(path), path_string]
        elif path.exists():
            command += ["--ro-bind", path_string, path_string]

    command += ["--dir", "/etc"]
    for path_string in ("/etc/fonts", "/etc/ld.so.cache", "/etc/machine-id"):
        if Path(path_string).exists():
            command += ["--ro-bind", path_string, path_string]
    for filename in ("passwd", "group"):
        synthetic_identity = run_dir / filename
        if synthetic_identity.exists():
            command += ["--ro-bind", str(synthetic_identity), f"/etc/{filename}"]

    # Keep the venv at its original absolute path. Some entry points and
    # extension modules depend on that layout.
    if venv_root != Path("/usr") and not venv_root.is_relative_to(Path("/usr")):
        command += _sandbox_path_args(venv_root)
        command += ["--ro-bind", str(venv_root), str(venv_root)]

    # Expose only the compositor socket, not the rest of XDG_RUNTIME_DIR
    # (which can contain D-Bus, PipeWire and credential-agent sockets).
    if wayland:
        command += _sandbox_path_args(wayland.socket)
        command += ["--ro-bind", str(wayland.socket), str(wayland.socket)]

    command += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", str(run_dir), "/work",
        "--chdir", "/work",
        "--setenv", "HOME", "/work",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "MPLCONFIGDIR", "/work/.matplotlib",
        "--setenv", "MPLBACKEND", matplotlib_backend.name,
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "PATH", f"{venv_root}/bin:/usr/bin",
    ]
    if wayland:
        command += [
            "--setenv", "XDG_RUNTIME_DIR", str(wayland.runtime_dir),
            "--setenv", "WAYLAND_DISPLAY", wayland.name,
            "--setenv", "XDG_SESSION_TYPE", "wayland",
            "--setenv", "GDK_BACKEND", "wayland",
            "--setenv", "GSK_RENDERER", "cairo",
            "--setenv", "QT_QPA_PLATFORM", "wayland",
            "--setenv", "QT_QUICK_BACKEND", "software",
            "--setenv", "LIBGL_ALWAYS_SOFTWARE", "1",
            "--setenv", "NO_AT_BRIDGE", "1",
            "--setenv", "GTK_A11Y", "none",
        ]
    python_args = [str(python_executable), "-I", "/work/snippet.py"]
    if matplotlib_backend.extra_site_packages:
        site_packages = str(matplotlib_backend.extra_site_packages)
        runner = (
            "import runpy,sys;"
            f"sys.path.append({site_packages!r});"
            "runpy.run_path('/work/snippet.py',run_name='__main__')"
        )
        python_args = [str(python_executable), "-I", "-c", runner]
    if wayland:
        python_args = [
            "/usr/bin/dbus-run-session",
            "--config-file=/work/dbus-session.conf",
            "--",
        ] + python_args

    command += [
        "/usr/bin/prlimit",
        "--cpu=60",
        "--as=4294967296",
        "--fsize=67108864",
        "--nofile=128",
        "--nproc=256",
        "--",
    ] + python_args
    return command


def run_python_sandboxed(snippet: str) -> tuple[int | None, str | None]:
    """Run a snippet under Bubblewrap, returning (exit code, error message)."""
    bwrap = find_bubblewrap()
    if not bwrap:
        return None, bubblewrap_help()

    with tempfile.TemporaryDirectory(prefix = "exllamav3-python-") as dirname:
        run_dir = Path(dirname)
        script = run_dir / "snippet.py"
        script.write_text(snippet, encoding = "utf-8")
        (run_dir / "passwd").write_text(
            f"sandbox:x:{os.getuid()}:{os.getgid()}:Sandbox user:/work:/usr/bin/nologin\n",
            encoding = "utf-8",
        )
        (run_dir / "group").write_text(
            f"sandbox:x:{os.getgid()}:\n",
            encoding = "utf-8",
        )
        (run_dir / "dbus-session.conf").write_text(_PRIVATE_DBUS_CONFIG, encoding = "utf-8")
        command = build_bubblewrap_command(bwrap, run_dir, wayland = find_wayland_display())

        try:
            process = subprocess.Popen(command)
        except OSError as exc:
            return None, f"Could not start Bubblewrap: {exc}"

        try:
            return process.wait(timeout = PYTHON_TIMEOUT_SECONDS), None
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout = 2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return None, f"Python snippet exceeded the {PYTHON_TIMEOUT_SECONDS}-second time limit"
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout = 2)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            return None, "Python snippet interrupted"
