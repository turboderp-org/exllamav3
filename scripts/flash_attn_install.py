#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich>=13.0"]
# ///
"""Install a prebuilt flash-attn wheel for the current environment.

Compiling flash-attn from source is extremely slow, so this helper finds the exact
`mjun0812/flash-attention-prebuild-wheels` GitHub release wheel that matches the
*current* environment (torch build + CUDA flavor + Python version + platform) and
installs it directly, instead of resolving through the index proxy.

It prints what it detected, shows the exact `uv pip install ...` command it will
run, and asks for confirmation ([Yn]) before running it.

Usage:
    uv run scripts/flash_attn_install.py [--python-bin /path/to/python]

The install target is picked in this order:
1. the active ``VIRTUAL_ENV``'s python (the throwaway env uv creates
   for this PEP 723 script is ignored);
2. the python uv would use for the project (``pyproject.toml`` in the current
   directory or any parent) via ``uv run python``;
3. an explicit ``--python-bin``;
4. an error if none of the above applies.

Only stdlib plus ``rich`` (PEP 723) are used to introspect the environment.

Wheel data comes from ``wheelhouse/releases.json`` (a cached copy of upstream's
manifest); if it is missing, the script downloads it fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
from pathlib import Path

try:
    from rich.console import Console
    from rich.prompt import Confirm
    from rich.table import Table
except ImportError:  # pragma: no cover - only when PEP 723 deps weren't applied
    from rich.console import Console
    from rich.prompt import Confirm
    from rich.table import Table

# Upstream manifest, same file the index-proxy uses.
_RELEASES_URL = ("https://raw.githubusercontent.com/mjun0812/"
                "flash-attention-prebuild-wheels/main/docs/data/releases.json")

# Where the cached upstream manifest lives in this repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RELEASES = _REPO_ROOT / "wheelhouse" / "releases.json"

# Wheel filenames carry the torch minor as "+cu<CUDA>torch<X.Y>" in the
# version's local segment; match the X.Y that follows "torch".
_TORCH_RE = re.compile(r"\+cu\d+torch(?P<torch>\d+\.\d+)")
_CUDA_RE = re.compile(r"\+cu(?P<cuda>\d+)torch")

# Map sysconfig.get_platform()-style host strings to the wheel platform substrings we
# accept for that host (checked in order). An empty list means "any platform".
# ARM is normalized to "arm64" (the canonical name for these wheels); the match
# patterns accept both the "aarch64" and "arm64" tags wheels may carry.
_HOST_PLATFORMS = {
    "linux": ["linux_x86_64", "manylinux.*x86_64"],
    "arm64": ["linux_aarch64", "manylinux.*aarch64", "arm64", "manylinux.*arm64"],
    "win": ["win_amd64"],
    "mac": ["macosx_x86_64", "macosx_arm64"],
}

console = Console()


def die(msg: str) -> None:
    console.print(f"[bold red]error:[/] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- detection
def _venv_python(venv: Path) -> Path | None:
    for cand in (venv / "bin" / "python", venv / "Scripts" / "python.exe",
                venv / "bin" / "python3"):
        if cand.exists():
            return cand
    return None


def _is_uv_ephemeral(venv: Path) -> bool:
    """True if *venv* is the throwaway env ``uv run`` creates for a PEP 723
    script (it lives under ``<UV_CACHE>/environments-v2/<name>``) and therefore
    must not be treated as the user's active virtualenv."""
    return any(part == "environments-v2" for part in venv.parts)


def _find_project_root(start: Path) -> Path | None:
    """Nearest ancestor of *start* (inclusive) that has a pyproject.toml."""
    for d in (start, *start.parents):
        if (d / "pyproject.toml").is_file():
            return d
    return None


def project_python() -> Path | None:
    """Resolve the python *uv run* would use for the project in cwd/parents."""
    root = _find_project_root(Path.cwd())
    if root is None:
        return None
    cmd = ["uv", "run", "-q", "python",
           "-c", "import sys; print(sys.executable)"]
    try:
        res = subprocess.run(cmd, cwd=str(root), capture_output=True,
                            text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    lines = res.stdout.strip().splitlines()
    return Path(lines[-1]) if lines else None


def resolve_target(python_bin: str | None) -> Path:
    """Pick the interpreter the install will target, in order:
    1. the active ``VIRTUAL_ENV``'s python (the throwaway env uv creates
   for this PEP 723 script is ignored);
    2. the python uv would use for the project (``pyproject.toml`` in cwd/parents)
       via ``uv run python``;
    3. a manually supplied ``--python-bin``;
    4. error."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and not _is_uv_ephemeral(Path(venv)):
        cand = _venv_python(Path(venv))
        if cand:
            return cand

    cand = project_python()
    if cand:
        return cand

    if python_bin:
        return Path(python_bin)

    die("could not determine an install target: no active VIRTUAL_ENV, no project "
        "(pyproject.toml) in the current directory or parents, and no --python-bin "
        "given")


_PROBE = """\
import json
import platform
import sys
import sysconfig

out = {
    "py": list(sys.version_info[:2]),
    "platform": sysconfig.get_platform(),
    "machine": platform.machine(),
    "py_impl": platform.python_implementation(),
    "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
}
try:
    import torch
    out["torch"] = torch.__version__
except Exception:
    out["torch"] = None
print(json.dumps(out))
"""


def introspect(python: Path) -> dict:
    """Ask *python* (the install-target interpreter) about its env: python version,
    torch version (+cuda), and platform. Returns a dict or raises."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                   encoding="utf-8") as fh:
        fh.write(_PROBE)
        probe = fh.name
    try:
        res = subprocess.run([str(python), probe], capture_output=True,
                            text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {python}: {exc}") from exc
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    if res.returncode != 0:
        raise RuntimeError(f"introspection failed on {python}: {res.stderr.strip()}")
    data = json.loads(res.stdout.strip().splitlines()[-1])
    return data


def py_tag(info: dict) -> str:
    major, minor = info["py"][0], info["py"][1]
    tag = f"cp{major}{minor}"
    if info.get("py_impl", "CPython").lower() != "cpython":
        tag = f"py{major}{minor}"
    return tag


# ---------------------------------------------------------------- wheel data
def load_releases(path: Path | None) -> dict:
    if path is not None and path.exists():
        console.print(f"Using wheel manifest [cyan]{path}[/]")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    console.print("Downloading upstream wheel manifest...")
    with urllib.request.urlopen(_RELEASES_URL, timeout=60) as resp:
        data = json.load(resp)
    if path is not None:  # cache a copy for next time
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                            encoding="utf-8") as fh:
                json.dump(data, fh)
                tmp = fh.name
            os.replace(tmp, path)
        except OSError:
            pass
    return data


def wheel_urls(releases: dict) -> dict[str, str]:
    """Dedup name -> browser_download_url (keep newest upload per wheel)."""
    by_name: dict[str, tuple[str, str]] = {}  # name -> (updated_at, url)
    for rel in releases.get("releases", []):
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if not name.endswith(".whl"):
                continue
            prev = by_name.get(name)
            upd = asset.get("updated_at", "")
            if prev is None or upd > prev[0]:
                by_name[name] = (upd, asset["browser_download_url"])
    return {name: url for name, (_, url) in by_name.items()}


def parse_wheel(name: str) -> dict | None:
    """Mirror of flash_wheels.parse_wheel; kept here so this script is standalone."""
    m = re.match(
        r"^flash_attn-(?P<version>[^-]+)-(?P<pyver>[^-]+)-"
        r"(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$", name)
    if not m:
        return None
    version = m.group("version")
    torch_m = _TORCH_RE.search(version)
    cuda_m = _CUDA_RE.search(version)
    return {
        "version": version,
        "torch": torch_m.group("torch") if torch_m else None,
        "cuda": cuda_m.group("cuda") if cuda_m else None,
        "pyver": m.group("pyver"),
        "abi": m.group("abi"),
        "pytag": m.group("platform").split(".")[0],
    }


def _host_tag(info: dict) -> str | None:
    p = info.get("platform", "")  # e.g. "linux-aarch64" or "win-amd64"
    p = p.lower()
    if "win" in p:
        return "win"
    if "darwin" in p or "mac" in p:
        return "mac"
    if "aarch64" in p or "arm64" in p:
        return "arm64"
    if "linux" in p:
        return "linux"
    return None


def find_wheel(urls: dict[str, str], info: dict) -> str | None:
    """Return the best-matching wheel name, or None."""
    want_torch = info.get("torch_minor")     # e.g. "2.13"
    want_cuda = info.get("cuda")             # e.g. "130"
    want_py = py_tag(info)                    # e.g. "cp312"
    want_abi = want_py + ("t" if info.get("gil_disabled") else "")
    host = _host_tag(info)

    # Score candidates: higher flash-attn base version wins; ties broken by platform
    # specificity (linux_x86_64 beats manylinux which is the same, so just stability).
    best = None
    best_score = None
    for name in urls:
        w = parse_wheel(name)
        if not w:
            continue
        if w["cuda"] != want_cuda or w["torch"] != want_torch:
            continue
        if w["abi"] != want_abi:
            continue
        if host is None:
            pass  # unknown host: accept any platform
        else:
            choices = _HOST_PLATFORMS.get(host, [])
            ptag = w["pytag"]
            if not any(re.fullmatch(ch, ptag) for ch in choices):
                continue
        # prefer newest flash-attn base version
        base = w["version"].split("+")[0]
        try:
            key = tuple(int(x) for x in base.split("."))
        except ValueError:
            key = (0,)
        score = key
        if best_score is None or score > best_score:
            best_score = score
            best = name
    return best


def nearest_torch_for(urls: dict[str, str], info: dict) -> str | None:
    """Find the nearest torch minor with a wheel for this CUDA+python+platform,
    so we can suggest a downgrade when no wheel matches the current torch."""
    want_cuda = info.get("cuda")
    want_abi = py_tag(info) + ("t" if info.get("gil_disabled") else "")
    host = _host_tag(info)

    def matches(w: dict) -> bool:
        if w["cuda"] != want_cuda or w["abi"] != want_abi:
            return False
        if host is not None:
            choices = _HOST_PLATFORMS.get(host, [])
            if not any(re.fullmatch(ch, w["pytag"]) for ch in choices):
                return False
        return True

    def minor(t: str) -> tuple:
        return tuple(int(x) for x in t.split("."))

    want = minor(info.get("torch_minor") or "")
    cands = {w["torch"] for w in (parse_wheel(n) for n in urls)
             if w and matches(w)}
    if not cands:
        return None
    return min(cands, key=lambda t: (abs(minor(t)[0] - want[0]),
                                     abs(minor(t)[1] - want[1])))


# ---------------------------------------------------------------- installer
def run_install(url: str, python: Path) -> None:
    cmd = ["uv", "pip", "install", "--python", str(python), url]
    console.print(f"\n[bold]Running:[/] [cyan]{' '.join(cmd)}[/]\n")
    try:
        proc = subprocess.run(cmd)
    except FileNotFoundError:
        die("could not find `uv` on PATH")
    if proc.returncode != 0:
        die(f"`uv pip install` failed with exit code {proc.returncode}")
    console.print("[bold green]Installed flash-attn successfully.[/]")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--releases-json", default=str(_DEFAULT_RELEASES),
                    help="Path to a cached releases.json (default: %(default)s)")
    ap.add_argument("--python-bin", default=None,
                    help="Python binary to use as a fallback install target "
                         "(used if there is no active VIRTUAL_ENV or project)")
    args = ap.parse_args()

    python = resolve_target(args.python_bin)
    try:
        info = introspect(python)
    except RuntimeError as exc:
        die(str(exc))

    version = info.get("torch")
    if not version:
        die(
            "torch is not installed in the active environment "
            f"({python}). Install a CUDA build of torch first."
        )

    # torch.__version__ is like "2.13.0+cu130" (no "torch" word). Derive the
    # minor "2.13" and the local CUDA segment "cu130" directly.
    torch_minor_m = re.match(r"(\d+\.\d+)", version)
    cuda_m = re.search(r"\+cu(\d+)", version)
    torch_minor = torch_minor_m.group(1) if torch_minor_m else None
    cuda = cuda_m.group(1) if cuda_m else None
    info["torch_minor"] = torch_minor
    info["cuda"] = cuda

    # Report what we detected.
    table = Table(title="Detected environment")
    table.add_column("Attribute", style="bold")
    table.add_column("Value")
    table.add_row("Python", f"{info['py'][0]}.{info['py'][1]} ({python})")
    table.add_row("Torch", version)
    table.add_row("CUDA flavor", f"cu{cuda}" if cuda else "(not a +cu build)")
    table.add_row("Platform", f"{info.get('platform')} ({info.get('machine')})")
    console.print(table)

    if torch_minor is None or cuda is None:
        die(
            f"Could not parse CUDA/torch flavor from `{version}`. "
            "Install a CUDA build of torch (e.g. torch==2.x.y+cu12x)."
        )

    releases = load_releases(Path(args.releases_json))
    urls = wheel_urls(releases)
    if not urls:
        die("no prebuilt wheels found in the manifest")
    wheel = find_wheel(urls, info)
    if wheel is None:
        hint = nearest_torch_for(urls, info)
        hint_txt = (
            f"\nThe nearest torch minor with a wheel for cu{cuda}/{py_tag(info)}/"
            f"{info.get('platform')} is torch=={hint}.*; try downgrading torch, "
            "or wait for upstream to build a wheel for this torch version."
            if hint else ""
        )
        die(
            f"No wheel matches torch=={torch_minor}.*, +cu{cuda}, "
            f"{py_tag(info)}, {info.get('platform')}. "
            "Check that a prebuilt wheel exists for this combination."
            f"{hint_txt}"
        )

    url = urls[wheel]
    console.print(
        f"\n[magenta]Matched wheel:[/] [cyan]{wheel}[/]\n"
        f"[bold]uv pip install[/] [green]{url}[/]"
    )

    if Confirm.ask("\nInstall this wheel into the active environment?", default=True):
        run_install(url, python)
    else:
        console.print("Aborted.")


if __name__ == "__main__":
    main()
