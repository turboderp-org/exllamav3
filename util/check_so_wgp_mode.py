#!/usr/bin/env python3
"""Scan a HIP fat .so (or extracted code-object dir) for .workgroup_processor_mode.

msgpack uint after the key: 0x00 = CU, 0x01 = WGP (gfx11 default).
Usage:
  check_so_wgp_mode.py <path-to-so-or-dir> [--require cu|wgp]
"""
import sys
from collections import Counter
from pathlib import Path

key = b".workgroup_processor_mode"


def scan_bytes(data: bytes) -> Counter:
    c = Counter()
    i = 0
    while True:
        j = data.find(key, i)
        if j < 0:
            break
        v = data[j + len(key)] if j + len(key) < len(data) else -1
        c[v] += 1
        i = j + len(key)
    return c


def scan_path(p: Path) -> Counter:
    c = Counter()
    if p.is_dir():
        for f in list(p.glob("*.o")) + list(p.glob("*.so")):
            c.update(scan_bytes(f.read_bytes()))
    else:
        c.update(scan_bytes(p.read_bytes()))
    return c


def main():
    args = sys.argv[1:]
    require = None
    if "--require" in args:
        i = args.index("--require")
        require = args[i + 1].lower()
        del args[i:i + 2]
    p = Path(args[0])
    c = scan_path(p)
    print("path", p)
    print("mode_bytes", {hex(k): v for k, v in sorted(c.items())})
    cu, wgp = c.get(0x00, 0), c.get(0x01, 0)
    if wgp and not cu:
        verdict = "WGP"
    elif cu and not wgp:
        verdict = "CU"
    else:
        verdict = "MIXED_OR_EMPTY"
    print("verdict", verdict)
    if require:
        if verdict != require.upper():
            raise SystemExit(f"required {require.upper()}, got {verdict} {dict(c)}")
        print("ELF_MODE_VERIFIED")


if __name__ == "__main__":
    main()
