"""Build a code corpus for profiling from a source tree.

Shipped as a generator rather than a blob: the corpus used for the reference `code_long`
profile is 17.4 MB of Python/C++/CUDA taken from the exllamav3 tree itself, which is
already on disk wherever this is being run and would otherwise be duplicated into git.

Domain matters as much as context length. A wikitext-fitted census applied to code is worth
~1.01x -- nothing. A code census on code is worth ~1.49x. Build the corpus from traffic
that resembles what you serve; if you serve an agentic coding workload, prefer real
transcripts over source files, since tool-call and diff structure route differently from
prose-heavy source.

  python util/make_code_corpus.py /path/to/src -o code_corpus.txt
  python util/make_code_corpus.py /path/to/src -o code_corpus.txt --max-mb 24
"""
import argparse, os

DEFAULT_EXTS = (".py", ".cpp", ".cu", ".h", ".cuh", ".hpp", ".c", ".cc", ".rs", ".go", ".ts")
SKIP_DIRS = {".git", "__pycache__", "build", "dist", "node_modules", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", "target"}


def build(roots, exts, max_bytes, min_bytes):
    out, nfiles, total = [], 0, 0
    for root in roots:
        # Sorted walk so the corpus is deterministic: the same tree yields byte-identical
        # output, which matters when a profile is meant to be reproducible.
        for dp, dn, fn in sorted(os.walk(root)):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(fn):
                if not f.endswith(exts):
                    continue
                p = os.path.join(dp, f)
                try:
                    t = open(p, encoding="utf8", errors="replace").read()
                except OSError:
                    continue
                if len(t) < min_bytes:
                    continue
                out.append("# ==== %s ====\n%s\n" % (os.path.relpath(p, root), t))
                nfiles += 1
                total += len(t)
                if total > max_bytes:
                    return out, nfiles, total
    return out, nfiles, total


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="source tree(s) to walk")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--max-mb", type=float, default=24.0,
                   help="stop after this much text (default 24; 12 windows x 64k tokens "
                        "needs roughly 3 MB, so the default leaves ample spread)")
    p.add_argument("--min-bytes", type=int, default=200,
                   help="skip files shorter than this (default 200)")
    p.add_argument("--ext", default=",".join(DEFAULT_EXTS),
                   help="comma-separated extensions")
    a = p.parse_args()

    exts = tuple(e if e.startswith(".") else "." + e for e in a.ext.split(","))
    chunks, nfiles, total = build(a.roots, exts, int(a.max_mb * 1e6), a.min_bytes)
    txt = "".join(chunks)
    with open(a.out, "w", encoding="utf8") as f:
        f.write(txt)
    print(f" -- wrote {a.out}: {nfiles} files, {len(txt)/1e6:.1f} MB")
    print(f"    next: moe_profile_build.py -corpus {a.out} -nprompts 12 -plen 65536 -gen 192")
