"""The long-input path cuts a prompt into chunks and encodes them as one parallel batch; the chunks must
still encode to the whole text's ids, or a growing prompt reuses none of its cached prefix. Which
positions may be cut is a property of the tokenizer: a run of newlines is a boundary on most
pre-tokenizers, yet a tokenizer whose pieces span whitespace fuses a whitespace-only line into one
token. Issues #564 and #599 are two whitespace runs the splitter used to cut inside, both found in the
field; this walks a whitespace run across the positions the splitter can cut at, so the next quirk of
that class is a failing check instead of a slow session. The special-token mode is covered by
tests/tokenizer/test_chunking.py, not here.

    python tests/tools/chunk_cut_fuzz.py MODEL_DIR... [--rounds N] [--seed S] [--chars N]

MODEL_DIR is the directory of a model (config.json, tokenizer.json and tokenizer_config.json); the model
itself is not loaded. Exits non-zero if any case disagrees.
"""
import argparse, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.model.config import Config
from exllamav3.tokenizer.tokenizer import Tokenizer, _CHUNK_MIN_CHARS, _CHUNK_MAX, _CHUNK_TARGET_CHARS

# Whitespace runs to slide across the cut positions, the two repros' kinds first
WHITESPACE = ["\n \n", "\n\t\n", "\n\n", "\r\n\r\n", "\n  \n", "\n \t \n", "\n\r\n", "\n\r \n", "\n\n\t",
              "\t\n \n", "\n\u3000\n", "\n\u00a0\n", "\n\f\n", "\n\v\n", "\n\n\n \n", "\n \n \n"]
FILLER = ["word", "42", "def", "日本", "x", "1234", "end.", "a+b", "tab\there"]


def tokenizer_for(directory: str) -> Tokenizer:
    """The tokenizer of a model directory, with no weights behind it"""
    return Tokenizer(config = Config.from_directory(directory))


def filler(rng: random.Random, chars: int) -> str:
    """Random filler text; it contains no newlines (so only the slid run offers a cut) and no entry
    ends in whitespace, so the run's boundaries sit exactly where placed, except the final truncation
    may cut right after the tab in "tab\there", leaving head ending in one that pulls the run's start
    candidate one char earlier"""
    out, total = [], 0
    while total < chars:
        entry = rng.choice(FILLER)
        out.append(entry)
        total += len(entry)
    return "".join(out)[:chars]


def cases(chars: int, rounds: int, rng: random.Random):
    """(name, text) with a whitespace run at each position the splitter can cut at: the splitter
    looks for the first newline at or past `start + n // k`, so the no-tail cases put the run's
    last newline exactly there, while the tailed cases put the run just past it, so the search
    lands on the run's first newline"""
    yield "issue599 space", "x" * 20_000 + "\n \n" + "y" * 19_997
    yield "issue599 tab", "x" * 20_000 + "\n\t\n" + "y" * 19_997
    for ws in WHITESPACE:
        last_nl = len(ws) - 1 - ws[::-1].find("\n")
        for tail in ("", " and then a bit more text"):
            n = chars + len(ws) + len(tail)
            k = min(_CHUNK_MAX, max(1, n // _CHUNK_TARGET_CHARS))
            step, start = n // k, 0
            for _ in range(k - 1):
                p = start + step - last_nl
                head = filler(rng, p)
                rest = filler(rng, n - p - len(ws) - len(tail))
                yield f"slide {ws!r} @{p + len(tail)}{' tail' if tail else ''}", head + tail + ws + rest
                start = p + len(tail) + len(ws)
    for r in range(rounds):
        # whitespace soup: the runs are everywhere, so the cut lands wherever it lands
        parts, total = [], 0
        while total < chars:
            entry = rng.choice(FILLER)
            parts.append(entry)
            total += len(entry)
            entry = rng.choice(WHITESPACE)
            parts.append(entry)
            total += len(entry)
        yield f"random {r}", "".join(parts)


def first_diff(want: list[int], got: list[int]) -> int:
    return next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(map(len, (want, got))))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("model_dir", nargs = "+")
    ap.add_argument("--rounds", type = int, default = 16, help = "randomised prompts per model (default 16)")
    ap.add_argument("--seed", type = int, default = 1, help = "corpus seed (default 1)")
    ap.add_argument("--chars", type = int, default = 2 * _CHUNK_TARGET_CHARS,
                    help = "prompt size, past the chunking threshold (default 2 targets)")
    args = ap.parse_args(argv)
    if args.chars < _CHUNK_MIN_CHARS:
        ap.error(f"--chars {args.chars} is too small: generated texts can be as short as "
                 f"{args.chars} chars, below the chunking threshold ({_CHUNK_MIN_CHARS})")
    failed = 0
    for directory in args.model_dir:
        tokenizer = tokenizer_for(directory)
        rng = random.Random(args.seed)
        bad = 0
        for name, text in cases(args.chars, args.rounds, rng):
            want = tokenizer.tokenizer.encode(text, add_special_tokens = False).ids
            got = tokenizer.encode(text, encode_special_tokens = False)[0].tolist()
            if got != want:
                bad += 1
                i = first_diff(want, got)
                print(f"  {name:34} len={len(text):8} whole={len(want):7} chunked={len(got):7} "
                      f"first diff at id {i}: {want[i:i + 4]} != {got[i:i + 4]}")
        failed += bad
        print(f"{directory}: {'FAIL' if bad else 'ok'} ({bad} of the corpus disagree)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
