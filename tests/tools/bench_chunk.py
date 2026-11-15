"""Long-prompt tokenization benchmark: time and correctness of the chunked encode path.

    python tests/tools/bench_chunk.py MODEL_DIR... 262144,1048576 [--traps] [--seed S]

Builds a prompt of about the given token count, times `Tokenizer.encode` (the feature under test),
and reports its ids against the single-threaded whole-text encode, which is the oracle: a prompt is
correct when the chunked ids are the whole text's ids, because that is what the prefix cache sees.

By default the prompt is a happy path (prose, code, CRLF and indentation mixed, seed 1); --traps
switches to the directed worst case for the cut-boundary bug class (#564, #599): a whitespace-only
line at every position the splitter searches for a cut, which is where mainline cuts inside a token.
One line of JSON per (model, size).
"""
import argparse, json, os, random, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.architecture import architectures
from exllamav3.tokenizer.tokenizer import Tokenizer


def tokenizer_for(directory: str) -> Tokenizer:
    """The tokenizer of a model directory, with no weights behind it"""
    with open(os.path.join(directory, "config.json"), encoding = "utf8") as f:
        arch = json.load(f)["architectures"][0]
    return Tokenizer(config = architectures.ARCHITECTURES[arch]["config_class"](directory))


WORDS = ["word", "42", "def", "日本", "x", "1234", "end.", "a+b", "tab\there", "return",
         "prompt", "cache", "prefix", "split", "boundary"]
LINES = [
    "def f(x):", "    return x + 1", "", "", "    # a comment", "\t\tindented",
    "| a | b |", "```py", "code = 42", "```", "\n", " \n", "\r\n\r\n", "end of block",
]
TRAPS = ["\n \n", "\n\t\n"]


def sentence(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(8)) + "."


def prompt(rng: random.Random, tok, target_tokens: int, traps: bool) -> str:
    if not traps:
        blocks = ["\n".join(sentence(rng) if rng.random() < 0.5 else rng.choice(LINES)
                             for _ in range(rng.randint(4, 24))) for _ in range(400)]
        sample = "\n\n".join(blocks[:80])
    else:
        blocks = None
        # word runs carry no whitespace, so each cut search lands exactly on its own trap
        sample = "".join(rng.choice(WORDS) for _ in range(4000))
    ratio = len(sample) / len(tok.tokenizer.encode(sample, add_special_tokens = False).ids)
    want = int(target_tokens * ratio)
    out, total, i = [], 0, 0
    while total < want:
        if not traps:
            b = blocks[i % len(blocks)]
        else:
            b = "".join(rng.choice(WORDS) for _ in range(128)) + TRAPS[i % 2]
        out.append(b)
        total += len(b)
        i += 1
    return "".join(out)[:want]


def median(fn, n = 3):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def mismatches(want: list[int], got: list[int]) -> int:
    n = min(len(want), len(got))
    return sum(a != b for a, b in zip(want, got)) + abs(len(want) - len(got))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("model_dir", nargs = "+")
    ap.add_argument("tokens", help = "comma-separated prompt sizes in tokens, e.g. 262144,1048576")
    ap.add_argument("--traps", action = "store_true", help = "directed worst case, not the happy path")
    ap.add_argument("--seed", type = int, default = 1, help = "prompt seed (default 1)")
    args = ap.parse_args(argv)
    sizes = [int(t) for t in args.tokens.split(",")]
    for directory in args.model_dir:
        tok = tokenizer_for(directory)
        rng = random.Random(args.seed)
        for target in sizes:
            text = prompt(rng, tok, target, args.traps)
            whole = tok.tokenizer.encode(text, add_special_tokens = False).ids
            t_whole = median(lambda: tok.tokenizer.encode(text, add_special_tokens = False))
            t_chunk = median(lambda: tok.encode(text, encode_special_tokens = False))
            chunked = tok.encode(text, encode_special_tokens = False)[0].tolist()
            print(json.dumps({
                "model": os.path.basename(directory),
                "traps": args.traps,
                "tokens": len(whole),
                "chars": len(text),
                "whole_ms": round(t_whole),
                "chunk_ms": round(t_chunk),
                "mismatches": mismatches(whole, chunked),
            }), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))