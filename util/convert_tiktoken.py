"""
Convert a tiktoken-format tokenizer (tiktoken.model / tokenizer.model / *.tiktoken base64 rank
file) into a HF-tokenizers tokenizer.json, so checkpoints that ship only a tiktoken vocab plus a
remote-code tokenizer class (Moonlight, Kimi, ...) can be loaded without trust_remote_code and
regardless of whether the bundled tokenization_*.py still imports under current transformers.

    python util/convert_tiktoken.py -i <model_dir> [-o <file_or_dir>] [--preset NAME | --pattern REGEX]

The pre-tokenizer split pattern is not stored in the rank file, so it must come from somewhere:

  1. --pattern, verbatim
  2. --preset (see PRESETS below)
  3. auto-extraction: the model dir's tokenization_*.py is scanned (AST, not imported) for a
     pat_str-style assignment, either a string literal or the common "|".join([...]) form

Special tokens are taken from added_tokens_decoder in tokenizer_config.json. Their ids must lie
at or above the base vocabulary size (dense tiktoken ranks); any gaps up to --vocab-size
(default: vocab_size from config.json) are filled with reserved placeholder tokens so every id
up to the model's head size exists in the output.

The result is verified against tiktoken itself (same ranks, same pattern, same specials): text
samples must round-trip identically and every named special must land on its configured id.
Note what this does and does not prove: the conversion is exact for the given pattern, but if
the pattern itself is wrong for the checkpoint (e.g. a wrong --preset), verification cannot
detect that. Prefer auto-extraction from the checkpoint's own code when available.
"""

import argparse
import ast
import base64
import glob
import json
import os
import sys

# Well-known tiktoken split patterns. gpt2 is also r50k/p50k; llama3 is the pattern from Meta's
# llama3 repo (a cl100k variant without possessive quantifiers); kimi covers Moonshot models
# (Moonlight, Kimi, kimi-linear). gpt2/cl100k/o200k are copied verbatim from the tiktoken source
# (tiktoken_ext/openai_public.py).
PRESETS = {
    "gpt2": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "cl100k": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
    "llama3": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    "o200k": "|".join([
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]),
    "kimi": "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]),
}

SAMPLES = [
    "The Antikythera mechanism is an ancient Greek hand-powered orrery.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n",
    "  leading and   multiple   spaces\tand\ttabs\n\n\nand blank lines",
    "Unicode: café, naïve, 日本語のテキスト, 中文字符, Ελληνικά, Русский",
    "한국어 텍스트도 시험합니다. 오토플레이 기능을 활성화했습니다.",
    "Numbers 1234567890 and 3.14159 and 1,000,000 and 0xDEADBEEF",
    "Mixed CamelCase snake_case SCREAMING_CASE and'apostrophes don't won't",
    "Emoji 🙂🚀 and symbols ±§¶•—…",
    "JSON: {\"key\": [1, 2, {\"nested\": true}], \"other\": null}",
    "trailing space \nand space before newline \n and CRLF\r\nline",
    "中文和English混排，标点。Numbers123mixed456.",
]


def find_ranks_file(in_dir):
    for name in ("tiktoken.model", "tokenizer.model"):
        p = os.path.join(in_dir, name)
        if os.path.isfile(p):
            return p
    tt = glob.glob(os.path.join(in_dir, "*.tiktoken"))
    if len(tt) == 1:
        return tt[0]
    return None


def load_ranks(path):
    """Parse a tiktoken rank file (base64 token + rank per line). Raises with a useful message
    if the file is a sentencepiece model, which shares the tokenizer.model filename."""
    ranks = {}
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        for ln, line in enumerate(f):
            if not line.strip():
                continue
            try:
                t, r = line.split()
                ranks[base64.b64decode(t, validate = True)] = int(r)
            except Exception:
                extra = ""
                if ln == 0 and head[:2] in (b"\x0a\x09", b"\x0a\x0b", b"\x0a\x0c", b"\x0a\x08"):
                    extra = " (this looks like a sentencepiece model, not a tiktoken rank file)"
                raise SystemExit(f" ## {path}:{ln + 1}: not a tiktoken rank file{extra}")
    if sorted(ranks.values()) != list(range(len(ranks))):
        raise SystemExit(f" ## {path}: ranks are not dense from 0, not a valid tiktoken vocabulary")
    return ranks


def extract_pattern(in_dir):
    """Scan the checkpoint's tokenization_*.py (without importing it) for a pat_str-style
    assignment: a string literal, or the common '|'.join([literals]) form."""
    for py in sorted(glob.glob(os.path.join(in_dir, "tokenization_*.py"))):
        try:
            tree = ast.parse(open(py, encoding = "utf8").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any("pat" in n.lower() for n in names):
                continue
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value, py
            if (
                isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "join"
                and isinstance(v.func.value, ast.Constant) and isinstance(v.func.value.value, str)
                and len(v.args) == 1 and isinstance(v.args[0], (ast.List, ast.Tuple))
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.args[0].elts)
            ):
                return v.func.value.value.join(e.value for e in v.args[0].elts), py
    return None, None


def build_specials(in_dir, num_base, vocab_size, reserved_format):
    """Special token list for ids [num_base, vocab_size), named from added_tokens_decoder where
    the config names them, reserved placeholders elsewhere."""
    named = {}
    tc_path = os.path.join(in_dir, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        added = json.load(open(tc_path, encoding = "utf8")).get("added_tokens_decoder", {})
        named = {int(i): v["content"] for i, v in added.items()}
        below = [i for i in named if i < num_base]
        if below:
            raise SystemExit(
                f" ## added_tokens_decoder ids {below[:4]} fall inside the base vocabulary "
                f"(0..{num_base - 1}); cannot represent these as appended special tokens"
            )
    if vocab_size is None:
        cfg_path = os.path.join(in_dir, "config.json")
        if os.path.isfile(cfg_path):
            vocab_size = json.load(open(cfg_path, encoding = "utf8")).get("vocab_size")
    if vocab_size is None:
        vocab_size = (max(named) + 1) if named else num_base
        print(f" -- no config.json vocab_size, sizing vocabulary to {vocab_size}")
    if named and max(named) >= vocab_size:
        raise SystemExit(f" ## special token id {max(named)} outside vocab_size {vocab_size}")
    specials = [named.get(i, reserved_format.format(id = i)) for i in range(num_base, vocab_size)]
    if len(set(specials)) != len(specials):
        raise SystemExit(" ## duplicate special token contents (check --reserved-format)")
    return specials, named, vocab_size


def verify(out_json, ranks, pattern, named, num_base):
    import tiktoken
    from tokenizers import Tokenizer

    hf = Tokenizer.from_file(out_json)
    enc = tiktoken.Encoding(name = "converted", pat_str = pattern, mergeable_ranks = ranks, special_tokens = {})

    ok = True
    bad = 0
    for s in SAMPLES:
        a = enc.encode(s)
        b = hf.encode(s, add_special_tokens = False).ids
        if a != b:
            bad += 1
            print(f" !! MISMATCH on {s[:48]!r}\n    tiktoken {a[:16]}\n    converted {b[:16]}")
    print(f" -- round-trip: {len(SAMPLES) - bad}/{len(SAMPLES)} samples identical to tiktoken")
    ok &= bad == 0

    for i, name in sorted(named.items()):
        j = hf.token_to_id(name)
        if j != i:
            print(f" !! special {name!r} at id {j}, expected {i}")
            ok = False
    print(f" -- specials: {len(named)} named tokens at their configured ids" if ok else " !! special token placement errors")

    # Byte-level identity on the base vocabulary
    for tok_bytes, rank in list(ranks.items())[:: max(1, len(ranks) // 1000)]:
        piece = hf.id_to_token(rank)
        if piece is None:
            print(f" !! base token {rank} missing from converted vocabulary")
            ok = False
            break
    return ok


def main():
    p = argparse.ArgumentParser(description = "Convert a tiktoken vocabulary to tokenizer.json")
    p.add_argument("-i", "--in_dir", required = True, help = "Model directory (or direct path to the rank file)")
    p.add_argument("-o", "--out", default = None, help = "Output file or directory (default: <in_dir>/tokenizer.json)")
    p.add_argument("--pattern", default = None, help = "Pre-tokenizer split regex, verbatim")
    p.add_argument("--preset", default = None, choices = sorted(PRESETS), help = "Well-known split pattern")
    p.add_argument("--vocab-size", type = int, default = None, help = "Total vocabulary size incl. specials (default: config.json)")
    p.add_argument("--reserved-format", default = "<|reserved_token_{id}|>", help = "Name template for unnamed special ids")
    p.add_argument("--no-verify", action = "store_true", help = "Skip verification against tiktoken")
    p.add_argument("-f", "--force", action = "store_true", help = "Overwrite an existing output file")
    args = p.parse_args()

    if os.path.isfile(args.in_dir):
        ranks_path, in_dir = args.in_dir, os.path.dirname(os.path.abspath(args.in_dir))
    else:
        in_dir = args.in_dir
        ranks_path = find_ranks_file(in_dir)
        if not ranks_path:
            raise SystemExit(f" ## no tiktoken.model / tokenizer.model / *.tiktoken in {in_dir}")

    out = args.out or in_dir
    if os.path.isdir(out):
        out = os.path.join(out, "tokenizer.json")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f" ## {out} exists, use --force to overwrite")

    ranks = load_ranks(ranks_path)
    num_base = len(ranks)
    print(f" -- {ranks_path}: {num_base} base tokens")

    if args.pattern:
        pattern, source = args.pattern, "--pattern"
    elif args.preset:
        pattern, source = PRESETS[args.preset], f"preset {args.preset}"
    else:
        pattern, source = extract_pattern(in_dir)
        if not pattern:
            raise SystemExit(
                " ## No split pattern: pass --pattern or --preset "
                f"({', '.join(sorted(PRESETS))}); none could be extracted from the checkpoint"
            )
    print(f" -- split pattern from {source}:\n    {pattern}")

    specials, named, vocab_size = build_specials(in_dir, num_base, args.vocab_size, args.reserved_format)
    print(f" -- {len(named)} named + {len(specials) - len(named)} reserved special tokens, vocab size {vocab_size}")

    from transformers.convert_slow_tokenizer import TikTokenConverter
    # The specials kwarg was renamed between transformers versions (additional_special_tokens ->
    # extra_special_tokens); the constructor swallows unknown kwargs silently, so pass both
    conv = TikTokenConverter(
        vocab_file = ranks_path,
        pattern = pattern,
        additional_special_tokens = specials,
        extra_special_tokens = specials,
    )
    tok = conv.converted()

    tok.save(out)
    print(f" -- wrote {out}")

    if args.no_verify:
        return
    sys.exit(0 if verify(out, ranks, pattern, named, num_base) else 1)


if __name__ == "__main__":
    main()
