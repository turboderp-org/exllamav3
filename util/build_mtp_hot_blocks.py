import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from collections import Counter
from pathlib import Path

from exllamav3 import Config, Tokenizer


BLOCK_SIZE = 16
GROUP_SIZE = 128
BLOCKS_PER_GROUP = GROUP_SIZE // BLOCK_SIZE
EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
    ".go", ".java", ".js", ".json", ".jsx", ".md", ".py",
    ".rs", ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def main(args, parser):

    if args.blocks <= 0 or args.blocks % BLOCKS_PER_GROUP:
        parser.error(f"--blocks must be a positive multiple of {BLOCKS_PER_GROUP}")

    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    counts = Counter()
    files = 0
    tokens = 0

    for root_name in args.corpus:
        root = Path(root_name)
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            text = path.read_text(encoding = "utf-8", errors = "ignore")[:args.max_chars_per_file]
            if not text:
                continue
            ids = tokenizer.encode(text, add_bos = False)[0].tolist()
            counts.update(token_id // GROUP_SIZE for token_id in ids)
            files += 1
            tokens += len(ids)

    required_ids = set(config.eos_token_id_list)
    required_ids.update(x for x in (config.bos_token_id, config.pad_token_id) if x is not None)
    draft_counts = Counter()
    for filename in args.draft_ids:
        with open(filename, "r", encoding = "utf-8") as f:
            ids = [int(line) for line in f if line.strip() and not line.lstrip().startswith("#")]
        if any(token_id < 0 or token_id >= config.vocab_size for token_id in ids):
            parser.error(f"Draft ID outside the model vocabulary in {filename}")
        draft_counts.update(token_id // GROUP_SIZE for token_id in ids)

    num_groups = args.blocks // BLOCKS_PER_GROUP
    required_groups = {token_id // GROUP_SIZE for token_id in required_ids}
    if num_groups < len(required_groups):
        parser.error(f"--blocks must provide room for all {len(required_groups)} special-token groups")

    selected_groups = set(required_groups)
    for frequencies in (draft_counts, counts):
        for group, _ in frequencies.most_common():
            if len(selected_groups) >= num_groups:
                break
            selected_groups.add(group)
    if len(selected_groups) < num_groups:
        for group in range((config.vocab_size + GROUP_SIZE - 1) // GROUP_SIZE):
            if len(selected_groups) >= num_groups:
                break
            selected_groups.add(group)

    selected_groups = sorted(selected_groups)
    selected = [
        group * BLOCKS_PER_GROUP + offset
        for group in selected_groups
        for offset in range(BLOCKS_PER_GROUP)
    ]
    covered = sum(count for group, count in counts.items() if group in selected_groups)
    output = Path(args.output)
    output.write_text(
        "# EXL3 MTP hot vocabulary: packed 16-token block IDs\n"
        f"# corpus_files={files} corpus_tokens={tokens} blocks={len(selected)} "
        f"corpus_coverage={covered / max(tokens, 1):.8f}\n" +
        "\n".join(map(str, selected)) + "\n",
        encoding = "utf-8",
    )
    print({
        "files": files,
        "tokens": tokens,
        "blocks": len(selected),
        "tokens_in_head": len(selected) * BLOCK_SIZE,
        "corpus_coverage": covered / max(tokens, 1),
        "output": str(output),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        allow_abbrev = False,
        description = "Build an aligned EXL3 vocabulary-subset map for Qwen MTP drafting.",
    )
    parser.add_argument("-m", "--model", required = True, help = "EXL3 model directory")
    parser.add_argument("-c", "--corpus", action = "append", required = True, help = "Representative file or directory (repeatable)")
    parser.add_argument("-b", "--blocks", type = int, default = 4096, help = "Packed 16-token blocks (default: 4096 = 65,536 tokens)")
    parser.add_argument("-o", "--output", required = True, help = "Output block-map path")
    parser.add_argument("--max_chars_per_file", type = int, default = 1_000_000)
    parser.add_argument(
        "--draft_ids", action = "append", default = [],
        help = "Observed unrestricted MTP proposal IDs; their groups are selected first",
    )
    _args = parser.parse_args()
    main(_args, parser)
