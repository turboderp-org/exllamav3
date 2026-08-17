import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from exllamav3 import Config, Tokenizer

"""
Decode a qbench_prompts.py trace back to human-readable text: every stored (context, response)
token-id pair is decoded with special tokens visible and written to a structured Markdown file,
one section per row.

    python eval/qbench_prompts_decode.py qbench_prompts.json -o qbench_prompts.md [-m model_dir]

Only the tokenizer is loaded, taken from the model directory recorded in the trace unless -m
overrides it. Fence lengths adapt to the decoded text, so responses containing Markdown code
blocks stay intact and the output remains mechanically parseable: rows are `## Row N`
headings, each with `### Input` and `### Response` subsections wrapping a single fenced block.
"""


def fence_for(*texts: str) -> str:
    """A backtick fence strictly longer than any backtick run in the given texts"""
    longest = 2
    for text in texts:
        run = 0
        for ch in text:
            run = run + 1 if ch == "`" else 0
            longest = max(longest, run)
    return "`" * (longest + 1)


def main(args):
    with open(args.trace, "r") as f:
        data = json.load(f)

    model_dir = args.model_dir or data.get("model")
    assert model_dir, "Trace has no model path; specify one with -m"
    tokenizer = Tokenizer.from_config(Config.from_directory(model_dir))

    rows = data["rows"]
    meta = data.get("meta", {})
    lines = [
        "# qbench prompt trace",
        "",
        f"- source: `{os.path.abspath(args.trace)}`",
        f"- model: `{data.get('model', '(unknown)')}`",
        f"- tokenizer: `{model_dir}`",
        f"- vocab size: {data.get('vocab_size', '(unknown)')}",
        f"- template vars: `{json.dumps(data.get('template_vars', {}))}`",
        f"- rows: {meta.get('rows', len(rows))}",
        f"- input tokens: {meta.get('input_tokens', '(unknown)')}",
        f"- output tokens: {meta.get('output_tokens', '(unknown)')}",
        "",
    ]

    for i, row in enumerate(rows):
        input_ids = torch.tensor(row["input_ids"], dtype = torch.long)
        response_ids = torch.tensor(row["response_ids"], dtype = torch.long)
        input_text = tokenizer.decode(input_ids, decode_special_tokens = True)
        response_text = tokenizer.decode(response_ids, decode_special_tokens = True)
        f = fence_for(input_text, response_text)
        lines += [
            f"## Row {i} (conversation {row['conversation']}, turn {row['turn']})",
            "",
            f"### Input ({input_ids.numel()} tokens)",
            "",
            f + "text",
            input_text,
            f,
            "",
            f"### Response ({response_ids.numel()} tokens)",
            "",
            f + "text",
            response_text,
            f,
            "",
        ]

    with open(args.output, "w", encoding = "utf8") as f:
        f.write("\n".join(lines))
    print(f" -- {len(rows)} rows -> {args.output} ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("trace", type = str, help = "Trace JSON produced by qbench_prompts.py")
    parser.add_argument("-m", "--model_dir", type = str, default = None, help = "Model directory for the tokenizer, default: the model path stored in the trace")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output Markdown file")
    main(parser.parse_args())
