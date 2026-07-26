import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from exllamav3 import Generator, Job, model_init
from exllamav3.util.progress import ProgressBar

"""
Generate a self-sampled, in-domain test trace for qbench: run a set of multi-turn conversations
through a good-precision quant of the target model with default sampling, and store the exact
token ids of every (context, response) pair until at least --min_tokens response tokens have
been collected.

    python eval/qbench_prompts.py -m <model_dir> [model_init options] \
        -o qbench_prompts.json --min_tokens 20000

The point: evaluating quants on external corpora measures divergence on text the model may
never produce itself (for heavily-aligned reasoning models, raw web text is so far out of
distribution that the noise floor inflates and KLD ordering degrades). The model's own sampled
output is in-distribution by definition; a qbench project can point `test_trace` at the output
file and KLD/ppl are then computed only over the sampled (response) token positions, with each
(context, response) pair as one variable-length row.

Reasoning traces are kept in the scored response. For context in follow-up turns the previous
response is cleaned the way deployment would (harmony final channel extracted / <think> blocks
dropped) before re-templating; the stored ids are exactly what the model saw and sampled either
way.
"""

# Seeds chosen to elicit long reasoning traces plus a substantial final answer, across domains.
# Follow-ups deepen the same conversation (self-review prompts produce natural reasoning).
CONVERSATIONS = [
    ("A farmer needs to cross a river with a wolf, a goat, a cabbage, and a second goat that "
     "eats wolves. The boat holds the farmer and two items. Find a safe crossing sequence.",
     ["Double-check your solution step by step and fix any mistakes.",
      "Now solve it again with a boat that holds only one item."]),
    ("Write a Python function that finds the longest palindromic substring in O(n) time, with "
     "an explanation of the algorithm.",
     ["What are the edge cases and how does your implementation handle them?",
      "Convert the implementation to Rust."]),
    ("Estimate how many piano tuners work in Chicago, showing your reasoning.",
     ["Which of your assumptions is most fragile, and how would the estimate change if it's off by 2x?",
      "Do the same estimate for Tokyo."]),
    ("Explain why the sky is blue but sunsets are red, at a level suitable for a curious "
     "12-year-old, then again for a physics undergraduate.",
     ["The undergraduate asks: why isn't the sky violet, since violet light scatters even more?",
      "Write a short quiz testing understanding of both explanations."]),
    ("Plan a 10-day trip through Japan for two people who like hiking and food but hate crowds, "
     "in late November. Include a day-by-day itinerary.",
     ["Adjust the plan for a total budget of $3000 excluding flights.",
      "One traveler sprains an ankle on day 4. Rework the remaining days."]),
    ("Prove that the square root of 2 is irrational, then explain where the same proof breaks "
     "down if you try it on the square root of 4.",
     ["Generalize: for which integers n is sqrt(n) irrational? Prove it.",
      "Explain the proof to someone who has never seen a proof by contradiction."]),
    ("Design a database schema for a public library system: books, copies, members, loans, "
     "holds, fines. Give the DDL and explain the key decisions.",
     ["A branch wants to support inter-library loans. What changes?",
      "Write the five queries the front desk will run most often."]),
    ("Three friends split a restaurant bill of $127.50. Alice pays twice what Bob pays, and Carol "
     "pays $7.50 more than Bob. During payment, a 15% service charge is added to the total. How "
     "much does each person actually pay?",
     ["Verify the arithmetic carefully and present the answer as a table.",
      "Now suppose the service charge only applies to Alice and Carol's shares."]),
    ("Summarize the key arguments for and against rent control, citing the standard economic "
     "models involved, and give your overall assessment.",
     ["Steelman the side your assessment went against.",
      "How does the picture change in a city where new construction is nearly impossible?"]),
    ("Write a 400-word short story about a lighthouse keeper who discovers the light attracts "
     "something other than ships, with a twist ending.",
     ["Critique your own story's pacing and rewrite the weakest paragraph.",
      "Retell the same story as a series of terse logbook entries."]),
]

TEMPLATE_VARS = dict(enable_thinking = True, reasoning_effort = "high")

col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"


def clean_response_for_context(tokenizer, response_ids: torch.Tensor) -> str:
    """Previous-turn assistant content for re-templating: reasoning segments dropped the way a
    deployed chat loop would drop them"""
    full = tokenizer.decode(response_ids, decode_special_tokens = True)
    if "<|channel|>final<|message|>" in full:              # gpt-oss harmony
        text = full.rsplit("<|channel|>final<|message|>", 1)[1]
        for stop in ("<|return|>", "<|end|>", "<|call|>"):
            text = text.split(stop)[0]
        return text.strip()
    plain = tokenizer.decode(response_ids, decode_special_tokens = False)
    if "</think>" in plain:                                # qwen/deepseek style
        plain = plain.rsplit("</think>", 1)[1]
    return plain.strip()


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, *_ = model_init.init(args)
    generator = Generator(model, cache, tokenizer, max_chunk_size = 2048)

    rows = []
    total_in = total_out = 0
    # Turn-major order: every conversation's first turn before any follow-ups, so a small token
    # budget still spreads across all domains instead of exhausting on the first conversations
    convs = [{"messages": [], "turns": [seed] + fu} for seed, fu in CONVERSATIONS]
    max_turns = max(len(c["turns"]) for c in convs)
    schedule = [(c_idx, t_idx) for t_idx in range(max_turns)
                for c_idx in range(len(convs)) if t_idx < len(convs[c_idx]["turns"])]
    with ProgressBar("Sampling", args.min_tokens) as pb:
        for c_idx, t_idx in schedule:
            if total_out >= args.min_tokens:
                break
            conv = convs[c_idx]
            if t_idx > 0 and len(conv["messages"]) < 2 * t_idx:
                continue   # earlier turn of this conversation failed; skip its follow-ups
            messages = conv["messages"]
            user_msg = conv["turns"][t_idx]
            if True:
                messages.append({"role": "user", "content": user_msg})
                input_ids = tokenizer.hf_chat_template(
                    messages, add_generation_prompt = True, **TEMPLATE_VARS)
                job = Job(
                    input_ids = input_ids,
                    max_new_tokens = args.max_new_tokens,
                    stop_conditions = config.eos_token_id_list,
                    decode_special_tokens = True,
                )
                generator.enqueue(job)
                chunks = []
                total_temp = total_out
                text_temp = ""
                while generator.num_remaining_jobs():
                    for result in generator.iterate():
                        if result["stage"] == "streaming" and "token_ids" in result:
                            chunks.append(result["token_ids"])
                            total_temp += result["token_ids"].shape[-1]
                            text = result["text"]
                            text_temp += text
                            if "\n" in text:
                                print(text_temp, end = "")
                                text_temp = ""
                                pb.update(min(total_temp, args.min_tokens))
                response_ids = torch.cat(chunks, dim = -1)[0] if chunks else torch.empty(0, dtype = torch.long)
                if response_ids.numel() == 0:
                    messages.pop()
                    continue
                rows.append({
                    "conversation": c_idx,
                    "turn": t_idx,
                    "input_ids": input_ids[0].tolist(),
                    "response_ids": response_ids.tolist(),
                })
                total_in += input_ids.shape[-1]
                total_out += response_ids.numel()
                pb.update(min(total_out, args.min_tokens))
                messages.append({
                    "role": "assistant",
                    "content": clean_response_for_context(tokenizer, response_ids),
                })
                print(f"\n{col_blue}---------------------------------------------------------------{col_default}\n")
                print(f"Input tokens:  {col_yellow}{total_in:6,}{col_default}")
                print(f"Output tokens: {col_yellow}{total_out:6,}{col_default}")
                print(f"\n{col_blue}---------------------------------------------------------------{col_default}\n")

    out = {
        "model": args.model_dir,
        "vocab_size": tokenizer.actual_vocab_size,
        "template_vars": TEMPLATE_VARS,
        "meta": {"rows": len(rows), "input_tokens": total_in, "output_tokens": total_out},
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f" -- {len(rows)} rows, {total_in:,} input + {total_out:,} output tokens -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 32768)
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output JSON file")
    parser.add_argument("--min_tokens", type = int, default = 20000, help = "Stop once this many response tokens are collected")
    parser.add_argument("--max_new_tokens", type = int, default = 4096, help = "Per-turn generation cap")
    main(parser.parse_args())
