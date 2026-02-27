import argparse
import os
import re
import sys
from dataclasses import dataclass

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init


@dataclass
class PromptCase:
    name: str
    prompt: str
    expect_hangul: bool
    expect_latin: bool


PROMPTS = [
    PromptCase(
        name="english_reasoning",
        prompt=(
            "Answer in English only. Explain briefly why the sky appears blue, "
            "then give one everyday example."
        ),
        expect_hangul=False,
        expect_latin=True,
    ),
    PromptCase(
        name="korean_reasoning",
        prompt=(
            "한국어로만 답하세요. 하늘이 파란 이유를 간단히 설명하고 "
            "일상 예시를 한 가지 들어주세요."
        ),
        expect_hangul=True,
        expect_latin=False,
    ),
    PromptCase(
        name="korean_math",
        prompt=(
            "한국어로 풀이를 보여주고 마지막 줄에는 답만 적어주세요. "
            "문제: 54*33"
        ),
        expect_hangul=True,
        expect_latin=False,
    ),
]


def _decode_one(tokenizer, ids: torch.Tensor) -> str:
    out = tokenizer.decode(ids, decode_special_tokens=False)
    if isinstance(out, list):
        return out[0]
    return out


@torch.inference_mode()
def generate_greedy(
    model,
    cache,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    use_decode_wrapper: bool,
):
    input_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    if input_ids.shape[-1] < 2:
        raise RuntimeError("Prompt is too short after tokenization")

    prefill_params = {
        "attn_mode": "flashinfer",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, cache.max_num_tokens),
        "flashinfer_use_decode_wrapper": use_decode_wrapper,
    }
    model.prefill(
        input_ids=input_ids[:, :-1],
        params=prefill_params,
    )

    cur = input_ids[:, -1:]
    gen_ids = []
    recurrent_states = prefill_params.get("recurrent_states")

    for i in range(max_new_tokens):
        params = {
            "attn_mode": "flashinfer",
            "cache": cache,
            "past_len": input_ids.shape[-1] - 1 + i,
            "batch_shape": (1, cache.max_num_tokens),
            "flashinfer_use_decode_wrapper": use_decode_wrapper,
            "recurrent_states": recurrent_states,
        }
        logits = model.forward(cur, params)
        recurrent_states = params.get("recurrent_states")
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_id.item()
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        gen_ids.append(token_id)
        cur = next_id

    if len(gen_ids) == 0:
        return "", 0
    gen_tensor = torch.tensor([gen_ids], dtype=torch.long)
    text = _decode_one(tokenizer, gen_tensor)
    return text, len(gen_ids)


def validate_text(text: str, expect_hangul: bool, expect_latin: bool):
    issues = []
    if "\ufffd" in text:
        issues.append("contains_unicode_replacement_char")
    if expect_hangul and re.search(r"[가-힣]", text) is None:
        issues.append("missing_hangul")
    if expect_latin and re.search(r"[A-Za-z]", text) is None:
        issues.append("missing_latin")
    return issues


def main():
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, default_cache_size=4096)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--mode",
        choices=["on", "off", "both"],
        default="both",
        help="decode wrapper usage",
    )
    args = parser.parse_args()

    model, config, cache, tokenizer = model_init.init(
        args,
        load_tokenizer=True,
        quiet=True,
        progress=False,
        max_chunk_size=4096,
    )

    modes = [True, False] if args.mode == "both" else [args.mode == "on"]
    has_failures = False

    for case in PROMPTS:
        print(f"\n=== {case.name} ===")
        print(f"PROMPT: {case.prompt}")
        for use_decode_wrapper in modes:
            label = "decode_wrapper_on" if use_decode_wrapper else "decode_wrapper_off"
            text, n_tokens = generate_greedy(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                prompt=case.prompt,
                max_new_tokens=args.max_new_tokens,
                use_decode_wrapper=use_decode_wrapper,
            )
            issues = validate_text(text, case.expect_hangul, case.expect_latin)
            status = "PASS" if len(issues) == 0 else "FAIL"
            if issues:
                has_failures = True
            print(f"[{label}] {status} tokens={n_tokens} issues={issues}")
            print(f"[{label}] OUTPUT: {text}\n")

    if has_failures:
        raise SystemExit(2)

    print("All multilingual smoke checks passed.")


if __name__ == "__main__":
    main()
