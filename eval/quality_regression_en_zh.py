import argparse
import os
import re
import statistics
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init


@dataclass
class Case:
    name: str
    lang: str  # "en" | "zh"
    prompt: str
    max_new_tokens: int = 96


CASES: list[Case] = [
    Case(
        name="en_reasoning_science",
        lang="en",
        prompt=(
            "Answer in English only. Explain why the sky looks blue in 3-4 sentences, "
            "then give one everyday example."
        ),
    ),
    Case(
        name="en_math_word_problem",
        lang="en",
        prompt=(
            "Answer in English only. A shop sells pencils at $2 each. "
            "If I buy 17 pencils and pay with $50, show the steps and final change."
        ),
    ),
    Case(
        name="en_instruction_json",
        lang="en",
        prompt=(
            "Return ONLY valid JSON with keys summary and risks. "
            "Text: The team migrated the API gateway and reduced timeout errors, "
            "but monitoring coverage is still incomplete."
        ),
    ),
    Case(
        name="en_logic",
        lang="en",
        prompt=(
            "Answer in English only. If all roses are flowers and some flowers fade quickly, "
            "what can and cannot be concluded about roses? Keep it concise."
        ),
    ),
    Case(
        name="en_short_translation",
        lang="en",
        prompt=(
            "Translate this to natural English and explain one key nuance: "
            "“先把问题拆开，再逐步验证。”"
        ),
    ),
    Case(
        name="zh_reasoning_science",
        lang="zh",
        prompt=(
            "请只用中文回答：简要解释为什么天空看起来是蓝色的，"
            "并给出一个日常生活中的例子。"
        ),
    ),
    Case(
        name="zh_math_steps",
        lang="zh",
        prompt="请用中文给出简要过程，最后一行只写答案：54*33",
    ),
    Case(
        name="zh_instruction_json",
        lang="zh",
        prompt=(
            "只输出合法JSON，键为“总结”和“风险”。"
            "文本：团队迁移了API网关，超时错误减少，但监控覆盖仍不完整。"
        ),
    ),
    Case(
        name="zh_logic",
        lang="zh",
        prompt=(
            "请只用中文回答：已知“所有A都是B，部分B是C”，"
            "分别说明能推出和不能推出的结论。"
        ),
    ),
    Case(
        name="zh_rewrite",
        lang="zh",
        prompt=(
            "请将下面这句话改写成更正式的中文，保持原意："
            "“先把问题拆开，再逐步验证。”"
        ),
    ),
]


def _build_args(model_dir: str, cache_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_dir=model_dir,
        gpu_split=None,
        load_metrics=False,
        override=None,
        tensor_parallel=False,
        tp_backend="native",
        tp_max_parallelism_attn=None,
        tp_max_parallelism_mlp=None,
        tp_max_parallelism_moe=None,
        tp_max_parallelism_linear=None,
        tp_moe_tensor_split=False,
        load_verbose=False,
        cache_size=cache_size,
        cache_quant=None,
    )


def decode_one(tokenizer, ids: torch.Tensor) -> str:
    text = tokenizer.decode(ids, decode_special_tokens=False)
    if isinstance(text, list):
        return text[0]
    return text


@torch.inference_mode()
def generate_greedy(model, cache, tokenizer, prompt: str, max_new_tokens: int):
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
    }
    model.prefill(
        input_ids=input_ids[:, :-1],
        params=prefill_params,
    )

    cur = input_ids[:, -1:]
    gen_ids: list[int] = []
    recurrent_states = prefill_params.get("recurrent_states")
    for i in range(max_new_tokens):
        params = {
            "attn_mode": "flashinfer",
            "cache": cache,
            "past_len": input_ids.shape[-1] - 1 + i,
            "batch_shape": (1, cache.max_num_tokens),
            "recurrent_states": recurrent_states,
        }
        logits = model.forward(cur, params)
        recurrent_states = params.get("recurrent_states")
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tid = next_id.item()
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break
        gen_ids.append(tid)
        cur = next_id

    if len(gen_ids) == 0:
        return input_ids, [], "", 0
    gen_tensor = torch.tensor([gen_ids], dtype=torch.long)
    text = decode_one(tokenizer, gen_tensor)
    return input_ids, gen_ids, text, len(gen_ids)


@torch.inference_mode()
def score_continuation_nll(model, cache, prompt_ids: torch.Tensor, continuation_ids: list[int]) -> float:
    if len(continuation_ids) == 0:
        return float("nan")

    prefill_params = {
        "attn_mode": "flashinfer",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, cache.max_num_tokens),
    }
    model.prefill(
        input_ids=prompt_ids[:, :-1],
        params=prefill_params,
    )

    past_len = prompt_ids.shape[-1] - 1
    cur = prompt_ids[:, -1:]
    nll_sum = 0.0
    recurrent_states = prefill_params.get("recurrent_states")

    for tid in continuation_ids:
        params = {
            "attn_mode": "flashinfer",
            "cache": cache,
            "past_len": past_len,
            "batch_shape": (1, cache.max_num_tokens),
            "recurrent_states": recurrent_states,
        }
        logits = model.forward(cur, params)
        recurrent_states = params.get("recurrent_states")
        logp = F.log_softmax(logits[:, -1, :], dim=-1)[0, tid]
        nll_sum += float(-logp)
        cur = torch.tensor([[tid]], dtype=torch.long)
        past_len += 1

    return nll_sum / len(continuation_ids)


def token_match_ratio(a: list[int], b: list[int]) -> float:
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    same = sum(1 for i in range(m) if a[i] == b[i])
    return same / m


def prefix_match_len(a: list[int], b: list[int]) -> int:
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m


def replacement_char_count(text: str) -> int:
    return text.count("\ufffd")


def lang_presence_score(text: str, lang: str) -> float:
    if not text:
        return 0.0
    if lang == "en":
        alpha = len(re.findall(r"[A-Za-z]", text))
        return alpha / max(len(text), 1)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk / max(len(text), 1)


def repetition_score(text: str) -> float:
    toks = text.split()
    if len(toks) < 6:
        return 0.0
    tri = {}
    for i in range(len(toks) - 2):
        key = (toks[i], toks[i + 1], toks[i + 2])
        tri[key] = tri.get(key, 0) + 1
    if not tri:
        return 0.0
    return max(tri.values()) / max(len(toks) - 2, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_model_dir", type=str, required=True)
    parser.add_argument("--quant_model_dir", type=str, required=True)
    parser.add_argument("--cache_size", type=int, default=4096)
    parser.add_argument("--max_cases", type=int, default=10)
    args = parser.parse_args()

    selected_cases = CASES[: args.max_cases]
    results_ref = {}
    results_quant = {}

    print(f"REF MODEL:   {args.ref_model_dir}")
    print(f"QUANT MODEL: {args.quant_model_dir}")
    print(f"CASES:       {len(selected_cases)}")
    print("")

    # Pass 1: reference model (generation + self NLL on its continuation)
    model, config, cache, tokenizer = model_init.init(
        _build_args(args.ref_model_dir, args.cache_size),
        load_tokenizer=True,
        quiet=True,
        progress=False,
        max_chunk_size=4096,
    )
    for case in selected_cases:
        prompt_ids, gen_ids, text, gen_len = generate_greedy(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            prompt=case.prompt,
            max_new_tokens=case.max_new_tokens,
        )
        nll_ref = score_continuation_nll(model, cache, prompt_ids, gen_ids)
        results_ref[case.name] = {
            "case": case,
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "text": text,
            "gen_len": gen_len,
            "nll_ref": nll_ref,
        }
        print(f"[REF:{case.name}] gen_len={gen_len} nll={nll_ref:.4f}")
    model.unload()

    print("")

    # Pass 2: quant model (generation + NLL on reference continuation)
    model_q, config_q, cache_q, tokenizer_q = model_init.init(
        _build_args(args.quant_model_dir, args.cache_size),
        load_tokenizer=True,
        quiet=True,
        progress=False,
        max_chunk_size=4096,
    )
    for case in selected_cases:
        ref = results_ref[case.name]
        prompt_ids_q, gen_ids_q, text_q, gen_len_q = generate_greedy(
            model=model_q,
            cache=cache_q,
            tokenizer=tokenizer_q,
            prompt=case.prompt,
            max_new_tokens=case.max_new_tokens,
        )
        nll_q_on_ref = score_continuation_nll(
            model_q,
            cache_q,
            prompt_ids_q,
            ref["gen_ids"],
        )
        results_quant[case.name] = {
            "gen_ids": gen_ids_q,
            "text": text_q,
            "gen_len": gen_len_q,
            "nll_q_on_ref": nll_q_on_ref,
        }
        print(f"[QNT:{case.name}] gen_len={gen_len_q} nll_on_ref={nll_q_on_ref:.4f}")
    model_q.unload()

    print("\n=== Detailed Report ===")
    all_match = []
    all_delta_nll = []
    all_ref_repl = []
    all_q_repl = []
    all_lang_ref = []
    all_lang_q = []
    all_rep_ref = []
    all_rep_q = []

    for case in selected_cases:
        ref = results_ref[case.name]
        qnt = results_quant[case.name]
        ratio = token_match_ratio(ref["gen_ids"], qnt["gen_ids"])
        pml = prefix_match_len(ref["gen_ids"], qnt["gen_ids"])
        repl_ref = replacement_char_count(ref["text"])
        repl_q = replacement_char_count(qnt["text"])
        lang_ref = lang_presence_score(ref["text"], case.lang)
        lang_q = lang_presence_score(qnt["text"], case.lang)
        rep_ref = repetition_score(ref["text"])
        rep_q = repetition_score(qnt["text"])
        delta_nll = qnt["nll_q_on_ref"] - ref["nll_ref"]

        all_match.append(ratio)
        all_delta_nll.append(delta_nll)
        all_ref_repl.append(repl_ref)
        all_q_repl.append(repl_q)
        all_lang_ref.append(lang_ref)
        all_lang_q.append(lang_q)
        all_rep_ref.append(rep_ref)
        all_rep_q.append(rep_q)

        print(
            f"[{case.name}] "
            f"match={ratio:.3f} prefix={pml} "
            f"nll_ref={ref['nll_ref']:.4f} nll_q_on_ref={qnt['nll_q_on_ref']:.4f} delta={delta_nll:+.4f} "
            f"repl(ref/q)={repl_ref}/{repl_q} lang(ref/q)={lang_ref:.3f}/{lang_q:.3f} "
            f"repeat(ref/q)={rep_ref:.3f}/{rep_q:.3f}"
        )
        print(f"  REF: {ref['text'][:220].replace(chr(10), ' ')}")
        print(f"  QNT: {qnt['text'][:220].replace(chr(10), ' ')}")

    print("\n=== Summary ===")
    print(f"avg_token_match_ratio: {statistics.mean(all_match):.4f}")
    print(f"avg_delta_nll_q_minus_ref_on_ref_cont: {statistics.mean(all_delta_nll):+.4f}")
    print(f"max_delta_nll_q_minus_ref_on_ref_cont: {max(all_delta_nll):+.4f}")
    print(f"replacement_chars_total_ref: {sum(all_ref_repl)}")
    print(f"replacement_chars_total_quant: {sum(all_q_repl)}")
    print(f"avg_lang_presence_ref: {statistics.mean(all_lang_ref):.4f}")
    print(f"avg_lang_presence_quant: {statistics.mean(all_lang_q):.4f}")
    print(f"avg_repetition_score_ref: {statistics.mean(all_rep_ref):.4f}")
    print(f"avg_repetition_score_quant: {statistics.mean(all_rep_q):.4f}")


if __name__ == "__main__":
    main()
