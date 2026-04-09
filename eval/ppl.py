import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.progress import ProgressBar
from exllamav3 import model_init, Tokenizer, Config
from datasets import load_dataset
from exllamav3.util.memory import free_mem
import torch
import torch.nn.functional as F
import math


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 2048):
    with ProgressBar("Tokenizing", rows) as pb:
        dataset_spec = { "dataset": "wiki2" }
        eval_tokens = tokenizer.encode(get_dataset_text(dataset_spec))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim = 0)[:, :]


@disk_lru_cache("_load_wikitext2_raw")
def _load_wikitext2_raw() -> str:
    """
    Get raw wikitext2 test split exactly as used by perplexity.c (datasets version differs slightly on whitespace)
    """

    import tempfile, zipfile, pathlib, urllib.request

    _WIKITEXT2_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

    cache_dir = pathlib.Path(tempfile.gettempdir()) / "llama_cpp_ppl_wikitext2"
    cache_dir.mkdir(parents = True, exist_ok = True)

    raw_path = cache_dir / "wikitext-2-raw" / "wiki.test.raw"
    if not raw_path.exists():

        zip_path = cache_dir / "wikitext-2-raw-v1.zip"
        if not zip_path.exists():
            print(f"Downloading WikiText-2 raw to {zip_path} ...")
            urllib.request.urlretrieve(_WIKITEXT2_URL, str(zip_path))

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))

        zip_path.unlink(missing_ok = True)
        if not raw_path.exists():
            raise FileNotFoundError(f"Failed to extract to {raw_path}.")

    with open(raw_path, "r", encoding = "utf-8") as f:
        return f.read()


def get_test_tokens_gguf(tokenizer) -> list[int]:
    text = _load_wikitext2_raw()
    token_ids = tokenizer.encode(text, add_bos = False)
    return token_ids[0].tolist()


def eval_gguf(model, config, tokenizer, args, forward_fn):

    # Dataset
    tokens = get_test_tokens_gguf(tokenizer)
    bos_id = tokenizer.bos_token_id
    n_ctx = args.ctx_size

    if len(tokens) < 2 * n_ctx:
        raise ValueError(
            f"Need at least {2 * n_ctx} tokens for n_ctx={n_ctx}, "
            f"but the dataset tokenizes to only {len(tokens)} tokens."
        )

    # Chunking (non-overlapping)
    n_chunks = len(tokens) // n_ctx
    first = n_ctx // 2
    print(f"Perplexity: {len(tokens)} tokens, {n_chunks} chunks, n_ctx={n_ctx}, scoring positions [{first}, {n_ctx - 2}]")

    total_nll = 0.0
    total_nll2 = 0.0
    total_count = 0
    per_chunk_ppl: list[float] = []

    for i in range(n_chunks):
        start = i * n_ctx
        chunk_tokens = tokens[start : start + n_ctx]

        # Optionally overwrite first token in each chunk with BOS
        if bos_id is not None:
            chunk_tokens = [bos_id] + chunk_tokens[1:]

        input_ids = torch.tensor([chunk_tokens], dtype=torch.long)
        logits = forward_fn(model, input_ids)
        logits = logits[0, :, :tokenizer.actual_vocab_size].float()

        score_logits = logits[first : n_ctx - 1]  # [n_ctx - 1 - first, vocab]
        score_tokens = tokens[start + first + 1 : start + n_ctx]
        score_tokens = torch.tensor(score_tokens, dtype = torch.long, device = score_logits.device)

        # Compute per-token NLL via log-softmax (matches llama.cpp's log_softmax function: logit[tok] - max - log(sum_exp))
        log_probs = F.log_softmax(score_logits, dim=-1)
        token_nlls = -log_probs[torch.arange(len(score_tokens), device=logits.device), score_tokens]

        nll_sum = token_nlls.sum().item()
        nll2_sum = (token_nlls * token_nlls).sum().item()
        n_scored = len(score_tokens)
        total_nll += nll_sum
        total_nll2 += nll2_sum
        total_count += n_scored
        running_ppl = math.exp(total_nll / total_count)
        per_chunk_ppl.append(running_ppl)

        print(f"[{i + 1}]{running_ppl:.4f}", end=",", flush=True)

    print()

    # Aggregate
    mean_nll = total_nll / total_count
    ppl = math.exp(mean_nll)
    mean_nll2 = total_nll2 / total_count
    variance = mean_nll2 - mean_nll * mean_nll
    if variance > 0 and total_count > 1:
        nll_stderr = math.sqrt(variance / (total_count - 1))
    else:
        nll_stderr = 0.0
    ppl_stderr = nll_stderr * ppl

    print(f"Final estimate: PPL = {ppl:.4f} +/- {ppl_stderr:.5f}")


def eval_default(model, config, tokenizer, args, forward_fn):

    # Dataset
    eval_ids = get_test_tokens(tokenizer, args.rows, eval_len = args.length)
    vocab_size = tokenizer.actual_vocab_size

    # Test
    logprob_sum = 0.0
    logprob_count = 0
    with ProgressBar("Evaluating", args.rows) as pb:
        for row in range(eval_ids.shape[0]):
            pb.update(row)
            input_ids = eval_ids[row:row + 1, :]
            logits = forward_fn(model, input_ids)
            logits = logits[:, :-1, :vocab_size].float()
            logits += 1e-10
            log_probs = F.log_softmax(logits, dim = -1)
            del logits
            target_ids = input_ids[:, 1:].to(log_probs.device)
            del input_ids
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += target_log_probs.sum().item()
            logprob_count += target_ids.numel()
            del log_probs
            del target_log_probs
            del target_ids
            torch.cuda.empty_cache()
        pb.update(args.rows)
        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

    print(f" -- Evaluated: {eval_ids.shape[0]} rows of {eval_ids.shape[1]} tokens")
    print(f" -- Perplexity: {perplexity:.6f}")



@torch.inference_mode()
def main(args):

    if not args.hf:
        model, config, _, tokenizer = model_init.init(
            args,
            override_dynamic_seq_len = 2048,
            max_output_size = 2048,
            max_output_factor = 5,
        )
        def forward_fn_exl3(_model, _input_ids):
            return _model.forward(_input_ids, {"attn_mode": "flash_attn_nc"})
        forward_fn = forward_fn_exl3

        bpw_layer, bpw_head, vram_bits = model.get_storage_info()
        print(f" -- Model: {args.model_dir}")
        print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    else:
        from transformers import AutoModelForCausalLM

        config = Config.from_directory(args.model_dir)
        tokenizer = Tokenizer.from_config(config)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map = "auto",
            torch_dtype = torch.half if args.hf_tight else torch.float if args.hf_fp32 else None,
        )

        if args.hf_tight:
            free_mem()
            model.half()
            free_mem()
        if args.hf_fp32:
            free_mem()
            model.float()
            free_mem()

        def forward_fn_hf(_model, _input_ids):
            return _model.forward(_input_ids)["logits"]
        forward_fn = forward_fn_hf


    if not args.gguf:
        eval_default(model, config, tokenizer, args, forward_fn)
    else:
        eval_gguf(model, config, tokenizer, args, forward_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = False)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-l", "--length", type = int, help = "Length", default = 2048)
    parser.add_argument("-g", "--gguf", action = "store_true", help = "Use GGUF-equivalent eval logic (ignores -r and -l)")
    parser.add_argument("-c", "--ctx-size", type = int, help = "For GGUF-equiv.: size of the prompt context (default: 512)", default = 512)
    parser.add_argument("-hf", "--hf", action = "store_true", help = "Use Transformers as backend (-m must be HF model)")
    parser.add_argument("-hf_t", "--hf_tight", action = "store_true", help = "For Transformers: Force FP16 dtype to save memory")
    parser.add_argument("-hf_fp32", "--hf_fp32", action = "store_true", help = "For Transformers: Force FP32 dtype")

    _args = parser.parse_args()
    main(_args)
