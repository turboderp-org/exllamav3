import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Tokenizer
from datasets import load_dataset
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


def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 512):
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


@torch.inference_mode()
def main(args):

    config_a = Config.from_directory(args.model_a)
    config_a.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config_a)
    model_a = Model.from_config(config_a)

    config_b = Config.from_directory(args.model_b)
    config_b.override_dynamic_seq_len(2048)
    model_b = Model.from_config(config_b)

    # Dataset
    eval_ids = get_test_tokens(tokenizer, args.rows)
    state_a = eval_ids
    state_b = eval_ids

    for idx, (module_a, module_b) in enumerate(zip(model_a.modules, model_b.modules)):

        module_a.load("cuda:0" if not module_a.caps.get("prefer_cpu") else "cpu")
        params_a = {}
        state_a = module_a.prepare_for_device(state_a, params_a)
        state_a = module_a.forward(state_a, params_a)
        module_a.unload()
        free_mem()

        module_b.load("cuda:0" if not module_b.caps.get("prefer_cpu") else "cpu")
        params_b = {}
        state_b = module_b.prepare_for_device(state_b, params_b)
        state_b = module_b.forward(state_b, params_b)
        module_b.unload()
        free_mem()

        if idx < args.keep_b:
            state_a = state_b.clone()

        max_diff = 0
        rfn_error_sum = 0
        rows = state_a.shape[0]
        for i in range(rows):
            sa = state_a[i].to(float, copy = True)
            sb = state_b[i].to(float)
            sa -= sb
            rfn_error_sum += (torch.linalg.norm(sa, 'fro') / torch.linalg.norm(sb, 'fro').mean()).item()
            sa.abs_()
            md = ((sa.max().item()) / torch.linalg.norm(sb, 'fro').mean()).item()
            max_diff = max(max_diff, md)

        del sa, sb
        rfn_error = rfn_error_sum / rows
        print(f" -- {module_a.key:40}   error: {rfn_error:.6f}   max_diff/norm: {max_diff:.6f}")

    # Compare logits
    topk_max = args.topk_max
    logprob_sum = [0, 0]
    logprob_count = [0, 0]
    kl_div_sum_ab = 0
    kl_div_sum_ba = 0
    topk_hits_sum = [[0] * topk_max, [0] * topk_max]
    topk_hits_count = [[0] * topk_max, [0] * topk_max]
    topk_agreement_sum = [0] * topk_max
    topk_agreement_count = [0] * topk_max

    def ppl(input_ids_, logits_):
        nonlocal logprob_sum, logprob_count
        logprob_sum_ = 0.0
        logprob_count_ = 0
        chunksize = logits_.shape[1] * 10240 // logits_.shape[1]
        b_ = 0
        while b_ < logits_.shape[1]:
            a_ = b_
            b_ = min(b_ + chunksize, logits_.shape[1])
            logits_f = logits_[a_:b_, :].float() + 1e-10
            target_ids = input_ids_[a_ + 1:b_ + 1].to(logits_.device)
            log_probs = F.log_softmax(logits_f, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum_ += token_log_probs.sum().item()
            logprob_count_ += target_ids.numel()
        return logprob_sum_, logprob_count_

    rows = state_a.shape[0]
    for j in range(rows):
        x = (state_a[j], state_b[j])
        input_ids = eval_ids[j]
        top_indices = []

        for i in [0, 1]:
            logits = x[i][:-1, :]
            logprob_sum__, logprob_count__ = ppl(input_ids, logits)
            logprob_sum[i] += logprob_sum__
            logprob_count[i] += logprob_count__

            _, top_index = torch.topk(logits, topk_max, dim = -1)
            top_index = top_index.cpu().view(-1, topk_max)
            top_indices.append(top_index)
            targets = input_ids[1:].view(-1, 1)

            for t in range(topk_max):
                top_slice = top_index[:, :t + 1]
                hits = torch.eq(targets, top_slice)
                row_hits = hits.any(dim = 1)
                topk_hits_sum[i][t] += row_hits.sum().item()
                topk_hits_count[i][t] += top_slice.shape[0]

        for t in range(topk_max):
            top_slice_a = top_indices[0][:, :t + 1]
            top_slice_b = top_indices[1][:, :t + 1]
            hits = torch.eq(top_slice_a, top_slice_b)
            row_hits = hits.all(dim = 1)
            topk_agreement_sum[t] += row_hits.sum().item()
            topk_agreement_count[t] += top_slice_a.shape[0]

        epsilon = 1e-10
        probs_a = torch.softmax(x[0].float(), dim = -1)
        probs_b = torch.softmax(x[1].float(), dim = -1)
        kl_div = F.kl_div(torch.log(probs_a + epsilon), probs_b, reduction = 'none')
        kl_div_sum_ab += kl_div.sum(dim = -1).mean().item()
        kl_div = F.kl_div(torch.log(probs_b + epsilon), probs_a, reduction = 'none')
        kl_div_sum_ba += kl_div.sum(dim = -1).mean().item()

    perplexity = [math.exp(-logprob_sum[i] / logprob_count[i]) for i in (0, 1)]
    kl_div_ab = kl_div_sum_ab / rows
    kl_div_ba = kl_div_sum_ba / rows

    # Perplexity for each model
    print(f" -- A perplexity: {perplexity[0]:11.8f}")
    print(f" -- B perplexity: {perplexity[1]:11.8f}")

    # Probability of the test label being in the top K tokens, for each model
    print(f" -- A label in top-K:")
    for t in range(topk_max):
        a_acc_ = topk_hits_sum[0][t] / topk_hits_count[0][t]
        print(f"      K = {t+1}: {a_acc_:6.4f}")
    print(f" -- B label in top-K:")
    for t in range(topk_max):
        a_acc_ = topk_hits_sum[1][t] / topk_hits_count[1][t]
        print(f"      K = {t+1}: {a_acc_:6.4f}")

    # Probability of exact top-K token match between models
    print(f" -- Top-K agreement, A vs B:")
    for t in range(topk_max):
        topk_agree_ = topk_agreement_sum[t] / topk_agreement_count[t]
        print(f"      K = {t+1}: {topk_agree_:6.4f}")

    # KLD, either way around
    print(f" -- KL divergence (A, B): {kl_div_ab:11.8f}")
    print(f" -- KL divergence (B, A): {kl_div_ba:11.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ma", "--model_a", type = str, help = "Model A", required = True)
    parser.add_argument("-mb", "--model_b", type = str, help = "Model B", required = True)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-kb", "--keep_b", type = int, help = "Maintain B state for number of modules", default = 0)
    parser.add_argument("-tkm", "--topk_max", type = int, default = 5, help = "Max top-K interval to test")

    _args = parser.parse_args()
    main(_args)
