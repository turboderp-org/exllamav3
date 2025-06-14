import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Cache, Tokenizer, model_init
from exllamav3.ext import exllamav3_ext as ext
from datasets import load_dataset
from exllamav3.modules import Linear
from exllamav3.modules.quant import LinearFP16, LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import regularize
import torch
import torch.nn.functional as F
import math
import numpy as np
import termplotlib as tpl

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_purple = "\u001b[35;1m"
col_cyan = "\u001b[36;1m"
col_white = "\u001b[37;1m"

block_chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, bos, eval_len = 2048, eval_stride = 512):
    with ProgressBar("Tokenizing", rows) as pb:
        dataset_spec = { "dataset": "wiki2" }
        eval_tokens = tokenizer.encode(get_dataset_text(dataset_spec))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            if bos is not None:
                r = torch.cat((bos, eval_tokens[:, a:b-1]), dim = -1)
            else:
                r = eval_tokens[:, a:b]
            seqs.append(r)
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim = 0)[:, :]



@torch.compile(fullgraph = True, mode = "reduce-overhead")
def count_threshold(x: torch.Tensor, abs_threshold: float) -> torch.Tensor:
    return (x.abs() > abs_threshold).sum(dtype = torch.int64)


def bchart(bins, min_value, max_value, cc, height = 10):
    maxcount = bins.max()
    lines = []
    for r in range(height):
        line = cc
        for c in range(len(bins)):
            if maxcount == 0:
                b = block_chars[0]
            else:
                y = (bins[c] / maxcount) * height - r
                y = max(min(y, 1), 0)
                b = block_chars[int(y * 8)]
            line += b
        line += col_default
        lines.append(line)
    lines.reverse()
    return lines


def histogram(args, tensor):

    nbins = args.histogram_bins
    stddev = tensor.std()
    min_value = tensor.amin().item()
    max_value = tensor.amax().item()

    # Middle histogram
    m_min_value = -stddev * 3
    m_max_value = stddev * 3

    m_bins = torch.empty((nbins // 2,), dtype = torch.long, device = tensor.device)
    ext.histogram(tensor, m_bins, m_min_value, m_max_value, True)
    m_count = m_bins.sum().item()
    m_bins = m_bins.cpu().numpy()

    # Low histogram
    l_min_value = min_value
    l_max_value = m_min_value
    l_bins = torch.empty((nbins // 4,), dtype = torch.long, device = tensor.device)
    ext.histogram(tensor, l_bins, l_min_value, l_max_value, True)
    l_count = l_bins.sum().item()
    l_bins = l_bins.cpu().numpy()

    # High histogram
    h_min_value = m_max_value + 0.001
    h_max_value = max_value
    h_bins = torch.empty((nbins // 4,), dtype = torch.long, device = tensor.device)
    ext.histogram(tensor, h_bins, h_min_value, h_max_value, True)
    h_count = h_bins.sum().item()
    h_bins = h_bins.cpu().numpy()

    total = l_count + m_count + h_count
    l_pct = l_count / total * 100.0
    m_pct = m_count / total * 100.0
    h_pct = h_count / total * 100.0

    bc_l = bchart(l_bins, l_min_value, l_max_value, col_yellow)
    bc_m = bchart(m_bins, m_min_value, m_max_value, col_cyan)
    bc_h = bchart(h_bins, h_min_value, h_max_value, col_yellow)
    bc = [f"{l} {m} {h}" for l, m, h in zip(bc_l, bc_m, bc_h)]

    bc.append(col_default + ("─" * (nbins // 4)) + "┬" + ("─" * (nbins // 2)) + "┬" + ("─" * (nbins // 4 )) + col_default)

    for vl, vml, vmh, vh in zip(
        [f"{l_min_value:.2e}", f"{l_count:,} elem", f"{l_pct:.5f} %"],
        [f"{m_min_value:.2e}", f"{m_count:,} elem", f"{m_pct:.5f} %"],
        [f"{m_max_value:.2e}", "", ""],
        [f"{h_max_value:.2e}", f"{h_count:,} elem", f"{h_pct:.5f} %"],
    ):
        ax = ""
        ax += col_yellow + vl + (" " * ((nbins // 4) - len(vl)))
        ax += " " + col_cyan + vml + (" " * ((nbins // 2) - len(vml) - len(vmh))) + vmh + " "
        ax += (" " * ((nbins // 4) - len(vh))) + col_yellow + vh + col_default
        bc.append(ax)

    print("\n".join(bc))
    print()


def stats(args, tensor):

    # inf/NaN values
    inf_nan =  torch.zeros(2, dtype = torch.long, device = tensor.device)
    ext.count_inf_nan(tensor, inf_nan)
    inf = inf_nan[0].item()
    nan = inf_nan[1].item()
    if inf or nan:
        print(f"{col_red}inf values                 : {col_white}{inf:,}{col_default}")
        print(f"{col_red}NaN values                 : {col_white}{nan:,}{col_default}")
        print(f"{col_red}Total numel                : {col_white}{tensor.numel():,}{col_default}")
        print()

    min_value = tensor.amin().item()
    max_value = tensor.amax().item()
    print(f"Min value                  : {col_white}{min_value:18.8f}{col_default}")
    print(f"Max value                  : {col_white}{max_value:18.8f}{col_default}")

    sigma = tensor.std(unbiased = False)
    print(f"Std. deviation             : {col_white}{sigma:18.8f}{col_default}")

    # mu = tensor.mean()
    # std2 = tensor.var(unbiased = False)
    # std4 = std2 ** 2
    # kurt = ((tensor - mu) ** 4).mean() / (std4 + 1e-10)
    # print(f"Kurtosis                   : {col_white}{kurt:18.8f}{col_default}")

    n6 = count_threshold(tensor, 6 * sigma)
    n6p = n6 / tensor.numel()
    if n6 > 0:
        print(f"Six-sigma exceedance       : {col_white}{n6p:18.8f}{col_default} ({col_white}{n6:,} elem{col_default})")
    else:
        print(f"Six-sigma exceedance       : {col_white}None{col_default}")

    print()


def inspect_state(args, state):
    state = state[:, args.skip_tokens:, :].to(torch.device(args.device))
    print(f"{col_blue}Hidden states{col_default}")
    print("─────────────")
    stats(args, state)
    histogram(args, state)


def inspect_module(args, module):
    linears = [m for m in module if isinstance(m, Linear)]
    for linear in linears:
        print(f"{col_blue}{linear.key}{col_default}")
        print("─" * len(linear.key))
        w = linear.inner.get_weight_tensor()
        stats(args, w)

        nbins = args.histogram_bins
        # bitrate = args.regularize_bits
        stddev = w.std(unbiased = False)

        min_value = -stddev * 4
        max_value = stddev * 4
        bins = torch.empty((nbins // 2,), dtype = torch.long, device = w.device)
        ext.histogram(w, bins, min_value, max_value, False)

        k, n = w.shape
        su = (torch.randn(k, device = w.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
        sv = (torch.randn(n, device = w.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
        quant_args = {
            # "K": bitrate,
            "apply_out_scales": None,
            "devices": [args.device],
        }
        apply_out_scales, w, g_scale, su, sv = regularize(
            w.float(),
            su,
            sv,
            quant_args,
            False,
            None,
            None,
            skip_g_scale = True
        )

        r_stddev = w.std(unbiased = False)
        r_min_value = -r_stddev * 4
        r_max_value = r_stddev * 4
        r_bins = torch.empty((nbins // 2,), dtype = torch.long, device = w.device)
        ext.histogram(w, r_bins, r_min_value, r_max_value, False)

        print(f"Reg. std. deviation        : {col_white}{r_stddev:18.8f}{col_default}")
        w4 = count_threshold(w, 4)
        w4p = w4 / w.numel()
        if w4 > 0:
            print(f"Reg. outliers >4           : {col_yellow}{w4p:18.8f}{col_default} ({col_yellow}{w4:,} elem{col_default})")
        else:
            print(f"Reg. outliers >4           : {col_white}None{col_default}")
        w8 = count_threshold(w, 8)
        w8p = w8 / w.numel()
        if w8 > 0:
            print(f"Reg. outliers >8           : {col_yellow}{w8p:18.8f}{col_default} ({col_yellow}{w8:,} elem{col_default})")
        else:
            print(f"Reg. outliers >8           : {col_white}None{col_default}")
        print()

        bc_pre = bchart(bins, min_value, max_value, col_default)
        bc_reg = bchart(r_bins, r_min_value, r_max_value, col_purple)
        bc = [f"{p}  {r}" for p, r in zip(bc_pre, bc_reg)]

        bc.append(col_default + ("─" * (nbins // 2)) + "  " + ("─" * (nbins // 2)) + col_default)

        for a, b, c, d in zip(
                [f"{min_value:.2e}", "Input layer"],
                [f"{max_value:.2e}", ""],
                [f"{r_min_value:.2e}", "Regularized layer"],
                [f"{r_max_value:.2e}", ""],
        ):
            ax = ""
            ax += col_default + a + (" " * ((nbins // 2) - len(a) - len(b))) + b + col_default
            ax += "  "
            ax += col_purple + c + (" " * ((nbins // 2) - len(c) - len(d))) + d + col_default
            bc.append(ax)

        print("\n".join(bc))
        print()


@torch.inference_mode()
def main(args):

    # Create model config
    config = Config.from_directory(args.model_dir)
    config.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)

    # Input state
    bos = None if not args.bos else torch.tensor([[config.bos_token_id]], dtype = torch.long)
    eval_ids = get_test_tokens(tokenizer, args.rows, bos)
    state = eval_ids

    # Streaming forward pass
    for idx, module in enumerate(model.modules):

        # Load next module
        print(f" -- Loading module: {col_green}{module.key}{col_default}")
        print()
        config.stc.begin_deferred_load()
        module.load(torch.device(args.device) if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()
        if (
            (args.from_layer is None or idx >= args.from_layer) and
            (args.to_layer is None or idx < args.to_layer) and
            not args.no_inspect_modules
        ):
            inspect_module(args, module)

        # Forward pass
        print(f" -- Forward pass")
        print()
        params = {}
        state = module.prepare_for_device(state, params)
        state = module.forward(state, params)
        if (args.from_layer is None or idx >= args.from_layer) and (args.to_layer is None or idx < args.to_layer):
            inspect_state(args, state)

        # Unload current module
        module.unload()
        config.stc.close()
        free_mem()

    # Test perplexity
    vocab_size = tokenizer.actual_vocab_size
    logprob_sum = 0.0
    logprob_count = 0
    with ProgressBar("Evaluating", args.rows) as pb:
        for row in range(state.shape[0]):
            pb.update(row)
            input_ids = eval_ids[row:row + 1, :]
            logits = state[row:row+1, ...]
            logits = logits[:, :-1, :vocab_size].float()
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

    print(f"{col_blue}Outputs{col_default}")
    print("───────")
    print(f"Perplexity                 : {col_white}{perplexity:.6f}{col_default}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-d", "--device", type = int, help = "CUDA device index", default = 0)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 10)
    parser.add_argument("-hb", "--histogram_bins", type = int, help = "Histogram bins", default = 160)
    parser.add_argument("-bos", "--bos", action = "store_true", help = "Add BOS token on each row")
    parser.add_argument("-skip", "--skip_tokens", type = int, help = "Skip tokens at start of context", default = 0)
    parser.add_argument("-fl", "--from_layer", type = int, help = "From layer", default = None)
    parser.add_argument("-tl", "--to_layer", type = int, help = "To layer", default = None)
    parser.add_argument("-nim", "--no_inspect_modules", action = "store_true", help = "Skip module inspection")
    # parser.add_argument("-rb", "--regularize_bits", type = int, help = "Target bitrate for regularization test", default = 4)
    _args = parser.parse_args()
    main(_args)
