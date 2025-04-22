import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules import TransformerBlock
from exllamav3.util.hadamard import get_hadamard_dt
from datasets import load_dataset
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from flash_attn import flash_attn_func
from ref_quant2 import quantquant
from exllamav3.ext import exllamav3_ext as ext
import math

torch.set_printoptions(precision = 8, sci_mode = False, linewidth = 200)

model_dir = "/mnt/str/models/llama3.1-8b-instruct/hf/"
device = "cuda:1"
target_layers = [0]
num_rows = 1

# Create input tensor
@disk_lru_cache("get_test_data")
def get_test_data():
    return "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )

# Sample Q and K tensors from forward pass, Llama type model
@disk_lru_cache("sample_qkv")
def sample_qkv(_model_dir, _target_layers, _num_rows):

    # Load model
    config = Config.from_directory(_model_dir)
    model = Model.from_config(config)
    model.load(device, progressbar = True)
    tokenizer = Tokenizer.from_config(config)

    test_data = get_test_data()[:100000]
    eval_tokens = tokenizer.encode(test_data)
    eval_len = 2048
    eval_stride = 512
    num_tokens = eval_tokens.shape[-1]
    seqs = []
    for a in range(0, num_tokens - eval_len, eval_stride):
        b = a + eval_len
        seqs.append(eval_tokens[:, a:b])
        if len(seqs) >= num_rows:
            break
    input_ids = torch.cat(seqs, dim = 0)[:, :]

    _samples_qkv = []
    params = {}
    x = model.prepare_inputs(input_ids, params)
    for idx, module in enumerate(model.modules):
        params["prefill"] = (idx == model.last_kv_module_idx)
        x = module.prepare_for_device(x, params)
        if isinstance(module, TransformerBlock):
            block_idx = int(module.key.split(".")[-1])
            if block_idx > max(_target_layers):
                break
            if block_idx in _target_layers:
                # Pre-attn norm
                y = module.attn_norm.forward(x, params, out_dtype = torch.half)
                # Projections and RoPE
                attn = module.attn
                bsz, seqlen, _ = y.shape
                position, positions, position_ids = 0, None, None
                q, k, v = attn.project_qkv(y, params)
                q = q.view(bsz, seqlen, attn.num_q_heads, attn.head_dim)
                k = k.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
                v = v.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
                q, k = attn.rope.apply(q, k, position, positions, position_ids)
                # Sample right before dot product
                _samples_qkv.append((q, k, v))
        # Advance state
        x = module.forward(x, params)

    return _samples_qkv

samples_qkv = sample_qkv(model_dir, target_layers, num_rows)

# Get attention scores and output
def attn(q, k, v):
    bsz, q_len, n_heads_q, head_dim = q.shape
    _, k_len, n_heads_k, _ = k.shape
    gqa = n_heads_q // n_heads_k
    k_int = k.repeat_interleave(gqa, dim = 2)
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k_int) / math.sqrt(head_dim)

    # Causal mask
    mask = torch.ones((k_len, k_len), dtype = torch.bool, device = q.device).triu(diagonal = 1)
    mask = mask[-q_len:, :]
    scores = scores.masked_fill_(mask, -65504.)

    # Now attention
    o = flash_attn_func(
        q = q,
        k = k,
        v = v,
        causal = True,
    )
    return o, scores

# Refence method
def int_quant(v, bits):
    m = 1 << (bits - 1)
    scales = torch.amax(v.abs(), dim = -1).unsqueeze(3)
    v = v / scales
    vq = (v * m).round().clamp(-m, m - 1)
    vq /= m
    vq *= scales
    return vq

# def quant_nf4(t):
#     scales = torch.amax(t.abs(), dim = -1).unsqueeze(3)
#     tq = t / scales
#     tqq = torch.empty_like(tq)
#     ext.test_nf4(tq, tqq)
#     tqq *= scales
#     return tqq

def quant_fp8(t):
    return t.to(torch.float8_e4m3fn).half()


# Kernel equiv reference
def kernel_ref_quant(v, bits):
    had32 = get_hadamard_dt(32, v.device, torch.half)
    w = v.view(-1, 32)
    m = 1 << (bits - 1)
    w = w @ had32 / math.sqrt(32)
    scales = torch.amax(w.abs(), dim = -1, keepdim = True).half()
    w = w / scales
    vq = (w * m).round().clamp(-m, m - 1)
    vq /= m
    vq *= scales
    vq = vq @ had32 / math.sqrt(32)
    vq = vq.view(v.shape)
    return vq

# KL divergence between softmax distributions
def kl_divergence_scores(s, s_prime, dim = -1, eps = 1e-8):
    alpha = F.softmax(s.float(), dim = dim)
    alpha_hat = F.softmax(s_prime.float(), dim = dim)
    kl_elementwise = alpha * (torch.log(alpha + eps) - torch.log(alpha_hat + eps))
    kl_per_item = kl_elementwise.sum(dim = dim)
    kl_mean = kl_per_item.mean()
    return kl_mean

# Normalized MSE
def nmse(o, o_prime):
    return (o - o_prime).square().mean() / o_prime.square().mean()


# Do stuff
def test_qkv(label, q, k, v, ref_o, ref_scores, q_rot = False, k_rot = False, v_rot = False):
    head_dim = q.shape[-1]
    had = get_hadamard_dt(head_dim, device, torch.half)
    if q_rot != k_rot: q = (q @ had) / math.sqrt(head_dim)
    if v_rot: v = (v @ had) / math.sqrt(head_dim)
    test_o, test_scores = attn(q, k, v)
    kld = kl_divergence_scores(test_scores, ref_scores)
    mse = nmse(test_o, ref_o)
    print(f"{label:26}   weights_kld: {kld:.6f}   output_nmse: {mse:.6f}")

with torch.inference_mode():

    head_dim = samples_qkv[0][0].shape[-1]
    had = get_hadamard_dt(head_dim, device, torch.half)

    for idx, (q, k, v) in zip(target_layers, samples_qkv):

        # Unquantized
        ref_o, ref_scores = attn(q, k, v)

        # Q4
        test_qkv(
            "Q4",
            q,
            int_quant(k, 4),
            int_quant(v, 4),
            ref_o,
            ref_scores
        )

        # Q6
        test_qkv(
            "Q6",
            q,
            int_quant(k, 6),
            int_quant(v, 6),
            ref_o,
            ref_scores
        )

        # Q8
        test_qkv(
            "Q8",
            q,
            int_quant(k, 8),
            int_quant(v, 8),
            ref_o,
            ref_scores
        )

        # Rotated Q4
        test_qkv(
            "Rot. Q4",
            q,
            int_quant((k @ had) / math.sqrt(head_dim), 4),
            int_quant((v @ had) / math.sqrt(head_dim), 4),
            ref_o,
            ref_scores,
            False, True, True
        )

        # Rotated Q6
        test_qkv(
            "Rot. Q6",
            q,
            int_quant((k @ had) / math.sqrt(head_dim), 6),
            int_quant((v @ had) / math.sqrt(head_dim), 6),
            ref_o,
            ref_scores,
            False, True, True
        )

        # Channel scales + rotated Q4
        psc_k = k.view(-1, k.shape[-2], k.shape[-1]).abs().mean(dim = 0)
        psc_v = v.view(-1, k.shape[-2], k.shape[-1]).abs().mean(dim = 0)
        test_qkv(
            "Rot. Q4 ch.scales",
            q,
            int_quant(((k / psc_k) @ had) / math.sqrt(head_dim), 4) @ had / math.sqrt(head_dim) * psc_k,
            int_quant(((v / psc_v) @ had) / math.sqrt(head_dim), 4) @ had / math.sqrt(head_dim) * psc_v,
            ref_o,
            ref_scores,
            False, False, False
        )

        # Channel scales + rotated Q4 RMS
        pscr_k = k.view(-1, k.shape[-2], k.shape[-1]).square().mean(dim = 0).sqrt()
        pscr_v = v.view(-1, k.shape[-2], k.shape[-1]).square().mean(dim = 0).sqrt()
        test_qkv(
            "Rot. Q4 ch.scales (RMS)",
            q,
            int_quant(((k / pscr_k) @ had) / math.sqrt(head_dim), 4) @ had / math.sqrt(head_dim) * pscr_k,
            int_quant(((v / pscr_v) @ had) / math.sqrt(head_dim), 4) @ had / math.sqrt(head_dim) * pscr_v,
            ref_o,
            ref_scores,
            False, False, False
        )

        # Rotated Q4 + Q6
        test_qkv(
            "Rot. Q4+Q6",
            q,
            int_quant((k @ had) / math.sqrt(head_dim), 4),
            int_quant((v @ had) / math.sqrt(head_dim), 6),
            ref_o,
            ref_scores,
            False, True, True
        )

        # NF4
        # k_nf4 = quant_nf4(k)
        # v_nf4 = quant_nf4(v)
        # test_qkv("NF4", q, k_nf4, v_nf4, ref_o, ref_scores, False, False, False)

        # Rotated NF4
        # k_h = (k @ had) / math.sqrt(128)
        # v_h = (v @ had) / math.sqrt(128)
        # k_h_nf4 = quant_nf4(k_h)
        # v_h_nf4 = quant_nf4(v_h)
        # test_qkv("RNF4", q, k_h_nf4, v_h_nf4, ref_o, ref_scores, False, True, True)

        # FP8
        test_qkv(
            "FP8 e4m3",
            q,
            quant_fp8(k),
            quant_fp8(v),
            ref_o,
            ref_scores,
            False, False, False
        )

        # Kernel
        for bits in range(2, 9):
            quant_shape = k.shape[:-1] + (128 // 32 * bits,)
            scale_shape = k.shape[:-1] + (128 // 32,)
            k_quant = torch.zeros(quant_shape, dtype = torch.int, device = k.device)
            k_scale = torch.zeros(scale_shape, dtype = torch.half, device = k.device)
            v_quant = torch.zeros(quant_shape, dtype = torch.int, device = k.device)
            v_scale = torch.zeros(scale_shape, dtype = torch.half, device = k.device)
            ext.quant_cache_cont(k, k_quant, k_scale)
            ext.quant_cache_cont(v, v_quant, v_scale)
            k_kern = torch.empty_like(k)
            v_kern = torch.empty_like(v)
            ext.dequant_cache_cont(k_quant, k_scale, k_kern)
            ext.dequant_cache_cont(v_quant, v_scale, v_kern)
            test_qkv(f"Kernel {bits} bits", q, k_kern, v_kern, ref_o, ref_scores, False, False, False)

        # Reference
        test_qkv(f"Kernel ref 4 bits",
            q,
            kernel_ref_quant(k, 4),
            kernel_ref_quant(v, 4),
            ref_o,
            ref_scores,
            False, False, False
        )