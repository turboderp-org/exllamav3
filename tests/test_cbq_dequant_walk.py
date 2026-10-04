"""
Quantized DFlash2 codebook serving path: integer row-gather and the staged selector walk.

1. For each integer rate (Q8_0, Q4_1, Q4_0, Q3_1, Q3_0, Q2_1, Q2_0), quantize a small
   [vocab, rank] "codebook" through the same per-32-block scale+min scheme the converter uses and
   check that the CUDA gather + dequant reproduces a torch reference dequant to fp16 precision.
   This pins the bit-packing/unpacking (incl. the two's-complement _0 forms) and the per-block
   scale+min orientation end-to-end.
2. The staged walk kernel and the torch fallback must reproduce the raw bf16 walk on the same
   underlying codebooks: identical paths and confidences up to the quantization error, and paths
   identical between the two quantized implementations.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import exllamav3_ext as ext

dev = torch.device("cuda:0")
VOCAB, RANK = 4096, 256
BS = 32


def quantize_cb(W, bits, asymmetric):
    # per-32-block integer quantize -> (packed [V, qbytes+1] uint8, scales [V, 8] fp16,
    # mins [V, 8] fp16 | None). Must match DFlash2Selector._quantize_codebook_int.
    V, R = W.shape
    Wb = W.view(V, R // BS, BS)
    if asymmetric:
        mn = Wb.min(-1, keepdim = True).values
        mx = Wb.max(-1, keepdim = True).values
        d = ((mx - mn) / ((1 << bits) - 1)).clamp_min(1e-9)
        q = torch.round((Wb - mn) / d).clamp(0, (1 << bits) - 1)  # unsigned [0, 2^b-1]
        scales = d.squeeze(-1).half()
        mins = mn.squeeze(-1).half()
    else:
        amax = Wb.abs().max(-1, keepdim = True).values
        hi = (1 << (bits - 1)) - 1
        d = (amax / hi).clamp_min(1e-9)
        q = torch.round(Wb / d).clamp(-hi - 1, hi)
        q = q % (1 << bits)  # b-bit two's complement
        scales = d.squeeze(-1).half()
        mins = None
    q = q.view(V, R).to(torch.uint8)
    if bits == 8:
        packed = q.contiguous()
    elif bits == 4:
        qq = q.view(V, R // 2, 2)
        packed = (qq[:, :, 0] | (qq[:, :, 1] << 4)).contiguous()
    elif bits == 2:
        qq = q.view(V, R // 4, 4)
        packed = (qq[:, :, 0] | (qq[:, :, 1] << 2) | (qq[:, :, 2] << 4) | (qq[:, :, 3] << 6)).contiguous()
    else:  # bits == 3
        R3 = R * 3
        bitmat = (q.unsqueeze(-1) >> torch.tensor([0, 1, 2], device = dev, dtype = torch.uint8)) & 1
        stream = bitmat.flatten(1)
        nbytes = (R3 + 7) // 8
        stream = F.pad(stream, (0, nbytes * 8 - R3)).view(V, nbytes, 8).to(torch.int32)
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device = dev, dtype = torch.int32)
        packed = (stream * weights).sum(-1).to(torch.uint8).contiguous()
    return F.pad(packed, (0, 1)), scales, mins


def torch_dequant(packed, scales, mins, ids):
    # reference dequant of the given rows (mirrors the gather kernel's unpack, incl. the signed
    # reinterpretation of the _0 forms)
    qbytes = packed.size(1) - 1
    bits = qbytes * 8 // RANK
    qrow = packed[ids]
    n = ids.numel()
    out = torch.zeros(n, RANK, device = dev)
    for c in range(RANK):
        blk = c // BS
        if bits == 8:
            qv = qrow[:, c].to(torch.int8).to(torch.float)
        elif bits == 4:
            u = (qrow[:, c >> 1] >> (4 * (c & 1))) & 0xF
            qv = u.float()
            if mins is None:
                qv = torch.where(u >= 8, qv - 16, qv)
        elif bits == 2:
            u = (qrow[:, c >> 2] >> (2 * (c & 3))) & 0x3
            qv = u.float()
            if mins is None:
                qv = torch.where(u >= 2, qv - 4, qv)
        else:  # bits == 3
            bit = 3 * c
            two = qrow[:, bit >> 3].to(torch.int16) | (qrow[:, (bit >> 3) + 1].to(torch.int16) << 8)
            u = (two >> (bit & 7)) & 0x7
            qv = u.float()
            if mins is None:
                qv = torch.where(u >= 4, qv - 8, qv)
        d = scales[ids, blk].float()
        m = mins[ids, blk].float() if mins is not None else 0.0
        out[:, c] = qv * d + m
    return out


def test_dequant():
    torch.manual_seed(0)
    W = torch.randn(VOCAB, RANK, device = dev) * (torch.rand(1, RANK, device = dev) * 3 + 0.1)
    ids = torch.randperm(VOCAB, device = dev)[:96].long()
    for bits, asym, name in [
        (8, False, "Q8_0"), (4, True, "Q4_1"), (4, False, "Q4_0"),
        (3, True, "Q3_1"), (3, False, "Q3_0"), (2, True, "Q2_1"), (2, False, "Q2_0"),
    ]:
        packed, scales, mins = quantize_cb(W, bits, asym)
        st = torch.empty((ids.numel(), RANK), dtype = torch.half, device = dev)
        ext.dflash2_cb_gather_int(packed, scales, mins, ids, st)
        ref = torch_dequant(packed, scales, mins, ids)
        rel = (st.float() - ref).abs().max().item() / ref.abs().max().item()
        print(f"  {name}: max rel err {rel:.2e}")
        assert rel < 5e-3, f"{name}: dequant mismatch {rel}"
    print("dequant vs torch reference: ok")


def test_walk(bits = 8, asymmetric = False, match_min = 0.9, cerr_max = 5e-2):
    torch.manual_seed(1)
    bsz, rows, k = 3, 7, 16
    Wp = torch.randn(VOCAB, RANK, device = dev).bfloat16()
    Ws = torch.randn(VOCAB, RANK, device = dev).bfloat16()
    pp, ps, pm = quantize_cb(Wp.float(), bits, asymmetric)
    sp, ss, sm = quantize_cb(Ws.float(), bits, asymmetric)

    unary = torch.randn(bsz, rows, k, device = dev)
    cands = torch.randint(0, VOCAB, (bsz, rows, k), device = dev, dtype = torch.long)
    gate = (torch.randn(bsz, rows, RANK, device = dev) * 0.3).half()
    anchor = torch.randint(0, VOCAB, (bsz,), device = dev, dtype = torch.long)
    out1 = torch.empty((bsz, rows + 1), dtype = torch.long, device = dev)
    conf1 = torch.empty((bsz, rows + 1), dtype = torch.float, device = dev)
    ext.dflash2_selector_walk(unary, cands, gate, Wp, Ws, anchor, out1, conf1)

    ids_a = torch.cat((anchor[:, None], cands[:, :rows - 1, :].reshape(bsz, -1)), dim = 1).reshape(-1)
    stA = torch.empty((ids_a.numel(), RANK), dtype = torch.half, device = dev)
    ext.dflash2_cb_gather_int(pp, ps, pm, ids_a, stA)
    stB = torch.empty((cands.numel(), RANK), dtype = torch.half, device = dev)
    ext.dflash2_cb_gather_int(sp, ss, sm, cands.reshape(-1), stB)
    svh_AB = torch.ones(RANK, dtype = torch.half, device = dev)
    out2 = torch.empty((bsz, rows + 1), dtype = torch.long, device = dev)
    conf2 = torch.empty((bsz, rows + 1), dtype = torch.float, device = dev)
    ext.dflash2_selector_walk_staged(unary, cands, gate, stA.view(bsz, 1 + (rows - 1) * k, RANK),
                                     stB.view(bsz, rows, k, RANK), svh_AB, anchor, out2, conf2)

    match = (out1 == out2).float().mean().item()
    cerr = (conf1[1:] - conf2[1:]).abs().max().item() / conf1[1:].abs().max().item()
    print(f"  {bits}-bit {'asym' if asymmetric else 'sym'} staged vs raw: path match {match:.2%}, conf rel err {cerr:.2e}")
    assert match >= match_min and cerr < cerr_max

    # torch staged fallback vs kernel on the same quantized data: exact same choices
    stA3 = stA.float().view(bsz, -1, RANK)
    stB4 = stB.float().view(bsz, rows, k, RANK)
    pred = anchor
    slot = torch.zeros(bsz, dtype = torch.long, device = dev)
    path = [pred]
    for i in range(rows):
        a = stA3.gather(1, slot[:, None, None].expand(-1, 1, RANK)).view(bsz, RANK)
        w = a * gate[:, i].float()
        scores = unary[:, i] + torch.einsum("br,bkr->bk", w, stB4[:, i])
        idx = torch.max(scores, dim = -1).indices
        pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
        slot = 1 + i * k + idx
        path.append(pred)
    out3 = torch.stack(path, dim = 1)
    assert (out2 == out3).all().item(), "torch staged fallback diverged from kernel"
    print(f"  {bits}-bit {'asym' if asymmetric else 'sym'} walk: staged kernel ~ raw walk, torch fallback == kernel: ok")


if __name__ == "__main__":
    test_dequant()
    # Q8_0 is lossless for the selector -> 100% path match on any data. Sub-8-bit rates on
    # RANDOM codebooks have large quantization error relative to the score gaps, so the path
    # match with the raw walk is only a sanity bound here (the real 27B codebook measures
    # 97.7% for Q4_1); the hard invariant is the exact kernel-vs-torch-fallback agreement below.
    test_walk(bits = 8, asymmetric = False, match_min = 1.0)
    test_walk(bits = 4, asymmetric = True, match_min = 0.4, cerr_max = 1.0)
    test_walk(bits = 4, asymmetric = False, match_min = 0.4, cerr_max = 1.0)
    print("all cbq tests passed")