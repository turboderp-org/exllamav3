"""
Reference codec for trellis-quantized token-embedding tables (the exl3_trellis_embed format
produced by util/convert_embedding.py).

A packed row is (G + D*K/16) little-endian uint16 words (stored as int16): the first G = D/256
words hold the per-256-column-group fp16 scales (bit pattern), the rest hold the tail-biting
ring bitstream where stream bits [i*K, (i+1)*K) are the low K bits of position i's 16-bit
trellis state (identical pack_rows/unpack_rows semantics as ngram_codec, generalized in D and
with G scale words). Reconstruction of the source row r (the decode chain):

    y[q]   = fp16(codebook[state_q]) * scale_word[q // 256]
    x      = (hadamard(y) * signs) * col_scale       (per 256-group, H/16 butterfly)

with signs regenerated per row from a seeded LCG keyed by the TABLE ROW ID (format-pinned;
see lcg_signs). Output is fp32. Pruned PAD rows are all-zero words: every group scale is 0,
so the row decodes to 0.0 regardless of codes/signs.

This module is the bit-exact CPU reference for the fused GPU kernel (exllamav3_ext/
trellis_embed.cu): every fp32 operation here is applied in the same order as in the kernel
(butterfly stage order, (v * 1/16) * sign * col_scale), so the two paths agree bitwise. The
transform (forward_transform) is the exact inverse of the decode chain in real arithmetic;
its Viterbi-side companion (per-group pre-scale + encode) lives in exllamav3/conversion/
embed.py, which stores the least-squares refit of those pre-scales (ls_refit_scales) as the
G scale words.
"""

from __future__ import annotations
import torch
from .ngram_codec import mul1_codebook as _mul1_codebook_ngram, CS_HEURISTIC, CS_MIN  # noqa: F401  (re-exported)
GROUP = 256
FORMAT = "exl3_trellis_embed"
FORMAT_VERSION = "1"

# Codebook-scale multiplier per K: groups are scaled to rms = cs before the trellis
# search; the per-row heuristic constants live in ngram_codec (shared with the n-gram
# quantizer) and are re-exported above

# Row-blocking budget (elements) for the bit-plane temporaries in pack_rows/unpack_rows/
# dequant_rows: the planes are (rows, D*K)-sized (K or 16 bits per element, int64), so an
# unblocked pass over a production chunk (8192 rows at D = 5120, K = 8) peaks at ~6 GiB of
# transients for a 40 MiB output. Rows are fully independent in all three passes, so the
# blocking is value- and bit-identical; ~2**22 int64 elements keeps each transient ~34 MiB.
PLANE_BUDGET = 1 << 22

# LCG constants (format-pinned): per-row seed key and the step, both mod 2**64
LCG_KEY_SEED_MUL = 0x9E3779B97F4A7C15
LCG_KEY_ROW_MUL = 0xC2B2AE3D27D4EB4F
LCG_STEP_MUL = 6364136223846793005
LCG_STEP_ADD = 1442695040888963407


def _i64(u: int) -> int:
    """uint64 -> the int64 (two's complement) bit pattern, for wraparound tensor arithmetic."""
    u &= (1 << 64) - 1
    return u - (1 << 64) if u >= (1 << 63) else u


def words_per_row(D: int, K: int, G: int | None = None) -> int:
    if G is None:
        G = D // GROUP
    assert D * K % 16 == 0, "D*K must be a multiple of 16"
    assert D % GROUP == 0 and G == D // GROUP, "one fp16 scale word per 256-column group"
    return G + D * K // 16


# Single format-pinned codebook home: exl3_lib.ngram_codec.mul1_codebook. The embedding
# codebook is bit-identical to the n-gram one, so this is only a device-defaulting wrapper
# (it feeds the fused kernel's LUT and the CPU decode; a second verbatim copy here would
# risk silent drift that breaks the fused-vs-reference bit-exactness).
def mul1_codebook(device = "cpu") -> torch.Tensor:
    """All 65536 decoded mul1 values, bit-exact with decode_3inst<2> (fp16)."""
    return _mul1_codebook_ngram(device)


def pack_rows(states: torch.Tensor, scales_f16: torch.Tensor, K: int, D: int,
              G: int | None = None) -> torch.Tensor:
    """
    states: (N, D) uint16/int trellis states from the encoder
    scales_f16: (N, G) float16 per-group scales
    Returns (N, G + D*K/16) int16 packed rows.

    Row-blocked (PLANE_BUDGET): the (blk, D, K) bit planes and the word sums are per-block
    temporaries; rows are independent, so the result is identical to the unblocked pass.
    """
    if G is None:
        G = D // GROUP
    N = states.shape[0]
    dev = states.device
    scale_words = scales_f16.to(torch.float16).view(torch.int16)
    if scale_words.dim() == 1:
        # (G,) shared across all rows: broadcast explicitly - an unsqueeze would give
        # (1, G) and the row-blocked copy below would slice empty blocks past the first
        scale_words = scale_words.expand(N, -1)
    assert scale_words.shape[1] == G
    W = D * K // 16
    out = torch.empty(N, G + W, dtype = torch.int16, device = dev)
    blk = max(1, PLANE_BUDGET // max(1, D * K))
    for lo in range(0, N, blk):
        hi = min(lo + blk, N)
        new_bits = states[lo:hi].to(torch.int64) & ((1 << K) - 1)                     # (blk, D)
        bits = (new_bits.unsqueeze(-1) >> torch.arange(K, device = dev)) & 1           # (blk, D, K)
        bits = bits.reshape(hi - lo, W, 16)
        words = (bits << torch.arange(16, device = dev)).sum(dim = -1)                 # (blk, W)
        words = (words & 0xFFFF).to(torch.uint16).view(torch.int16)
        out[lo:hi, :G] = scale_words[lo:hi]
        out[lo:hi, G:] = words
    return out.contiguous()


def ring_states_from_codes(new: torch.Tensor, K: int) -> torch.Tensor:
    """(N, D) K-bit code symbols -> the 16-bit ring windows the DECODER uses:
    state_i = new_i | new_{i-1} << K | (new_{i-2} & m2) << 2K | ... (masked to 16 bits, ring
    wrap). Single home for the recurrence: unpack_rows derives it from the packed words and
    ls_refit_scales derives it from the encoder's states, so the refit always fits against the
    states the decoder will actually see (an encoder whose head-of-row ring window is not
    tail-biting consistent would otherwise be fitted against states that never decode)."""
    st = new
    shift = K
    lag = 1
    while shift < 16:
        width = min(K, 16 - shift)
        st = st | ((torch.roll(new, shifts = lag, dims = 1) & ((1 << width) - 1)) << shift)
        shift += K
        lag += 1
    return st


def unpack_rows(packed: torch.Tensor, K: int, D: int, G: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_rows: returns (states (N, D) int64, scales (N, G) float16).

    state_i is the 16 ring bits ending at stream bit (i+1)*K - 1, i.e.
    state_i = new_i | new_{i-1} << K | new_{i-2} << 2K | ... (masked to 16
    bits, ring wrap; ring_states_from_codes). Built by shifts/or, NOT a (N, D,
    16) gather - that intermediate is production-sized and the gather form is
    also wrong for K that does not divide 16 (torch advanced-index expansion).
    The word-window planes are row-blocked (PLANE_BUDGET); rows are
    independent, so the result is identical to the unblocked pass.
    """
    if G is None:
        G = D // GROUP
    dev = packed.device
    N = packed.shape[0]
    scales = packed[:, :G].contiguous().view(torch.float16)
    W = D * K // 16
    states = torch.empty(N, D, dtype = torch.int64, device = dev)
    blk = max(1, PLANE_BUDGET // max(1, D * K))
    ar16 = torch.arange(16, device = dev)
    arK = torch.arange(K, device = dev)
    for lo in range(0, N, blk):
        hi = min(lo + blk, N)
        words = packed[lo:hi, G:].view(torch.uint16).to(torch.int64)                   # (blk, W)
        bits = (words.unsqueeze(-1) >> ar16) & 1                                       # (blk, W, 16)
        new = (bits.reshape(hi - lo, D, K) << arK).sum(dim = -1)
        states[lo:hi] = ring_states_from_codes(new, K)
    return states, scales


def dequant_rows(packed: torch.Tensor, K: int, codebook: torch.Tensor, D: int,
                 G: int | None = None) -> torch.Tensor:
    """Decode chain steps 1-2: packed (N, G + D*K/16) int16 -> transformed-space y (N, D) fp32.

    The codebook lookup is row-blocked (PLANE_BUDGET) so the int64 states plane from
    unpack_rows never doubles into a full fp32 y at once; the per-element product is
    identical, only the temporaries are smaller.
    """
    if G is None:
        G = D // GROUP
    states, scales = unpack_rows(packed, K, D, G)
    N = packed.shape[0]
    y = torch.empty(N, D, dtype = torch.float32, device = packed.device)
    blk = max(1, PLANE_BUDGET // max(1, D))
    for lo in range(0, N, blk):
        hi = min(lo + blk, N)
        sc = scales[lo:hi].float().view(hi - lo, G, 1).expand(-1, -1, D // G).reshape(hi - lo, D)
        y[lo:hi] = codebook[states[lo:hi]].float() * sc
    return y


def lcg_signs(row_ids: torch.Tensor, D: int, seed: int) -> torch.Tensor:
    """
    Per-element +-1 signs (N, D) fp32 from the format's LCG stream: per row r = row_ids[n],
    lcg = seed * LCG_KEY_SEED_MUL + r * LCG_KEY_ROW_MUL + 1 (uint64 wrap), then one
    lcg = lcg * LCG_STEP_MUL + LCG_STEP_ADD per element in ascending q (the stream continues
    across 256-groups in ascending group order). sign = (lcg >> 63) ? -1 : +1.

    Vectorized with the affine fast-forward: T^t(x) = A_t * x + C_t with (A_t, C_t) built from
    the binary powers of the step (all arithmetic int64 = uint64 wraparound), so the result is
    bit-exact with the scalar stream without a D-step Python loop.
    """
    dev = row_ids.device
    r = row_ids.to(torch.int64).reshape(-1, 1)
    key0 = _i64(seed * LCG_KEY_SEED_MUL + 1)
    lcg0 = key0 + r * _i64(LCG_KEY_ROW_MUL)                                            # int64 wrap
    # element q (0-based) is drawn after q+1 steps: (A, C) = T^(q+1), built from the binary
    # powers T^(2^j) = (a_j, c_j): a_{j+1} = a_j^2, c_{j+1} = c_j * (1 + a_j), all mod 2**64
    t = torch.arange(1, D + 1, dtype = torch.int64, device = dev)
    A = torch.ones(D, dtype = torch.int64, device = dev)
    C = torch.zeros(D, dtype = torch.int64, device = dev)
    aj = torch.tensor(_i64(LCG_STEP_MUL), dtype = torch.int64, device = dev)
    cj = torch.tensor(_i64(LCG_STEP_ADD), dtype = torch.int64, device = dev)
    step = 0
    while (1 << step) <= D:
        sel = (t >> step) & 1 == 1
        A = torch.where(sel, aj * A, A)
        C = torch.where(sel, aj * C + cj, C)
        cj = cj * (1 + aj)
        aj = aj * aj
        step += 1
    lcg = A.unsqueeze(0) * lcg0 + C.unsqueeze(0)                                       # (N, D) int64 wrap
    return torch.where(lcg < 0, -1.0, 1.0)


def hadamard_butterfly(y: torch.Tensor, D: int) -> torch.Tensor:
    """
    (N, D) -> (N, D): per-256-group in-place Sylvester Hadamard (unnormalized H), fp32.
    Element j is paired with j ^ b for b = 1,2,4,...,128 within each group; the low side of
    each pair gets x[j] + x[j^b], the high side x[j^b] - x[j], stage by stage. This is the
    exact scalar operation order of the fused GPU kernel and of the CPU AVX2 inverse (they
    apply the 1/16 of the orthonormal H/16 as one scale afterwards); all paths are
    bit-identical.
    """
    G = D // GROUP
    x = y.reshape(-1, G, GROUP).clone()
    j = torch.arange(GROUP, device = y.device)
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        partner = x[:, :, j ^ b]
        high = (j & b) != 0
        x = torch.where(high, partner - x, x + partner)
    return x.reshape(-1, D)


_codebooks = {}


def _codebook(device):
    key = str(device)
    if key not in _codebooks:
        _codebooks[key] = mul1_codebook(device)
    return _codebooks[key]


def decode_transform(y: torch.Tensor, col_scales: torch.Tensor, seed: int, D: int,
                     row_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Decode chain steps 3-4: transformed-space y (N, D) fp32 -> source-space (N, D) fp32.
    The exact inverse of forward_transform (the orthonormal H/16 is self-inverse, so the
    butterfly + 1/16 undoes the encode's transform up to fp32 rounding). row_ids are the
    TABLE ROW IDs used as the sign-stream keys (default: arange)."""
    if row_ids is None:
        row_ids = torch.arange(y.shape[0], dtype = torch.int64)
    had = hadamard_butterfly(y, D)
    return ((had * (1.0 / 16.0)) * lcg_signs(row_ids, D, seed).to(had.device)) * col_scales.float().to(had.device)


def dequant_rows_transformed(packed: torch.Tensor, col_scales: torch.Tensor, K: int, seed: int,
                             D: int, G: int | None = None, row_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    Full decode chain on gathered rows: packed (N, G + D*K/16) int16 + col_scales (D,) fp16
    -> source-space rows (N, D) fp32. row_ids are the TABLE ROW IDs used as the sign-stream
    keys (default: arange, i.e. the rows of a whole table). Bit-exact reference for
    ext.trellis_embed_gather.
    """
    y = dequant_rows(packed, K, _codebook(packed.device), D, G)
    return decode_transform(y, col_scales, seed, D, row_ids)


def forward_transform(w: torch.Tensor, col_scales: torch.Tensor, seed: int,
                      row_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    Encode-side forward transform, the inverse of dequant_rows_transformed's steps 3-4:
    x = w / col_scale; x *= signs (same LCG stream, r = row_ids); y = (H/16) x per group.
    The orthonormal H/16 is self-inverse, so the decode chain's butterfly + 1/16 scale
    undoes this (up to fp32 rounding). w: (N, D) any float dtype -> transformed (N, D) fp32.
    """
    N, D = w.shape
    if row_ids is None:
        row_ids = torch.arange(N, dtype = torch.int64)
    x = (w.float() / col_scales.float().unsqueeze(0)) * lcg_signs(row_ids, D, seed).to(w.device)
    return hadamard_butterfly(x, D) * (1.0 / 16.0)


def group_prescale(w: torch.Tensor, K: int, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-group pre-scale exactly as trellis VQ stores it (mirrors conversion/ngram.py's
    quantize_rows heuristic, applied per 256-column group; the ring still spans the full
    row, only the pre-scale and the stored scales are per group):

        rms, absmax_norm per group
        cs_row   = clamp(gamma / absmax_norm, CS_MIN, cs_hi)     (CS_HEURISTIC per K)
        scale0   = clamp(rms / cs_row, 1e-8).to(fp16), where(>0, scale0, 1.0)
        w_pre    = w / scale0

    scale0 is the heuristic pre-scale, hence the stored G scale words whenever no LS refit
    runs (this function works in transformed space and cannot see the source rows a refit
    would fit against; ls_refit_scales is the refit and conversion/embed.py supplies it).
    Returns (w_pre (N, D) fp32 contiguous, scale0 (N, G) fp16).
    """
    # group_prescale uses the module-level CS_HEURISTIC/CS_MIN
    N, D = w.shape
    assert D % group == 0
    G = D // group
    gamma, cs_hi = CS_HEURISTIC[K]
    wg = w.view(N, G, group)
    rms = wg.square().mean(dim = 2).sqrt()
    absmax_norm = wg.abs().amax(dim = 2) / rms.clamp(min = 1e-12)
    cs_row = (gamma / absmax_norm.clamp(min = 1e-6)).clamp(min = CS_MIN, max = cs_hi)
    # the fp16 max cap is a storage bound, not a heuristic one: the scale0 word is stored
    # verbatim and the fused kernel feeds it straight into the butterfly, so an fp16 overflow
    # (transformed-group rms > ~3*65504, reachable when a column scale sits at the fp16 floor
    # while the source values in that group are normal) would turn the row into NaN embeddings
    scale0 = (rms / cs_row).clamp(min = 1e-8, max = 65504.0).to(torch.float16).float()
    scale0 = torch.where(scale0 > 0, scale0, 1.0)
    w_pre = (wg / scale0.unsqueeze(2)).view(N, D).contiguous()
    return w_pre, scale0.to(torch.float16)


def ls_refit_scales(states: torch.Tensor, w: torch.Tensor, scale0: torch.Tensor,
                    col_scales: torch.Tensor, seed: int, row_ids: torch.Tensor, K: int, D: int,
                    G: int | None = None, codebook: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """
    Least-squares refit of the stored per-group scale words.

    The decode is linear in the scale word (the Hadamard butterfly is per 256-column group,
    the LCG signs and the column scales are elementwise), so with q = the SOURCE-space decode
    of these states at unit scale, the reconstruction of group g is exactly s_g * q_g and

        s*_g = (w_g . q_g) / (q_g . q_g)

    minimizes the per-group source-space squared error over all s. scale0 is itself an fp16
    value, hence just another candidate: the error is a parabola in s, monotone in |s - s*|,
    and fp16(s*) is the closest fp16 point to s*, so in exact arithmetic the refit cannot
    raise the error of any group (in fp32 one group in ~20 000 can end marginally worse from
    decode rounding; the aggregate always improves). The measured effect is small because the
    pre-scale heuristic already sits within half an fp16 step of the optimum for most groups:
    +0.01..0.02 dB SQNR over K = 4..8, with ~80% of the stored words unchanged at K = 8.

    states (N, D) int16/int64 Viterbi states (only the low K bits are the codes: the ring
    windows are rebuilt with ring_states_from_codes, exactly as the decoder does them),
    w (N, D) fp32 SOURCE rows, scale0 (N, G) fp16 heuristic pre-scales, col_scales (D,) fp16,
    row_ids the TABLE ROW IDs used as sign-stream keys. Whole-chunk, no blocking: the
    transients are the LUT gather (int64), the unit-scale decode and the sign plane, i.e.
    about 27 bytes per element of the chunk (1.1 GiB measured at chunk_rows = 8192, D = 5120).

    Returns (scales (N, G) fp16 - the words to store, info dict). Guards: a group whose q is
    identically zero (all-zero source row, or column scales at the fp16 floor with an
    all-zero group) gets scale word 0, which is also the pad-row contract; an fp16-non-finite
    s* (only reachable if the LS optimum leaves the fp16 range) falls back to scale0 instead
    of storing inf into a word the fused kernel feeds straight into the butterfly. info:
        groups        refitted (row, group) pairs
        moved         groups whose fp16 word differs from scale0
        rel_shift     sum |s/scale0 - 1| over the scale0 > 0 groups
        nonfinite     groups whose fp16(s*) was non-finite (fell back to scale0)
        degenerate    groups with q == 0 (scale word 0)
        err_sq_s0     sum |w - scale0 * q|^2, fp64
        err_sq_refit  sum |w - scales * q|^2 at the STORED words, fp64
    The two error sums are analytic (same pass, fp64: the fp32 c - b^2/a form cancels away at
    high SQNR), so they are diagnostics only - conversion/embed.py still reports SQNR/rfn on
    the decoder output.

    Rows and groups are independent, so any chunking of the caller's is value-identical.
    """
    if G is None:
        G = D // GROUP
    N = states.shape[0]
    dev = states.device
    cb = _codebook(dev) if codebook is None else codebook
    s0 = scale0.to(dev)
    ring = ring_states_from_codes(states.to(torch.int64) & ((1 << K) - 1), K)
    q = cb[ring].float()
    q = ((hadamard_butterfly(q, D) * (1.0 / 16.0)) *
         lcg_signs(row_ids, D, seed).to(dev)) * col_scales.float().to(dev)
    qg = q.view(N, G, GROUP)
    wg = w.to(dev).view(N, G, GROUP)
    a = qg.square().sum(2, dtype = torch.float64)                      # (N, G) fp64 moments
    b = (wg * qg).sum(2, dtype = torch.float64)
    c = wg.square().sum(2, dtype = torch.float64)
    # a == 0 (nothing to fit) must give 0, not nan: divide by inf instead of by 0
    s = (b / torch.where(a > 0, a, torch.full_like(a, float("inf")))).to(torch.float32)
    sf = s.to(torch.float16)
    bad = torch.isfinite(sf).logical_not_()
    if bool(bad.any()):
        # an LS optimum outside the fp16 range is not representable: fall back to the
        # heuristic word (capped in group_prescale, hence finite), and to 0 if even that is
        # not finite - a group that decodes to 0 beats inf fed into the butterfly
        sf = torch.where(bad, torch.where(torch.isfinite(s0), s0, torch.zeros_like(s0)), sf)
    sfa = sf.float()
    live = s0 > 0
    info = {
        "groups": N * G,
        "moved": int((sf != s0).sum()),
        "rel_shift": float((sfa[live] / s0[live].float() - 1.0).abs().sum()),
        "nonfinite": int(bad.sum()),
        "degenerate": int((a == 0).sum()),
        "err_sq_s0": float((c - 2.0 * s0.double() * b + s0.double().square() * a).sum()),
        "err_sq_refit": float((c - 2.0 * sfa.double() * b + sfa.double().square() * a).sum()),
    }
    return sf, info


def quantize_rows_grouped(w: torch.Tensor, K: int, D: int, encode_fn,
                          group: int = GROUP, refit = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode-chain companion of forward_transform for one chunk of transformed rows: nan guard,
    per-group pre-scale (group_prescale), Viterbi search through encode_fn(w_pre, K, D) ->
    states (N, D), and pack_rows with the stored scale words. Returns (packed
    (N, G + D*K/16) int16, scales (N, G) fp16 - the words actually stored).

    refit: optional callable refit(states (N, D) int64, scale0 (N, G) fp16) ->
    ((N, G) fp16, info) that replaces the heuristic pre-scales as the stored scale words
    (ls_refit_scales is the implementation; conversion/embed.py supplies it with the source
    rows, which the transformed-space encoder cannot see). The refit only refits per-group
    magnitudes, so the code words of a table encoded with or without it are bit-identical.
    Without refit, the fp16 pre-scales ARE the stored scale words.

    All-zero source rows (vocab pad slots) pack to all-zero words deterministically,
    bypassing the encoder: group_prescale's where(scale0 > 0, scale0, 1.0) guard would
    otherwise store a 1.0 scale for the empty groups and the Viterbi pass would emit
    codebook garbage for them (max |dequant| ~55 measured). The contract requires pad
    rows to dequant to exactly 0.0 (zero scale words -> zero output regardless of
    codes or sign stream).
    """
    w = torch.nan_to_num(w.float(), nan = 0.0, posinf = 0.0, neginf = 0.0)
    w_pre, scale0 = group_prescale(w, K, group)
    states = encode_fn(w_pre, K, D)
    scales = scale0
    if refit is not None:
        scales, _ = refit(states.to(torch.int64), scale0)
    packed = pack_rows(states.to(torch.int64), scales, K, D, D // group)
    zero_rows = (w == 0).all(dim = 1)
    if zero_rows.any():
        packed[zero_rows] = 0
        scales = scales.clone()
        scales[zero_rows] = 0
    return packed, scales
