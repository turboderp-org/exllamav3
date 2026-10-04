#pragma once

// Cross-lane moves for the hyper-connection kernels (hc_mix.cu) on DPP.
//
// __shfl_down / __shfl_xor lower to ds_bpermute, an LDS round trip, and these kernels chain many of them:
// the partials kernel's M + 1 warp sums at the end of an otherwise short kernel, and the finalize kernel's
// sinkhorn, a serial chain of shuffles and divides on one warp. Every one of those moves stays within a row
// of 16 lanes (or row 1 -> row 0 for the offset-16 step), which DPP does in the VALU: row_shl:n for
// __shfl_down by n < 16, v_permlanex16 for 16, quad_perm for xor 1 / 2, row_xmask for xor 4 / 8. The value
// each consumed lane receives and the order of every add are unchanged, so the results are bit-identical.

template <int CTRL>
__device__ __forceinline__ float hc_dpp(float v)
{
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, false));
}

// Each lane receives the same lane of the other row of 16 (row 0 lane i <- lane i + 16 and vice versa)
__device__ __forceinline__ float hc_permlanex16(float v)
{
    return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

#define HC_DPP_ROW_SHL(n) (0x100 + (n))
#define HC_DPP_QUAD_XOR1  0xB1        // quad_perm [1, 0, 3, 2]
#define HC_DPP_QUAD_XOR2  0x4E        // quad_perm [2, 3, 0, 1]
#define HC_DPP_ROW_XMASK(n) (0x160 + (n))

// The __shfl_down tree (offsets 16, 8, 4, 2, 1); lane 0 ends with the same sum in the same order
__device__ __forceinline__ float hc_warp_sum_lane0(float v)
{
    v += hc_permlanex16(v);
    v += hc_dpp<HC_DPP_ROW_SHL(8)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(4)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(2)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(1)>(v);
    return v;
}

// __shfl_xor(v, O) for O in {1, 2, 4, 8}, within a row of 16
template <int O>
__device__ __forceinline__ float hc_xor16(float v)
{
    if constexpr (O == 1) return hc_dpp<HC_DPP_QUAD_XOR1>(v);
    else if constexpr (O == 2) return hc_dpp<HC_DPP_QUAD_XOR2>(v);
    else return hc_dpp<HC_DPP_ROW_XMASK(O)>(v);
}
