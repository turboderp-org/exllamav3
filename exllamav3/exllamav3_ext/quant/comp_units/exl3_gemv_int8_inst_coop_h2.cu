#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

// 2.5 bpw
void* exl3_gemv_int8_coop_sel_h2(bool c_fp32, bool residual)
{
    if (c_fp32)  return residual ? (void*) exl3_gemv_int8_coop_kernel<2, true, true, true>
                                 : (void*) exl3_gemv_int8_coop_kernel<2, true, false, true>;
    else         return residual ? (void*) exl3_gemv_int8_coop_kernel<2, false, true, true>
                                 : (void*) exl3_gemv_int8_coop_kernel<2, false, false, true>;
}
