#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

// 3.5 bpw
void* exl3_gemv_int8_coop_sel_h3(bool c_fp32, bool residual)
{
    if (c_fp32)  return residual ? (void*) exl3_gemv_int8_coop_kernel<3, true, true, true>
                                 : (void*) exl3_gemv_int8_coop_kernel<3, true, false, true>;
    else         return residual ? (void*) exl3_gemv_int8_coop_kernel<3, false, true, true>
                                 : (void*) exl3_gemv_int8_coop_kernel<3, false, false, true>;
}
