#include <cuda_fp16.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stloader.h"
#include "hadamard.h"

#include "norm.cuh"
#include "hgemm.cuh"
#include "rope.cuh"
#include "activation.cuh"
#include "softcap.cuh"
#include "routing.cuh"
#include "gdn.cuh"
#include "causal_conv1d.cuh"
#include "add.cuh"

#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/util.cuh"
#include "quant/exl3_devctx.cuh"

#include "generator/strings.h"
#include "generator/sampling_basic.cuh"
#include "generator/gumbel.cuh"
#include "generator/rep_pen.cuh"
#include "generator/cache.cuh"

#include "cache/q_cache.cuh"

#include "histogram.cuh"

#include "parallel/context.cuh"
#include "parallel/broadcast.cuh"
#include "parallel/barrier.cuh"
#include "parallel/gather.cuh"
#include "parallel/all_reduce.cuh"

#include "libtorch/gated_delta_net.h"
#include "libtorch/linear.h"
#include "libtorch/gated_rmsnorm.h"
#include "libtorch/mlp.h"
#include "libtorch/blocksparse_mlp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stloader_read", &stloader_read, "stloader_read");
    m.def("stloader_open_file", &stloader_open_file, "stloader_open_file");
    m.def("stloader_close_file", &stloader_close_file, "stloader_close_file");
    py::class_<TensorLoadJob>(m, "TensorLoadJob")
        .def(py::init<std::vector<uintptr_t>, size_t, size_t, uintptr_t, bool, bool, bool, int>());
    m.def("stloader_deferred_cpu", &stloader_deferred_cpu, py::arg("jobs"));
    m.def("stloader_deferred_cuda", &stloader_deferred_cuda, py::arg("jobs"), py::arg("max_chunk_size"));

    m.def("rms_norm", &rms_norm, "rms_norm");
    m.def("gated_rms_norm", &gated_rms_norm, "gated_rms_norm");
    m.def("softcap", &softcap, "softcap");

    m.def("routing_ds3_nogroup", &routing_ds3_nogroup, "routing_ds3_nogroup");
    m.def("routing_std", &routing_std, "routing_std");

    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("pg_init_context", &pg_init_context, "pg_init_context");
    m.def("pg_broadcast", &pg_broadcast, "pg_broadcast");
    m.def("pg_broadcast_ll", &pg_broadcast_ll, "pg_broadcast_ll");
    m.def("pg_barrier", &pg_barrier, "pg_barrier");
    m.def("pg_gather", &pg_gather, "pg_gather");
    m.def("pg_all_reduce", &pg_all_reduce, "pg_all_reduce");
    m.def("pg_all_reduce_cpu", &pg_all_reduce_cpu, "pg_all_reduce_cpu");
    m.def("run_cpu_reduce_jobs", &run_cpu_reduce_jobs, "run_cpu_reduce_jobs");
    m.def("end_cpu_reduce_jobs", &end_cpu_reduce_jobs, "end_cpu_reduce_jobs");

    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("test_distribution", &test_distribution, "test_distribution");
    m.def("decode", &decode, "decode");
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_signs", &pack_signs, "pack_signs");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");
    m.def("exl3_gemv", &exl3_gemv, "exl3_gemv");
    m.def("exl3_gemm_num_kernel_shapes", &exl3_gemm_num_kernel_shapes, "exl3_gemm_num_kernel_shapes");
    m.def("exl3_gemm_shape_compat", &exl3_gemm_shape_compat, "exl3_gemm_shape_compat");
    m.def("g_get_cc", &g_get_cc, "g_get_cc");
    m.def("g_get_num_sms", &g_get_num_sms, "g_get_num_sms");
    m.def("exl3_mgemm", &exl3_mgemm, "exl3_mgemm");
    m.def("hgemm", &hgemm, "hgemm");
    m.def("rope", &rope, "rope");
    m.def("gen_mrope_pos_ids", &gen_mrope_pos_ids, "gen_mrope_pos_ids");
    m.def("silu_mul", &silu_mul, "silu_mul");
    m.def("gelu_mul", &gelu_mul, "gelu_mul");
    m.def("relu2_mul", &relu2_mul, "relu2_mul");
    m.def("xielu", &xielu, "xielu");
    m.def("add_sigmoid_gate", &add_sigmoid_gate, "add_sigmoid_gate");
    m.def("add_sigmoid_gate_proj", &add_sigmoid_gate_proj, "add_sigmoid_gate_proj");
    m.def("add", &add, "add");

    m.def("gated_delta_net_fused_op", &gated_delta_net_fused_op, "gated_delta_net_fused_op");
    m.def("cuda_recurrent_gated_delta_rule", &cuda_recurrent_gated_delta_rule, "cuda_recurrent_gated_delta_rule");

    m.def("argmax_sample", &argmax_sample, "argmax_sample");
    m.def("gumbel_sample", &gumbel_sample, "gumbel_sample");
    m.def("gumbel_noise_f16", &gumbel_noise_f16, "gumbel_noise_f16");
    m.def("gumbel_noise_f32", &gumbel_noise_f32, "gumbel_noise_f32");
    m.def("gumbel_noise_log", &gumbel_noise_log, "gumbel_noise_log");
    m.def("apply_rep_pens", &apply_rep_pens, "apply_rep_pens");
    m.def("apply_pres_freq_pens", &apply_pres_freq_pens, "apply_pres_freq_pens");

    m.def("cache_rotate", &cache_rotate, "cache_rotate");

    m.def("partial_strings_match", &partial_strings_match, "partial_strings_match");
    m.def("count_match_tensor", &count_match_tensor, "count_match_tensor");

    m.def("quant_cache_cont", &quant_cache_cont, "quant_cache_cont");
    m.def("dequant_cache_cont", &dequant_cache_cont, "dequant_cache_cont");
    m.def("quant_cache_paged", &quant_cache_paged, "quant_cache_paged");
    m.def("dequant_cache_paged", &dequant_cache_paged, "dequant_cache_paged");

    m.def("count_inf_nan", &count_inf_nan, "count_inf_nan");
    m.def("histogram", &histogram, "histogram");

    m.def("blocksparse_mlp_routing", &blocksparse_mlp_routing, "blocksparse_mlp_routing");

    #include "libtorch/linear_bc.h"
    #include "libtorch/gated_delta_net_bc.h"
    #include "libtorch/gated_rmsnorm_bc.h"
    #include "libtorch/mlp_bc.h"
    #include "libtorch/blocksparse_mlp_bc.h"
}