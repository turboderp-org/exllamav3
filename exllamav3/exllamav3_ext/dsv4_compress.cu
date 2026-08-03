#include <cuda_fp16.h>
#include "dsv4_compress.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

/*
Fused DeepSeek-V4 compressor step (stateful/cached path, bsz 1). Replaces the torch
composition in DSV4Compressor.forward + DSV4RingCompressorState: buffer cat, ape bias,
overlapping Ca/Cb window assembly (incl. overlap snapshot restore/save), per-column softmax
pooling, RMS norm, rope at the window positions, and the copy into the per-slot pools.

Layout/invariants (see cache/dsa.py):

  - Projected rows are position-addressed: ring row = abs_pos % buf_rows, with
    buf_rows = PAGE_SIZE + m >= max rewind + fill. Rows at abs >= pos0 come from this
    chunk's kv_new/gate_new, rows below from the ring.
  - Window w (w = 0..nw-1) emits pool entry ec0 + w (ec0 = pos0 / m) from source rows at
    abs positions [(ec0 + w) * m, (ec0 + w + 1) * m). Overlapping (CSA/indexer) adds the
    PREVIOUS window's Ca slice (columns [0, hd)) as entries 0..m-1; for w = 0 that comes
    from the overlap snapshot ring keyed (ec0 - 1) % depth (gate saved WITH its ape bias),
    or is masked out entirely when ec0 == 0.
  - The window kernel only reads ring rows at abs < pos0 and snapshot (ec0 - 1) % depth;
    the snapshot write for (ec_new - 1) % depth is done by BLOCK 0 after its restore read
    (single-block ordering -- a last-window writer could race block 0's read when
    nw % depth == 0), from kv_new / pre-pos0 ring rows only. The ring store runs as a
    second, stream-ordered kernel so chunks larger than buf_rows - fill cannot alias the
    window reads.
  - Gate softmax runs per COLUMN over the window entries, fp32, matching
    (kv * gate.softmax(dim = 2)).sum(dim = 2).
  - Norm is a weighted RMSNorm over hd; rope is GPT-J pairs on the trailing rope_dim
    columns at theta = inv_freq * (ec0 + w) * m.
  - Output entry ec0 + w is split-stored: columns [0, Wa) to dest_a, [Wa, hd) to dest_b
    (pool_c/pool_r), or all to dest_a when dest_b is null (pool_idx).

If pos_ptr is non-null the kernels read the absolute position from device memory instead
of the baked scalar, and the window grid is padded to seq / m + 1 blocks with an in-kernel
w < nw guard: launches become CUDA-graph-replayable across positions with fixed buffers.
*/

#define NUM_THREADS_STORE 256

__global__ __launch_bounds__(1024)
void dsv4_compress_windows_kernel
(
    const half* __restrict__ kv_new,     // (seq, W)
    const half* __restrict__ gate_new,   // (seq, W)
    const half* __restrict__ ring_kv,    // (buf_rows, W)
    const half* __restrict__ ring_gate,  // (buf_rows, W)
    float* __restrict__ ovl,             // (depth, 2, m, hd) or nullptr
    const float* __restrict__ ape,       // (m, W)
    const half* __restrict__ norm_w,     // (hd)
    const float eps,
    const float* __restrict__ inv_freq,  // (rd / 2)
    half* __restrict__ dest_a,           // (cap, Wa)
    half* __restrict__ dest_b,           // (cap, hd - Wa) or nullptr
    const int* __restrict__ pos_ptr,     // device position override or nullptr
    const int pos_base,
    const int seq,
    const int m,
    const int buf_rows,
    const int ovl_depth,
    const int W,
    const int hd,
    const int rd,
    const int Wa,
    const bool overlap
)
{
    extern __shared__ float sh[];        // comp[hd] + reduce[hd / 32]
    float* sh_comp = sh;
    float* sh_red = sh + hd;

    int pos0 = pos_ptr ? *pos_ptr : pos_base;
    int ec0 = pos0 / m;
    int nw = (pos0 + seq) / m - ec0;

    int w = blockIdx.x;
    if (w >= nw) return;
    int c = threadIdx.x;                 // column; blockDim.x = hd padded to a warp multiple
    bool active = c < hd;
    int t_warps = blockDim.x / 32;

    // Softmax-pooled window entry, online, fp32
    int E = overlap ? 2 * m : m;
    float mx = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int e = 0; active && e < E; ++e)
    {
        float kvv, gv;
        if (overlap && e < m && w == 0)
        {
            // Previous window's Ca slice from the snapshot ring (ape already applied)
            if (ec0 == 0) continue;
            const float* snap = ovl + (size_t) ((ec0 - 1) % ovl_depth) * 2 * m * hd;
            kvv = snap[e * hd + c];
            gv = snap[(m + e) * hd + c];
        }
        else
        {
            int abs_pos, sc;
            if (overlap && e < m)
            {
                abs_pos = (ec0 + w - 1) * m + e;
                sc = c;
            }
            else
            {
                int r = overlap ? e - m : e;
                abs_pos = (ec0 + w) * m + r;
                sc = overlap ? hd + c : c;
            }
            if (abs_pos >= pos0)
            {
                kvv = __half2float(kv_new[(size_t) (abs_pos - pos0) * W + sc]);
                gv = __half2float(gate_new[(size_t) (abs_pos - pos0) * W + sc]);
            }
            else
            {
                size_t row = (size_t) (abs_pos % buf_rows) * W;
                kvv = __half2float(ring_kv[row + sc]);
                gv = __half2float(ring_gate[row + sc]);
            }
            gv += ape[(abs_pos % m) * W + sc];
        }
        float nm = fmaxf(mx, gv);
        float alpha = __expf(mx - nm);
        float p = __expf(gv - nm);
        l = l * alpha + p;
        acc = acc * alpha + p * kvv;
        mx = nm;
    }
    float comp = active ? acc / l : 0.0f;

    // Weighted RMS norm over the hd columns
    float sq = comp * comp;
    for (int offset = 16; offset > 0; offset >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, offset);
    if ((c % 32) == 0) sh_red[c / 32] = sq;
    __syncthreads();
    if (c < 32)
    {
        sq = c < t_warps ? sh_red[c] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sq += __shfl_down_sync(0xffffffffu, sq, offset);
        if (c == 0) sh_red[0] = sq;
    }
    __syncthreads();
    float rmr = rsqrtf(sh_red[0] / (float) hd + eps);
    float normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);
    if (active) sh_comp[c] = normed;
    __syncthreads();
    if (!active) return;

    // GPT-J rope on the trailing rd columns at the window position
    float out = normed;
    int c0 = hd - rd;
    if (c >= c0)
    {
        int p = (c - c0) / 2;
        float theta = inv_freq[p] * (float) ((ec0 + w) * m);
        float cs = __cosf(theta);
        float sn = __sinf(theta);
        float v_e = sh_comp[c0 + p * 2];
        float v_o = sh_comp[c0 + p * 2 + 1];
        out = ((c - c0) & 1) ? (v_o * cs + v_e * sn) : (v_e * cs - v_o * sn);
    }

    size_t drow = (size_t) (ec0 + w);
    if (!dest_b || c < Wa)
        dest_a[drow * Wa + c] = __float2half_rn(out);
    else
        dest_b[drow * (size_t) (hd - Wa) + (c - Wa)] = __float2half_rn(out);

    // Overlap snapshot for the next call: last window's Ca slice, written by block 0 AFTER
    // its restore read (see header comment). Sources are kv_new / pre-pos0 ring rows only,
    // both stable within this kernel
    if (overlap && w == 0)
    {
        float* snap = ovl + (size_t) ((ec0 + nw - 1) % ovl_depth) * 2 * m * hd;
        for (int e = 0; e < m; ++e)
        {
            int abs_pos = (ec0 + nw - 1) * m + e;
            float kvv, gv;
            if (abs_pos >= pos0)
            {
                kvv = __half2float(kv_new[(size_t) (abs_pos - pos0) * W + c]);
                gv = __half2float(gate_new[(size_t) (abs_pos - pos0) * W + c]);
            }
            else
            {
                size_t row = (size_t) (abs_pos % buf_rows) * W;
                kvv = __half2float(ring_kv[row + c]);
                gv = __half2float(ring_gate[row + c]);
            }
            snap[e * hd + c] = kvv;
            snap[(m + e) * hd + c] = gv + ape[(abs_pos % m) * W + c];
        }
    }
}


__global__ __launch_bounds__(NUM_THREADS_STORE)
void dsv4_compress_store_kernel
(
    const half* __restrict__ kv_new,     // (seq, W)
    const half* __restrict__ gate_new,   // (seq, W)
    half* __restrict__ ring_kv,          // (buf_rows, W)
    half* __restrict__ ring_gate,        // (buf_rows, W)
    const int* __restrict__ pos_ptr,
    const int pos_base,
    const int seq,
    const int buf_rows,
    const int W,
    const int j0                         // first chunk row to store (seq - buf_rows clamp)
)
{
    int pos0 = pos_ptr ? *pos_ptr : pos_base;
    size_t i = (size_t) blockIdx.x * NUM_THREADS_STORE + threadIdx.x;
    size_t total = (size_t) (seq - j0) * W;
    if (i >= total) return;
    int j = j0 + (int) (i / W);
    int c = (int) (i % W);
    size_t src = (size_t) j * W + c;
    size_t dst = (size_t) ((pos0 + j) % buf_rows) * W + c;
    ring_kv[dst] = kv_new[src];
    ring_gate[dst] = gate_new[src];
}


// SWA ring append for the graphed decode step: rows land at ring[*pos - *ring_beg + j].
// The shift/rebase branches are handled host-side BEFORE replay (rare, page-granular);
// this kernel assumes the chunk fits, which the host guarantees by correcting ring_beg
// first. pos/ring_beg are device scalars so the captured graph replays across steps.

__global__ __launch_bounds__(NUM_THREADS_STORE)
void dsv4_ring_append_kernel
(
    const half* __restrict__ kv,         // (seq, D)
    half* __restrict__ ring,             // (ring_rows, D)
    const int* __restrict__ pos_ptr,
    const int* __restrict__ ring_beg_ptr,
    const int seq,
    const int D,
    const int ring_rows
)
{
    int offset = *pos_ptr - *ring_beg_ptr;
    size_t i = (size_t) blockIdx.x * NUM_THREADS_STORE + threadIdx.x;
    size_t total = (size_t) seq * D;
    if (i >= total) return;
    int j = (int) (i / D);
    int c = (int) (i % D);
    int row = offset + j;
    if (row < 0 || row >= ring_rows) return;
    ring[(size_t) row * D + c] = kv[(size_t) j * D + c];
}

void dsv4_ring_append_gr
(
    const at::Tensor& kv,
    at::Tensor& ring,
    const at::Tensor& pos,
    const at::Tensor& ring_beg,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(kv.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(kv, kHalf);
    TORCH_CHECK_DTYPE(ring, kHalf);
    TORCH_CHECK_DTYPE(pos, kInt);
    TORCH_CHECK_DTYPE(ring_beg, kInt);
    TORCH_CHECK_SHAPES(kv, -1, ring, -1, 1);
    int seq = kv.size(0);
    int D = kv.size(-1);
    size_t total = (size_t) seq * D;
    dsv4_ring_append_kernel<<<CEIL_DIVIDE(total, NUM_THREADS_STORE), NUM_THREADS_STORE, 0, stream>>>
    (
        (const half*) kv.data_ptr(), (half*) ring.data_ptr(),
        (const int*) pos.data_ptr(), (const int*) ring_beg.data_ptr(),
        seq, D, (int) ring.size(0)
    );
    cuda_check(cudaPeekAtLastError());
}

void dsv4_ring_append
(
    const at::Tensor& kv,
    at::Tensor ring,
    const at::Tensor& pos,
    const at::Tensor& ring_beg
)
{
    dsv4_ring_append_gr(kv, ring, pos, ring_beg, nullptr);
}

void dsv4_compress_gr
(
    const at::Tensor& kv_new,
    const at::Tensor& gate_new,
    at::Tensor& ring_kv,
    at::Tensor& ring_gate,
    c10::optional<at::Tensor>& ovl,
    const at::Tensor& ape,
    const at::Tensor& norm_w,
    float rms_norm_eps,
    const at::Tensor& inv_freq,
    at::Tensor& dest_a,
    c10::optional<at::Tensor>& dest_b,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(kv_new.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(kv_new, kHalf);
    TORCH_CHECK_DTYPE(gate_new, kHalf);
    TORCH_CHECK_DTYPE(ring_kv, kHalf);
    TORCH_CHECK_DTYPE(ring_gate, kHalf);
    TORCH_CHECK_DTYPE(ape, kFloat);
    TORCH_CHECK_DTYPE(norm_w, kHalf);
    TORCH_CHECK_DTYPE(inv_freq, kFloat);
    TORCH_CHECK_DTYPE(dest_a, kHalf);
    TORCH_CHECK_SHAPES_FULL(kv_new, gate_new);
    TORCH_CHECK_SHAPES_FULL(ring_kv, ring_gate);
    TORCH_CHECK_SHAPES(kv_new, -1, ring_kv, -1, 1);
    TORCH_CHECK_SHAPES(kv_new, -1, ape, -1, 1);

    int seq = kv_new.size(0);
    int W = kv_new.size(-1);
    int buf_rows = ring_kv.size(0);
    int Wa = dest_a.size(-1);
    int hd = Wa + (dest_b ? dest_b.value().size(-1) : 0);
    int rd = inv_freq.size(0) * 2;
    bool overlap = W == 2 * hd;
    TORCH_CHECK(overlap || W == hd, "dsv4_compress: W must be hd or 2 * hd");
    TORCH_CHECK(norm_w.size(0) == hd, "dsv4_compress: norm weight size mismatch");
    TORCH_CHECK(rd % 2 == 0 && rd <= hd && hd <= 1024, "dsv4_compress: bad dims");
    TORCH_CHECK(!overlap || ovl, "dsv4_compress: overlapping mode requires snapshot ring");
    int ovl_depth = ovl ? ovl.value().size(0) : 1;
    if (ovl)
    {
        TORCH_CHECK_DTYPE(ovl.value(), kFloat);
        TORCH_CHECK(ovl.value().size(2) == m && ovl.value().size(3) == hd, "dsv4_compress: snapshot ring shape mismatch");
    }

    const int* pos_ptr = nullptr;
    if (position_tensor)
    {
        TORCH_CHECK_DTYPE(position_tensor.value(), kInt);
        pos_ptr = (const int*) position_tensor.value().data_ptr();
    }

    int nw = (position + seq) / m - position / m;
    int grid_w = pos_ptr ? seq / m + 1 : nw;

    if (grid_w > 0)
    {
        int threads = CEIL_DIVIDE(hd, 32) * 32;
        size_t shmem = (hd + threads / 32) * sizeof(float);
        dsv4_compress_windows_kernel<<<grid_w, threads, shmem, stream>>>
        (
            (const half*) kv_new.data_ptr(),
            (const half*) gate_new.data_ptr(),
            (const half*) ring_kv.data_ptr(),
            (const half*) ring_gate.data_ptr(),
            (float*) OPTPTR(ovl),
            (const float*) ape.data_ptr(),
            (const half*) norm_w.data_ptr(),
            rms_norm_eps,
            (const float*) inv_freq.data_ptr(),
            (half*) dest_a.data_ptr(),
            (half*) OPTPTR(dest_b),
            pos_ptr,
            position,
            seq, m, buf_rows, ovl_depth, W, hd, rd, Wa,
            overlap
        );
        cuda_check(cudaPeekAtLastError());
    }

    int j0 = seq > buf_rows ? seq - buf_rows : 0;
    size_t total = (size_t) (seq - j0) * W;
    dsv4_compress_store_kernel<<<CEIL_DIVIDE(total, NUM_THREADS_STORE), NUM_THREADS_STORE, 0, stream>>>
    (
        (const half*) kv_new.data_ptr(),
        (const half*) gate_new.data_ptr(),
        (half*) ring_kv.data_ptr(),
        (half*) ring_gate.data_ptr(),
        pos_ptr,
        position,
        seq, buf_rows, W, j0
    );
    cuda_check(cudaPeekAtLastError());
}


void dsv4_compress
(
    const at::Tensor& kv_new,
    const at::Tensor& gate_new,
    at::Tensor ring_kv,
    at::Tensor ring_gate,
    c10::optional<at::Tensor> ovl,
    const at::Tensor& ape,
    const at::Tensor& norm_w,
    float rms_norm_eps,
    const at::Tensor& inv_freq,
    at::Tensor dest_a,
    c10::optional<at::Tensor> dest_b,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m
)
{
    dsv4_compress_gr
    (
        kv_new,
        gate_new,
        ring_kv,
        ring_gate,
        ovl,
        ape,
        norm_w,
        rms_norm_eps,
        inv_freq,
        dest_a,
        dest_b,
        position,
        position_tensor,
        m,
        nullptr
    );
}
