
// Disable for now
const bool output_had = false;

template
<
    int bits,
//    bool output_had,
    bool c_fp32,
    int TILESIZE_M,
    int TILESIZE_K,
    int TILESIZE_N,
    int SH_STAGES,
    int FRAG_STAGES
>
__global__ __launch_bounds__(NUM_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ locks,
    const uint16_t* __restrict__ sv
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_K = TILEBLOCKS_K;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (NUM_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = 4 * NUM_THREADS;                                       // in floats
    // TODO: Maybe flush global->shared pipeline before reduction step so sh_c can share memory with sh_a and sh_b

    // Sanity checks
    static_assert(NUM_THREADS == 256);
    static_assert(TILESIZE_M % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % NUM_THREADS;
    int sub_k = threadIdx.x / NUM_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    Dim3 size = { size_m, size_k, size_n };
    Dim3 tiles = { CEIL_DIVIDE(size_m, TILESIZE_M), size_k / TILESIZE_K, size_n / TILESIZE_N };
    Dim3 blocks = { 1, tiles.k * TILEBLOCKS_K, tiles.n * TILEBLOCKS_N };

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles.numel_b() * blockIdx.x / num_slices;
    int slice_end = tiles.numel_b() * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles.k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles.k); };

    // Batch dimension
    int slice_m = index_m(slice_beg);
    int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const int sh0_a_stride_k = TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, NUM_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * NUM_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * NUM_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        pred_a_gl[i] = m < max_m;
    }

    int gl_b_stride_k = blocks.n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, NUM_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * NUM_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * NUM_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * blocks.n * 256 / 16 * bits / 8 * k + n;
        pred_b_gl[i] = i * NUM_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles.k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles.k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    register FragA frag_a[FRAG_STAGES][FRAGS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[FRAGS_M][FRAGS_N_PER_WARP];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles.k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    // TODO: Rearrange into ldmatrix friendly layout while loading?
                    // @p seems to crash on Blackwell but does not perform better on Ampere and Ada anyway
                    // cp_async_pred(sh + NUM_THREADS * i + t, gl + load_a_gl[i], pred_a_gl[i]);
                    if (pred_a_gl[i]) cp_async(sh + NUM_THREADS * i + t, gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // @p seems to crash on Blackwell but does not perform better on Ampere and Ada anyway
                    // cp_async_pred(sh + NUM_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + NUM_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments
        {
            // TODO: Resolve bank conflicts
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int c = lane_id / 16;
            int4* sha = (int4*) sh1_a_ptr + r * TILESIZE_K / 8 + c;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                ldsm4(frag_a[buf][m], sha + (m * 16) * TILESIZE_K / 8 + sub_k * 16 / 8);
        }

        // B fragments
        int r0 = lane_id / 2;
        int c0 = (lane_id % 2) * 8;

        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits>(shb, r0 * 16 + c0, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n] = {};
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i, int m, int n)
        {
            // TODO: Shuffle to avoid bank conflicts here? Doesn't seem to be a bottleneck
            // TODO: Always accumulates entire C fragment, could be limited when size_m < 16
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                for (int i = 0; i < 4; ++i)
                    *sh_red++ = frag_c[m][n][i];
            }
            __syncthreads();
        };

        auto add = [&] (int i, int m, int n)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                for (int i = 0; i < 4; ++i)
                    frag_c[m][n][i] += *sh_red++;
            }
            __syncthreads();
        };

        for (int m = 0; m < FRAGS_M; ++m)
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1, m, n);
                add(0, m, n);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1, m, n);
                add(0, m, n);
                store(2, m, n);
                add(0, m, n);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3, m, n);
                add(2, m, n);
                store(1, m, n);
                add(0, m, n);
                store(2, m, n);
                add(0, m, n);
            }
        }
    };

    // Output hadamard transform
    auto apply_output_had = [&] ()
    {
    /*
        auto shuffle_had_fx32 = [&](float v)
        {
            for (int i = 1; i < 32; i <<= 1)
            {
                float pv = __shfl_xor_sync(0xffffffff, v, i);
                uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
                int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
                *vi ^= (sfm & 0x80000000);
                v = v + pv;
            }
            return v;
        };

        // Operates on 1x128 tiles
        int n_tiles = TILESIZE_N / 128;
        int m_tiles = max_m;
        int num_tiles = n_tiles * m_tiles;

        // One warp per tile
        int tile_idx = threadIdx.x / 32;
        int tile_stride = blockDim.x / 32;

        while (tile_idx < num_tiles)
        {
            // Offset of tile slice
            int tile_i_x = tile_idx % n_tiles;
            int tile_i_y = tile_idx / n_tiles;
            half4* c_ptr = (half4*) (gl_c_ptr + size_n * tile_i_y + 128 * tile_i_x);

            // Load
            half4 v = c_ptr[lane_id];

            // 4 element had
            float v0 = __half2float(__low2half(v.x));
            float v1 = __half2float(__high2half(v.x));
            float v2 = __half2float(__low2half(v.y));
            float v3 = __half2float(__high2half(v.y));
            float h0 = v0 + v1 + v2 + v3;
            float h1 = v0 - v1 + v2 - v3;
            float h2 = v0 + v1 - v2 - v3;
            float h3 = v0 - v1 - v2 + v3;

            // 32 element had, warp shuffle
            h0 = shuffle_had_fx32(h0);
            h1 = shuffle_had_fx32(h1);
            h2 = shuffle_had_fx32(h2);
            h3 = shuffle_had_fx32(h3);
            h0 *= 0.088388347648f;  // 1/sqrt(128)
            h1 *= 0.088388347648f;
            h2 *= 0.088388347648f;
            h3 *= 0.088388347648f;
            v.x = __floats2half2_rn(h0, h1);
            v.y = __floats2half2_rn(h2, h3);

            // Sign flips
            int i = (TILESIZE_N / 128 * slice2_n + tile_i_x) * 32 + lane_id;
            uint32_t signs = (uint32_t) (sv[i / 4] >> ((i % 4) * 4));  // TODO: preload to smem (if bottleneck)
            v.x = h2xor(v.x, ((signs & 1) << 15) | ((signs & 2) << 30));  // TODO: pre-unpack (if bottleneck)
            v.y = h2xor(v.y, ((signs & 4) << 13) | ((signs & 8) << 28));

            // Store
            c_ptr[lane_id] = v;

            // Advance
            tile_idx += tile_stride;
        }
        */
    };

    // Output reduction
    auto reduce = [&] ()
    {
        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles.k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks.n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles.k;

        int n0 = warp_id * FRAGS_N_PER_WARP;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        // TODO: Use an intermediate layout to make these writes coalesce
        if (!sub_k && !first)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *c_ptr++;
                            frag_c[m][n][1] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][0] += interm.x;
                            frag_c[m][n][1] += interm.y;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *c_ptr++;
                            frag_c[m][n][3] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][2] += interm.x;
                            frag_c[m][n][3] += interm.y;
                        }
                    }
                }
            }
        }

        // All but last threadblock in column threadblocks write the intermediate result to global memory
        if (!sub_k && !last)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        // Last block also performs output hadamard (using all threads)
        if (last)
        {
            // TODO: Determine if this is a bottleneck, could maybe be done in smem or registers
            if constexpr (output_had)
            {
                __syncthreads();
                apply_output_had();
            }
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        for (int m = 0; m < FRAGS_M; ++m)
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
    };

    // Start global to shared pipeline
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            async_load_gl();
            wait_stage();
            load_frags(0);
            matmul(0);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            async_load_gl();
            wait_stage();
            load_frags(1);
            matmul(0);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;

            async_load_gl();
            wait_stage();
            load_frags(0);
            matmul(1);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            async_load_gl();
            wait_stage();
            load_frags(1);
            matmul(0);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;

            async_load_gl();
            wait_stage();
            load_frags(2);
            matmul(1);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;

            async_load_gl();
            wait_stage();
            load_frags(0);
            matmul(2);
            if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
            advance2();
            if (!slice2_iters) break;
        }
    }
}
