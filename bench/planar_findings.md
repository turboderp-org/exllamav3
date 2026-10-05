# AVX2 planar dword layout (bench_planar.cpp / bench_e2e.cpp)

Question: the AVX2 gather (avx2_gather_half) resolves each 8-column half-row by walking the
tile's registers -- one vpermd + blend per candidate register, O(bits) per half -- because a
row's source dwords are scattered through the tile. bench/swz_findings.md localized the K3/K4/K7
warm collapse (-30..-64% in the band-2 kernel) and the general high-bitrate compute cost to
exactly this walk plus its register pressure. Can a smarter *storage order* remove it?

Insight: within a half-row the tc-perm ring index advances by 32 per column, so the source
dwords advance by exactly `bits` per column -- they form ONE residue class mod bits of the
tile's 8*bits dwords (proved at compile time: planar_reg() static_asserts it for every
bits x row x half x word_sel). Repacking each tile's dwords so every class lives in one
8-dword register,

    new[q] = old[bits * (q % 8) + q / 8]        (dwords; integer rates)

makes every gather a SINGLE vpermd (vpermd is full 256-bit cross-lane). Orthogonal to the
group-2 tile-order swizzle (it permutes *within* tiles); the two compose. Half-integer rates
are excluded (their per-column word step is not integral, the classes do not partition the
tile); K1's permutation is the identity, so K1 keeps the existing kernels.

Design: production kernels avx2_swz2_planar (band-2 body, planar gathers, bit-exact vs
avx2_tiles by construction) + rules exl3_moe_cpu_swizzle_group (AVX2: group 2 for ALL integer
rates when planar is on; historical {1,2,5,6,8} with EXL3_MOE_CPU_PLANAR=0) and
exl3_moe_cpu_planar_layout (AVX2, integer rates 2-8). Loader repack: rehome()/install() in
moe_cpu_host.py (group permute + (8, bits) dword transpose in one strided copy); GPU staging
restore: moe_unswizzle.cu planar branch (per-dword inverse; index math mirrored and verified
CPU-side in bench_e2e test_unswizzle_mirror -- the CUDA itself is untested, no NVIDIA device
on the dev box).

Variants (bench_planar, k=2944; warm n=2944, cold n=32768 + 256 MB read-flush; medians):
  PROD  production avx2_tiles (native layout)
  PLAN  avx2_tiles body + planar gathers (research only; SWZP dominates it everywhere)
  SWZP  = production avx2_swz2_planar, group-2 + planar
  SWZG  production avx2_swz_tiles, group-2 native dwords (the adopted pre-planar kernel)

## Results (Zen 5 HX 370, 1T, taskset; Gw/s)

Warm m=1 (compute-bound regime -- the walk's signature):

    K1 18.0/18.0  K2 15.8/17.9  K3 14.8/17.6  K4 13.6/18.0  K5 8.0/17.1
    K6 6.9/17.2   K7 5.3/16.7   K8 9.9/18.1        (PROD/SWZP)

    SWZP vs PROD: K2 +13% K3 +19% K4 +33% K5 +113% K6 +146% K7 +214% K8 +84%; every rate
    lands at the ~17-18 Gw/s accumulate ceiling, i.e. the extraction is off the critical
    path. (An earlier cooler session measured the same shape at 11.5 Gw/s: +21/+40/+117/
    +143/+219/+89% for K3-K8. The gain scales with how ALU-starved the machine is.)

Cold m=1 (offloaded-expert streaming, the shipping regime):

    SWZP vs PROD: K2 +110% K3 +145% K4 +165% K5 +172% K6 +286% K7 +368% K8 +119%.
    SWZP vs the adopted SWZG (marginal value of planar over today's swizzle):
    K2 +7% K5 +111% K6 +147% K8 +73% -- note SWZG here measures WORSE than swz_findings'
    5950X numbers at K5/K6 (8.1/7.2 vs 10.6/10.7 warm-cold gap; this machine), re-confirm
    per-rate on AVX2 hardware before trusting either host's absolutes. K1: SWZG 17.9 >
    SWZP 15.2 -- binary-equivalent kernels differing only in codegen (planar is identity at
    K1), so this is layout luck; the rule keeps K1 on avx2_swz_tiles either way.

Warm m=4: SWZP vs PROD: K1 +22% K2 +22% K3 +20% K4 +22% K5 +51% K6 +62% K7 +96% K8 +44%.

Repeatability: cold within +-3% across 3 interleaved sweeps (better than swz_findings'
experience on this laptop; the 512 MB flush of the older harness was replaced with 256 MB).

Correctness: SWZP/PLAN bit-exact vs PROD in bench_planar (memcmp of full outputs, K1-K8 x
m1-4, warm+cold fixtures). bench_e2e drives the production dispatch
(make_layer/forward_raw, 4 pool threads, E=4 H=I=512, rows 1/5/9 incl. multi-chunk splits
and wide-row folds, all 8 integer rates): native-bytes layer vs loader-repacked (group-2 +
planar) layer bit-exact under EXL3_MOE_CPU_MAX_ISA=avx2 (planar active K2-K8), under the
VBMI tier (planar correctly inert), and with EXL3_MOE_CPU_PLANAR=0 (rule reverts verbatim).

## Decision

Ship as measured here on Zen 5: AVX2 tier, integer rates 2-8 swizzle to group 2 + planar;
K1 group-2 walking kernel (unchanged); half rates and other tiers untouched. Kill switch
EXL3_MOE_CPU_PLANAR=0 restores the previous rule and bytes exactly. Expected end-to-end
shape: largest gains on K5-K7 decodes and any workload that was cold/DRAM-bound at high
bitrates; K1/K2 near-neutral warm.

Follow-ups (deliberately not in this change):
- Word-pairing (dword_pair_wins analogue) under planar gathers: the register headroom that
  killed it on AVX2 is gone; may lift the m=1 ceiling further.
- Band-4 on AVX2 (planar frees ~6 ymm): longer sequential runs at m=1.
- CUDA planar inverse runs on-device without a test here; verify GPU-prefill numerics of a
  swizzled+planar layer against EXL3_MOE_CPU_PLANAR=0 once on NVIDIA hardware.
- The K1 SWZP/SWZG codegen gap suggests a -march-neutral alignment study if chasing last %.
## Confirmation on the AVX2 shipping host (2nd machine, g_isa=Avx2 native)

bench_e2e: ALL PASS (planar active K2-K8, bit-exact, incl. rows 5/9 multi-chunk/wide-fold).

Warm m=1 SWZP vs PROD (Gw/s ceilings differ from Zen 5; ratios are the story):

    K1 +5%  K2 +23%  K3 +21%  K4 +46%  K5 +190%  K6 +224%  K7 +345%  K8 +127%

Cold m=1 SWZP vs PROD:

    K1 +28%  K2 +173%  K3 +76%  K4 +87%  K5 +212%  K6 +237%  K7 +356%  K8 +150%

Cold SWZP vs SWZG (marginal planar value over the pre-planar adopted swizzle):

    K1 0%  K2 +9%  K3 +99%  K4 +109%  K5 +70%  K6 +60%  K7 +357%  K8 +100%

Notes: K1 SWZP == SWZG here (16.0 vs 16.0) -- the Zen 5 K1 gap was codegen, not layout,
confirming the K1 exemption is cost-free either way. PLAN's K4 cold outlier (3.95) is this
harness's known cold-regime variance on non-shipping variants; SWZP was monotone across the
sweep. The narrower issue width vs Zen 5 amplifies the walk's cost exactly as predicted --
the gains are LARGER on the shipping tier than on the dev box. (5950X-class DDR4 host; fill
in exact model here when recording.)
