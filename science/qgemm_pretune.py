import sys, os
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util import Timer
from exllamav3.util.memory import free_mem
from tabulate import tabulate
import numpy as np

num_warmup_passes = 10
num_benchmark_iter_a = 20
num_benchmark_iter_b = 40
outlier_trim = 0.3
assume_cache = 384 * 1024 ** 2

devices = [
    "cuda:1",
    "cuda:2",
    "cuda:3",
]

shapes_m = [1]

shapes_k = [
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4096,
    5120,
    8192,
    12288,
    14336,
    16384,
    24576,
]

shapes_n = [
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4096,
    5120,
    8192,
    12288,
    14336,
    16384,
    24576,
    51200,
    128000,
]

shape_indices_128 = [1, 2]
shape_indices_256 = [3]
shape_indices_512 = [4]

Ks = [1, 2, 3, 4, 5, 6, 7, 8]

mgemm_bszm_io = [
    (1, 8),
    (8, 1),
]

g_spin = 0

def get_abc(K, m, k, n, device):
    proto_a = torch.randn((m, k), dtype = torch.half, device = device)
    proto_b = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.short, device = device)
    proto_c = torch.zeros((m, n), dtype = torch.half, device = device)
    proto_suh = torch.randn((k,), dtype = torch.half, device = device)
    proto_svh = torch.randn((n,), dtype = torch.half, device = device)

    # Create enough clones to cycle through to prevent L2 cache from skewing results
    proto_size = proto_a.numel() * 2 + proto_b.numel() * 2 + proto_c.numel() * 2
    num_buffers = max(assume_cache // proto_size + 1, 2)
    a = [proto_a.clone() for _ in range(num_buffers)]
    b = [proto_b.clone() for _ in range(num_buffers)]
    c = [proto_c.clone() for _ in range(num_buffers)]
    suh = [proto_suh.clone() for _ in range(num_buffers)]
    svh = [proto_svh.clone() for _ in range(num_buffers)]
    return a, b, c, suh, svh


def get_abc_m(K, m, k, n, device, bszm_in, bszm_out):
    bszm = max(bszm_in, bszm_out)
    proto_a = torch.randn((bszm_in, m, k), dtype = torch.half, device = device)
    proto_b = [torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.short, device = device) for _ in range(bszm)]
    proto_c = torch.zeros((bszm_out, m, n), dtype = torch.half, device = device)
    proto_suh = [torch.randn((k,), dtype = torch.half, device = device) for _ in range(bszm)]
    proto_svh = [torch.randn((n,), dtype = torch.half, device = device) for _ in range(bszm)]

    # Create enough clones to cycle through to prevent L2 cache from skewing results
    proto_size = proto_a.numel() * 2 + sum(p.numel() for p in proto_b) * 2 + proto_c.numel() * 2
    num_buffers = max(assume_cache // proto_size + 1, 2)
    a = [proto_a.clone() for _ in range(num_buffers)]
    b = [[proto_b_.clone() for proto_b_ in proto_b] for _ in range(num_buffers)]
    c = [proto_c.clone() for _ in range(num_buffers)]
    suh = [[proto_suh_.clone() for proto_suh_ in proto_suh] for _ in range(num_buffers)]
    svh = [[proto_svh_.clone() for proto_svh_ in proto_svh] for _ in range(num_buffers)]
    trellis = b
    ptrs_suh = [torch.tensor([suh__.data_ptr() for suh__ in suh_], dtype = torch.long, device = device) for suh_ in suh]
    ptrs_svh = [torch.tensor([svh__.data_ptr() for svh__ in svh_], dtype = torch.long, device = device) for svh_ in svh]
    ptrs_trellis = [torch.tensor([trellis__.data_ptr() for trellis__ in trellis_], dtype = torch.long, device = device) for trellis_ in trellis]
    return a, b, c, suh, svh, ptrs_suh, ptrs_svh, ptrs_trellis


def warmup(a, b, c, suh, svh, shape_idx):
    global g_spin
    num_buffers = len(a)
    for _ in range(num_warmup_passes):
        i = g_spin % num_buffers
        g_spin += 1
        ext.exl3_gemm(a[i], b[i], c[i], suh[i], a[i], svh[i], shape_idx, 0, 0, 64)


def warmup_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis):
    num_buffers = len(a)
    num_exp = ptrs_trellis[0].shape[0]
    m_indices = torch.arange(0, num_exp, dtype = torch.long, device = a[0].device).unsqueeze(0)
    K = b[0][0].shape[-1] // 16
    for i_ in range(num_warmup_passes):
        i = i_ % num_buffers
        ext.exl3_mgemm(
            a[i],
            ptrs_trellis[i],
            c[i],
            ptrs_suh[i],
            a[i],
            ptrs_svh[i],
            m_indices,
            None,
            K,
            -1,
            0,
            0,
            -1,
            -1,
            0
        )


def benchmark(a, b, c, suh, svh, shape_idx, num_iter, num_sms):
    num_buffers = len(a)
    dummy = c[0][0, 0].item()
    with Timer() as t:
        for i_ in range(num_iter):
            i = i_ % num_buffers
            ext.exl3_gemm(a[i], b[i], c[i], suh[i], a[i], svh[i], shape_idx, 0, 0, num_sms)
        dummy = c[i][0, 0].item()
    mean_time_ms = t.interval / num_iter * 1000
    return mean_time_ms


def benchmark_per_launch(a, b, c, suh, svh, shape_idx, num_iter, num_sms, trim = outlier_trim, stream = None):
    global g_spin
    device = a[0].device
    if stream is None:
        stream = torch.cuda.current_stream(device)

    torch.cuda.synchronize(device)

    # Precreate events to reduce overhead jitter
    starts = [torch.cuda.Event(enable_timing = True) for _ in range(num_iter)]
    stops  = [torch.cuda.Event(enable_timing = True) for _ in range(num_iter)]

    # Timed loop
    for it in range(num_iter):
        i = g_spin % len(a)
        g_spin += 1
        starts[it].record(stream)
        ext.exl3_gemm(a[i], b[i], c[i], suh[i], a[i], svh[i], shape_idx, 0, 0, num_sms)
        stops[it].record(stream)

    # Ensure all recorded events are complete before reading times
    torch.cuda.synchronize(device)

    # Collect per-iteration latency in milliseconds (GPU time)
    per_ms = np.array([starts[it].elapsed_time(stops[it]) for it in range(num_iter)], dtype = np.float64)

    # Robust stats
    median = float(np.median(per_ms))
    mean = float(per_ms.mean())
    std = float(per_ms.std(ddof=1)) if num_iter > 1 else 0.0

    # Simple symmetric trimming
    if 0.0 < trim < 0.5 and num_iter > 4:
        lo = np.quantile(per_ms, trim)
        hi = np.quantile(per_ms, 1.0 - trim)
        trimmed = per_ms[(per_ms >= lo) & (per_ms <= hi)]
        trimmed_mean = float(trimmed.mean()) if trimmed.size else float('nan')
    else:
        trimmed = per_ms
        trimmed_mean = mean

    return {
        "per_launch_ms": per_ms, # numpy array of length num_iter
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "trimmed_mean_ms": trimmed_mean,
        "trim_bounds_ms": (
            float(lo) if 'lo' in locals() else None,
            float(hi) if 'hi' in locals() else None
        ),
        "kept_count": int(trimmed.size),
        "total_count": int(num_iter),
    }


def benchmark_per_launch_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis, num_iter, num_sms, trim = outlier_trim, stream = None):
    device = a[0].device
    if stream is None:
        stream = torch.cuda.current_stream(device)

    torch.cuda.synchronize(device)

    num_exp = ptrs_trellis[0].shape[0]
    m_indices = torch.arange(0, num_exp, dtype = torch.long, device = a[0].device).unsqueeze(0)
    K = b[0][0].shape[-1] // 16

    # Precreate events to reduce overhead jitter
    starts = [torch.cuda.Event(enable_timing = True) for _ in range(num_iter)]
    stops  = [torch.cuda.Event(enable_timing = True) for _ in range(num_iter)]

    # Timed loop
    for it in range(num_iter):
        i = it % len(a)
        starts[it].record(stream)
        for _ in range(2):
            ext.exl3_mgemm(
                a[i],
                ptrs_trellis[i],
                c[i],
                ptrs_suh[i],
                a[i],
                ptrs_svh[i],
                m_indices,
                None,
                K,
                shape_idx,
                0,
                0,
                -1,
                -1,
                num_sms
            )
        stops[it].record(stream)

    # Ensure all recorded events are complete before reading times
    torch.cuda.synchronize(device)

    # Collect per-iteration latency in milliseconds (GPU time)
    per_ms = np.array([starts[it].elapsed_time(stops[it]) / 2 for it in range(num_iter)], dtype = np.float64)

    # Robust stats
    median = float(np.median(per_ms))
    mean = float(per_ms.mean())
    std = float(per_ms.std(ddof=1)) if num_iter > 1 else 0.0

    # Simple symmetric trimming
    if 0.0 < trim < 0.5 and num_iter > 4:
        lo = np.quantile(per_ms, trim)
        hi = np.quantile(per_ms, 1.0 - trim)
        trimmed = per_ms[(per_ms >= lo) & (per_ms <= hi)]
        trimmed_mean = float(trimmed.mean()) if trimmed.size else float('nan')
    else:
        trimmed = per_ms
        trimmed_mean = mean

    return {
        "per_launch_ms": per_ms, # numpy array of length num_iter
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "trimmed_mean_ms": trimmed_mean,
        "trim_bounds_ms": (
            float(lo) if 'lo' in locals() else None,
            float(hi) if 'hi' in locals() else None
        ),
        "kept_count": int(trimmed.size),
        "total_count": int(num_iter),
    }

def get_floor(hist_sms):
    best = float("inf"), 0, 0
    for l, s, i in hist_sms:
        if l < best[0]:
            best = l, s, i
    for l, s, i in hist_sms:
        if l < best[0] * 1.008:
            return l, s, i
    return best


def test_latencies(mod, K, m, k, n, a, b, c, suh, svh, device_idx, shape_indices):

    g_hist = []

    for shape_idx in shape_indices:

        max_sms = ext.g_get_num_sms(device_idx)

        warmup(a, b, c, suh, svh, shape_idx)

        hist_sms = []
        for num_sms in range(0, max_sms + 1, 8):
            # mean = benchmark(a, b, c, suh, svh, shape_idx, num_benchmark_iter_a, max(num_sms, 1))
            num_sms_ = max(num_sms, 1)
            stats = benchmark_per_launch(a, b, c, suh, svh, shape_idx, num_benchmark_iter_a, num_sms_)
            mean = stats["trimmed_mean_ms"]
            hist_sms.append((mean, num_sms_, shape_idx))

        _, best_sms, _ = get_floor(hist_sms)

        sms_a = best_sms - 8
        sms_b = best_sms + 8
        if sms_a < 0:
            sms_a = 0
            sms_b = 16
        if sms_b > num_sms + 4:
            sms_b = num_sms + 4
            sms_a = sms_b - 16

        warmup(a, b, c, suh, svh, shape_idx)

        for num_sms in range(sms_a, sms_b + 1, 2):
            # mean = benchmark(a, b, c, suh, svh, shape_idx, num_benchmark_iter_b, min(max(num_sms, 1), max_sms))
            num_sms_ = min(max(num_sms, 1), max_sms)
            stats = benchmark_per_launch(a, b, c, suh, svh, shape_idx, num_benchmark_iter_b, num_sms_)
            mean = stats["trimmed_mean_ms"]
            g_hist.append((mean, num_sms_, shape_idx))

    g_hist = sorted(g_hist, key = lambda t: t[1])
    best_g_lat, best_g_sms, best_g_idx = get_floor(g_hist)

    print(f"mod {mod}, dev {device_idx}, m {m:6}, k {k:6}, n {n:6}, K {K}, mean {best_g_lat:8.5f} ms, num_sms {best_g_sms:3}, shape_idx {best_g_idx}")
    return best_g_lat, best_g_sms, best_g_idx


def test_latencies_m(mod, K, m, k, n, a, b, c, suh, svh, device_idx, ptrs_suh, ptrs_svh, ptrs_trellis, shape_indices, bszm_in, bszm_out):

    best_g_lat = float("inf")
    best_g_sms = None
    best_g_idx = None

    best_l_lat = float("inf")
    best_l_sms = None
    best_l_idx = None

    for shape_idx in shape_indices:

        # Skip incompatible shapes
        # if not ext.exl3_gemm_shape_compat(shape_idx, m, k, n, K):
        #     continue

        max_sms = ext.g_get_num_sms(device_idx)

        warmup_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis)

        best_sms_lat = float("inf")
        for num_sms in range(0, max_sms + 1, 8):
            # mean = benchmark(a, b, c, suh, svh, shape_idx, num_benchmark_iter_a, max(num_sms, 1))
            stats = benchmark_per_launch_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis, num_benchmark_iter_a, max(num_sms, 1))
            mean = stats["trimmed_mean_ms"]
            if mean < best_sms_lat:
                best_sms_lat = mean
                best_sms = max(num_sms, 1)

        sms_a = best_sms - 8
        sms_b = best_sms + 8
        if sms_a < 0:
            sms_a = 0
            sms_b = 16
        if sms_b > num_sms + 4:
            sms_b = num_sms + 4
            sms_a = sms_b - 16

        warmup_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis)

        for num_sms in range(sms_a, sms_b + 1, 2):
            # mean = benchmark(a, b, c, suh, svh, shape_idx, num_benchmark_iter_b, min(max(num_sms, 1), max_sms))
            stats = benchmark_per_launch_m(a, b, c, suh, svh, shape_idx, ptrs_suh, ptrs_svh, ptrs_trellis, num_benchmark_iter_b, min(max(num_sms, 1), max_sms))
            mean = stats["trimmed_mean_ms"]
            if mean < best_g_lat:
                best_g_lat = mean
                best_g_sms = min(max(num_sms, 1), max_sms)
                best_g_idx = shape_idx
            if mean < best_l_lat:
                best_l_lat = mean
                best_l_sms = min(max(num_sms, 1), max_sms)
                best_l_idx = shape_idx

        print(f"mod {mod}, dev {device_idx}, m {m:6}, k {k:6}, n {n:6}, K {K}, mean {best_l_lat:8.5f} ms, num_sms {best_l_sms:3}, shape_idx {best_l_idx}, i/o {bszm_in}/{bszm_out}")

    print()
    print(f" --> mod {mod}, dev {device_idx}, m {m:6}, k {k:6}, n {n:6}, K {K}, mean {best_g_lat:8.5f} ms, num_sms {best_g_sms:3}, shape_idx {best_g_idx}, i/o {bszm_in}/{bszm_out}")
    print()
    return best_g_lat, best_g_sms, best_g_idx


def tune_shape(K, m, k, n, device):

    a, b, c, suh, svh = get_abc(K, m, k, n, device)
    device_idx = torch.device(device).index
    cc = ext.g_get_cc(device_idx)

    res = {
        "K": K,
        "m": m,
        "k": k,
        "n": n,
        "cc": cc,
        "bszm_in": 1,
        "bszm_out": 1,
    }
    res_128 = None
    res_256 = None
    res_512 = None

    if True:
        lat_128, sms_128, idx_128 = test_latencies(128, K, m, k, n, a, b, c, suh, svh, device_idx, shape_indices_128)
        res_128 = res.copy()
        res_128.update({"lat": lat_128, "sms": sms_128, "idx": idx_128})

    if n % 256 == 0:
        lat_256, sms_256, idx_256 = test_latencies(256, K, m, k, n, a, b, c, suh, svh, device_idx, shape_indices_256)
        if lat_128 < lat_256:
            lat_256, sms_256, idx_256 = lat_128, sms_128, idx_128
        res_256 = res.copy()
        res_256.update({"lat": lat_256, "sms": sms_256, "idx": idx_256})

    if n % 512 == 0:
        lat_512, sms_512, idx_512 = test_latencies(512, K, m, k, n, a, b, c, suh, svh, device_idx, shape_indices_512)
        if lat_128 < lat_512:
            lat_512, sms_512, idx_512 = lat_128, sms_128, idx_128
        if lat_256 < lat_512:
            lat_512, sms_512, idx_512 = lat_256, sms_256, idx_256
        res_512 = res.copy()
        res_512.update({"lat": lat_512, "sms": sms_512, "idx": idx_512})

    return res_128, res_256, res_512


def tune_shape_m(K, m, k, n, device, bszm_in, bszm_out):

    a, b, c, suh, svh, ptrs_suh, ptrs_svh, ptrs_trellis = get_abc_m(K, m, k, n, device, bszm_in, bszm_out)
    device_idx = torch.device(device).index
    cc = ext.g_get_cc(device_idx)

    res = {
        "K": K,
        "m": m,
        "k": k,
        "n": n,
        "cc": cc,
        "bszm_in": 1,
        "bszm_out": 1,
    }
    res_128 = None
    res_256 = None
    res_512 = None

    if True:
        lat_128, sms_128, idx_128 = test_latencies_m(128, K, m, k, n, a, b, c, suh, svh, device_idx, ptrs_suh, ptrs_svh, ptrs_trellis, shape_indices_128, bszm_in, bszm_out)
        res_128 = res.copy()
        res_128.update({"lat": lat_128, "sms": sms_128, "idx": idx_128})

    if n % 256 == 0:
        lat_256, sms_256, idx_256 = test_latencies_m(256, K, m, k, n, a, b, c, suh, svh, device_idx, ptrs_suh, ptrs_svh, ptrs_trellis, shape_indices_256, bszm_in, bszm_out)
        if lat_128 < lat_256:
            lat_256, sms_256, idx_256 = lat_128, sms_128, idx_128
        res_256 = res.copy()
        res_256.update({"lat": lat_256, "sms": sms_256, "idx": idx_256})

    if n % 512 == 0:
        lat_512, sms_512, idx_512 = test_latencies_m(512, K, m, k, n, a, b, c, suh, svh, device_idx, ptrs_suh, ptrs_svh, ptrs_trellis, shape_indices_512, bszm_in, bszm_out)
        if lat_128 < lat_512:
            lat_512, sms_512, idx_512 = lat_128, sms_128, idx_128
        if lat_256 < lat_512:
            lat_512, sms_512, idx_512 = lat_256, sms_256, idx_256
        res_512 = res.copy()
        res_512.update({"lat": lat_512, "sms": sms_512, "idx": idx_512})

    return res_128, res_256, res_512


def tune_gemm():
    out_128 = "struct TSample samples_128[] =\n{\n"
    out_256 = "struct TSample samples_256[] =\n{\n"
    out_512 = "struct TSample samples_512[] =\n{\n"
    for device in devices:
        for m in shapes_m:
            for k in shapes_k:
                for n in shapes_n:
                    for ki, K in enumerate(Ks):
                        res_128, res_256, res_512 = tune_shape(K, m, k, n, device)
                        r = res_128
                        if r:
                            out_128 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']} }},\n"
                        r = res_256
                        if r:
                            out_256 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']} }},\n"
                        r = res_512
                        if r:
                            out_512 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']} }},\n"
    out_128 = out_128 + "    { 0, 0, 0, 0, 0, 0, 0 }\n};"
    out_256 = out_256 + "    { 0, 0, 0, 0, 0, 0, 0 }\n};"
    out_512 = out_512 + "    { 0, 0, 0, 0, 0, 0, 0 }\n};"
    print(out_128)
    print()
    print(out_256)
    print()
    print(out_512)
    print()


def tune_mgemm():
    out_128 = "struct TMSample msamples_128[] =\n{\n"
    out_256 = "struct TMSample msamples_256[] =\n{\n"
    out_512 = "struct TMSample msamples_512[] =\n{\n"
    for device in devices:
        for m in shapes_m:
            for k in shapes_k:
                for n in shapes_n:
                    for ki, K in enumerate(Ks):
                        for (bszm_in, bszm_out) in mgemm_bszm_io:
                            res_128, res_256, res_512 = tune_shape_m(K, m, k, n, device, bszm_in, bszm_out)
                            r = res_128
                            if r:
                                out_128 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']}, {r['bszm_in']}, {r['bszm_out']} }},\n"
                            r = res_256
                            if r:
                                out_256 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']}, {r['bszm_in']}, {r['bszm_out']} }},\n"
                            r = res_512
                            if r:
                                out_512 += f"    {{ {r['cc']}, {r['K']}, {r['m']}, {r['k']}, {r['n']}, {r['idx']}, {r['sms']}, {r['bszm_in']}, {r['bszm_out']} }},\n"
    out_128 = out_128 + "    { 0, 0, 0, 0, 0, 0, 0, 0, 0 }\n};"
    out_256 = out_256 + "    { 0, 0, 0, 0, 0, 0, 0, 0, 0 }\n};"
    out_512 = out_512 + "    { 0, 0, 0, 0, 0, 0, 0, 0, 0 }\n};"
    print(out_128)
    print()
    print(out_256)
    print()
    print(out_512)
    print()


@torch.inference_mode()
def main():
    tune_gemm()
    # tune_mgemm()

if __name__ == "__main__":
    main()
