"""Real paged-decode kernels against dense FP32 attention; no model or native build.

Run from a prebuilt checkout (device supplied by the standard test fixture):
    python -m pytest -q tests/kernels/attention/test_paged_fixed_partition_gpu.py
"""
import importlib.machinery
import importlib.util


def test_fixed_partitions_dense_reference_and_page_rebase(device):
    import pytest
    import torch

    spec = importlib.util.find_spec("exllamav3_ext")
    if spec is None or not spec.origin or not any(
        spec.origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES
    ):
        pytest.skip("Needs a prebuilt native extension; this test never builds it")
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode as decode

    generator = torch.Generator(device="cpu").manual_seed(1234)
    page_size, heads, kv_heads, dim = 256, 8, 2, 96
    # Both exact/ragged tails, short single-split and multi-split, and spare empty
    # splits from the capacity bound. q_len>1 is tested as-is, never normalized.
    cases = (
        (256, 30, 1, 0, 15, False),
        (768, 767, 1, 256, 127, True),
        (3072, 2045, 8, 512, 511, False),
        (3072, 2047, 1, 512, 511, True),
    )
    with torch.inference_mode():
        for capacity, past, rows, shift, window, use_sinks in cases:
            def random(shape):
                return (torch.randn(shape, generator=generator, dtype=torch.float16) * 0.25).to(device)

            q = random((1, rows, heads, dim))
            logical_k = random((capacity, kv_heads, dim))
            logical_v = random((capacity, kv_heads, dim))
            new_k = logical_k[past:past + rows].unsqueeze(0).contiguous()
            new_v = logical_v[past:past + rows].unsqueeze(0).contiguous()
            sinks = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32, device=device) if use_sinks else None
            pages = capacity // page_size
            kwargs = dict(causal=True, window_size=(window, 0), sinks=sinks, max_kv_len=capacity)

            def execute(offset, rotation, **partition):
                order = [(i + rotation) % pages for i in range(pages)]
                table = torch.tensor([order], dtype=torch.int32, device=device)
                kc = torch.zeros((pages, page_size, kv_heads, dim), dtype=torch.float16, device=device)
                vc = torch.zeros_like(kc)
                length = past - offset
                for page in range(pages):
                    start = page * page_size
                    count = min(page_size, max(0, length - start))
                    if count:
                        kc[order[page], :count] = logical_k[offset + start:offset + start + count]
                        vc[order[page], :count] = logical_v[offset + start:offset + start + count]
                seq = torch.tensor([length], dtype=torch.int32, device=device)
                output = decode(q, new_k, new_v, kc, vc, table, seq, **kwargs, **partition)
                for row in range(rows):
                    page, lane = divmod(length + row, page_size)
                    assert torch.equal(kc[order[page], lane], new_k[0, row])
                    assert torch.equal(vc[order[page], lane], new_v[0, row])
                assert torch.isfinite(output).all().item()
                return output

            refs = []
            for row in range(rows):
                end = past + row + 1
                start = max(0, end - 1 - window)
                k = logical_k[start:end].float().repeat_interleave(heads // kv_heads, dim=1)
                v = logical_v[start:end].float().repeat_interleave(heads // kv_heads, dim=1)
                scores = torch.einsum("hd,thd->ht", q[0, row].float(), k) * dim ** -0.5
                if sinks is not None:
                    probabilities = torch.cat((scores, sinks[:, None]), dim=1).softmax(dim=1)[:, :-1]
                else:
                    probabilities = scores.softmax(dim=1)
                refs.append(torch.einsum("ht,thd->hd", probabilities, v).half())
            reference = torch.stack(refs).unsqueeze(0)
            default = execute(0, 1)
            explicit_none = execute(0, 1, split_size=None)
            assert torch.equal(default.view(torch.int16), explicit_none.view(torch.int16))
            torch.testing.assert_close(default, reference, atol=0.003, rtol=0.003)
            for size in (256, 512):
                fixed = execute(0, 1, split_size=size)
                repeated = execute(0, 1, split_size=size)
                assert torch.equal(fixed.view(torch.int16), repeated.view(torch.int16))
                torch.testing.assert_close(fixed, reference, atol=0.003, rtol=0.003)
                if shift:
                    assert past - window >= shift
                    rebased = execute(shift, 2, split_size=size)
                    # Fixed key groups do not promise bitwise invariance of the
                    # parallel final reduction or of different q_len layouts.
                    torch.testing.assert_close(rebased, reference, atol=0.003, rtol=0.003)
    torch.cuda.synchronize(device)


def visible_keys(total, q_len, row, causal, window):
    if (type(total) is not int or type(q_len) is not int or type(row) is not int or
            total <= 0 or not 0 < q_len <= total or not 0 <= row < q_len or
            type(causal) is not bool or len(window) != 2):
        raise ValueError('invalid total/query population')
    left, right = window
    if any(type(value) is not int or value < -1 for value in window):
        raise ValueError('invalid window')
    absolute = total - q_len + row
    return [key for key in range(total)
            if (not causal or key <= absolute) and
            (left < 0 or key >= absolute - left) and
            (right < 0 or key <= absolute + right)]


def test_fixed_partitions_mixed_windows_dense_reference(device):
    import pytest
    import torch

    spec = importlib.util.find_spec("exllamav3_ext")
    if spec is None or not spec.origin or not any(
        spec.origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES
    ):
        pytest.skip("Needs a prebuilt native extension; this test never builds it")
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode as decode

    generator = torch.Generator(device='cpu').manual_seed(251026)
    # Mixed window suite: five noncausal cases/14 comparisons plus one
    # causal/right-window intersection control/two comparisons.
    # capacity, past, query rows, rebase distance, causal, (left,right), sinks
    cases = (
        (256, 30, 4, 0, False, (-1, -1), False),
        (1024, 767, 8, 0, False, (127, 3), True),
        (3072, 1537, 8, 512, False, (191, 3), False),
        (1024, 1000, 16, 256, False, (127, -1), True),
        (1024, 767, 8, 0, True, (0, 5), False),
        (1024, 767, 8, 0, False, (-1, 3), True),
    )
    page_size, heads, kv_heads, dim = 256, 8, 2, 96
    with torch.inference_mode():
        for index, (capacity, past, rows, shift, causal, window, use_sinks) in enumerate(cases):
            def random(shape):
                return (torch.randn(shape, generator=generator, dtype=torch.float16) * 0.25).to(device)
            q = random((1, rows, heads, dim))
            logical_k = random((capacity, kv_heads, dim))
            logical_v = random((capacity, kv_heads, dim))
            k = logical_k[past:past + rows].unsqueeze(0).contiguous()
            v = logical_v[past:past + rows].unsqueeze(0).contiguous()
            sinks = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32, device=device) if use_sinks else None
            total = past + rows
            references = []
            key_sets = []
            for row in range(rows):
                keys = visible_keys(total, rows, row, causal, window)
                if not keys:
                    raise RuntimeError('fixture has no visible real keys')
                if shift and any(key < shift for key in keys):
                    raise RuntimeError('rebase would discard visible keys')
                key_sets.append(keys)
                ks = logical_k[keys].float().repeat_interleave(heads // kv_heads, dim=1)
                vs = logical_v[keys].float().repeat_interleave(heads // kv_heads, dim=1)
                scores = torch.einsum('hd,thd->ht', q[0, row].float(), ks) * dim ** -0.5
                if sinks is not None:
                    probabilities = torch.cat((scores, sinks[:, None]), dim=1).softmax(dim=1)[:, :-1]
                else:
                    probabilities = scores.softmax(dim=1)
                references.append(torch.einsum('ht,thd->hd', probabilities, vs).half())
            reference = torch.stack(references).unsqueeze(0)
            pages = capacity // page_size
            for size in (256, 512):
                for offset in ([0, shift] if shift else [0]):
                    order = [(page + (2 if offset else 1)) % pages for page in range(pages)]
                    kc = torch.full((pages, page_size, kv_heads, dim), 9.0, dtype=torch.float16, device=device)
                    vc = torch.full_like(kc, -9.0)
                    length = past - offset
                    for page in range(pages):
                        start = page * page_size
                        count = min(page_size, max(0, length - start))
                        if count:
                            kc[order[page], :count] = logical_k[offset + start:offset + start + count]
                            vc[order[page], :count] = logical_v[offset + start:offset + start + count]
                    table = torch.tensor([order], dtype=torch.int32, device=device)
                    seq = torch.tensor([length], dtype=torch.int32, device=device)
                    actual = decode(q, k, v, kc, vc, table, seq, causal=causal, window_size=window,
                                    sinks=sinks, max_kv_len=capacity, split_size=size)
                    if not bool(torch.isfinite(actual).all()):
                        raise RuntimeError('nonfinite production output')
                    for row in range(rows):
                        page, lane = divmod(length + row, page_size)
                        if not torch.equal(kc[order[page], lane], k[0, row]) or not torch.equal(vc[order[page], lane], v[0, row]):
                            raise RuntimeError('append output differs')
                    torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.003)
    torch.cuda.synchronize(device)
